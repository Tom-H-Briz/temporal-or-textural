"""MSRVTT 1k-A zero-shot retrieval gate for UMT-B/16 25M (umt25m_brief_241126.md §2).
Official protocol (exp/zero_shot/ret_msrvtt/b16.py + tasks/retrieval_utils.py): 4 frames,
middle sampling, 224 px, max_txt_l=32, VTC ranking (criterions.get_sim), VTM rerank of the
VTC top k_test=128 (unfilled entries -100), eval_frame_ensemble='concat'. Reports T2V + V2T
R@1/5/10, VTC-only and VTM-reranked. Gate: VTM-reranked T2V R@1 within ±2 pt of published 35.2.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import InterpolationMode, Resize

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "notebooks")]
from umt_wrapper import UMTProcessor, sample_frames_umt          # noqa: E402
from umt25m_wrapper import build_umt25m                          # noqa: E402

SCRATCH = Path(os.environ.get("SCRATCHDIR", ROOT / "data"))
CFG = {
    "anno":       Path(os.environ.get("MSRVTT_ANNO", SCRATCH / "data/msrvtt/anno/msrvtt_ret_test1k.json")),
    "video_dir":  Path(os.environ.get("MSRVTT_VIDEO_DIR", SCRATCH / "data/msrvtt")),
    "ckpt":       Path(os.environ.get("UMT25M_CKPT", ROOT / "models/umt_ckpts/b16_25m.pth")),
    "out_dir":    ROOT / "outputs/analysis/umt25m_msrvtt_gate",
    "num_frames": 4, "img_size": 224, "max_txt_l": 32, "k_test": 128,
    "batch_size": 32, "text_bs": 256,   # itm batching handled inside rerank (math identical)
    "limit": int(os.environ.get("LIMIT", 0)),         # >0: smoke on the first LIMIT test videos
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gate_ref": 35.2, "gate_margin": 2.0,             # user decision 24/11/26: ±2 pt (~1.3 SE)
}


def pre_text(text):
    """Copied from dataset/utils.py pre_text (official caption cleaning; every caption
    goes through it, so feeding raw captions would change the queries)."""
    text = re.sub(r"([,.'!?\"()*#:;~])", '', text.lower())
    text = text.replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
    text = re.sub(r"\s{2,}", ' ', text)
    return text.rstrip('\n').strip(' ')


def load_annotations(path, limit=0):
    """VINDLU ret json: entries carry the media id under 'image' (even for video; see
    caption_dataset.py) or 'video' — real test1k ids already end in '.mp4', normalised to
    a stem here so resolve_videos adds the extension exactly once. caption is str or list
    (each caption = one text row, 1:1 video for test1k). Returns (video_ids, texts,
    txt2img, img2txt)."""
    raw = json.load(open(path))
    if isinstance(raw, dict):
        raw = raw.get("annotations") or next(v for v in raw.values() if isinstance(v, list))
    if not (isinstance(raw, list) and raw and isinstance(raw[0], dict)):
        raise ValueError(f"{path}: expected list of dicts, got {type(raw).__name__}")
    if limit:
        raw = raw[:limit]
    rows = []
    for e in raw:
        vid = Path(e.get("video", e.get("image"))).stem
        if not vid or "caption" not in e:
            raise ValueError(f"{path}: entry missing video/image or caption; keys={sorted(e)}")
        caps = e["caption"] if isinstance(e["caption"], list) else [e["caption"]]
        rows += [(vid, pre_text(c)) for c in caps]
    vids = sorted({v for v, _ in rows})
    v2i = {v: i for i, v in enumerate(vids)}
    txt2img = [v2i[vid] for vid, _ in rows]
    img2txt = {}
    for t, i in enumerate(txt2img):
        img2txt.setdefault(i, []).append(t)
    return vids, [c for _, c in rows], txt2img, img2txt


def resolve_videos(video_dir, vids):
    """{vid: mp4 path}; direct hit first, one recursive scan as fallback (zip may nest a dir)."""
    paths, scan = {}, []
    for v in vids:
        hit = video_dir / f"{v}.mp4"
        if hit.exists():
            paths[v] = hit
        else:
            scan.append(v)
    if scan:
        found = {p.stem: p for p in video_dir.rglob("*.mp4")}
        missing = [v for v in scan if v not in found]
        if missing:
            raise FileNotFoundError(f"{len(missing)}/{len(vids)} videos missing under {video_dir}; e.g. {missing[:3]}")
        paths.update({v: found[v] for v in scan})
        print(f"resolved {len(scan)} videos via recursive scan under {video_dir}")
    return [paths[v] for v in vids]


def decode_frames(path, num_frames):
    """Official read_frames_av path: decode every frame, then get_frame_indices('middle')."""
    with av.open(str(path)) as c:
        frames = [f.to_rgb().to_ndarray() for f in c.decode(video=0)]
    idx = sample_frames_umt(len(frames), num_frames)
    return np.stack([frames[i] for i in idx])


@torch.no_grad()
def encode_texts(model, tok, texts):
    """retrieval_utils.extract_text_feats (text_bs=256, padding='max_length', truncation).
    Returns l8 hidden states, add_mask and projected CLS, all on CFG['device']."""
    out_h, out_m, out_p = [], [], []
    for s in range(0, len(texts), CFG["text_bs"]):
        enc = tok(texts[s:s + CFG["text_bs"]], padding="max_length", truncation=True,
                  max_length=CFG["max_txt_l"], return_tensors="pt").to(CFG["device"])
        h, m = model.encode_text(enc.input_ids, enc.attention_mask)
        out_h.append(h)
        out_m.append(m)
        out_p.append(model.text_proj(h[:, 0]))
    return torch.cat(out_h), torch.cat(out_m), torch.cat(out_p)


@torch.no_grad()
def encode_videos(model, paths):
    """retrieval_utils.extract_vision_feats with eval_offload: x_vis (cross-attn input)
    offloaded to cpu; per-frame projected embeddings kept for VTC."""
    proc = UMTProcessor()
    xs, projs = [], []
    for s in range(0, len(paths), CFG["batch_size"]):
        chunk = [proc(decode_frames(p, CFG["num_frames"]))["pixel_values"][0]
                 for p in paths[s:s + CFG["batch_size"]]]
        x_vis, pooled = model.encode_video(torch.stack(chunk).to(CFG["device"]))
        xs.append(x_vis.cpu())
        projs.append(model.vision_proj(pooled).cpu())
        if (s // CFG["batch_size"]) % 10 == 0:
            print(f"  videos {s + len(chunk)}/{len(paths)}", flush=True)
    return torch.cat(xs), torch.cat(projs)


@torch.no_grad()
def rerank(model, l8_h, add_mask, x_vis, sim, k):
    """retrieval_utils.evaluation rerank: each row's VTC top-k pairs go through the fusion
    encoder (official itm batch 32 via vtm_score's pair dimension), other entries stay -100.
    Returns (i2t_x (Nv, Nt), t2i_x (Nt, Nv))."""
    n_v, n_t = sim.shape
    i2t_x = torch.full((n_v, n_t), -100.0, device=sim.device)
    t2i_x = torch.full((n_t, n_v), -100.0, device=sim.device)
    for i in range(n_v):
        idx = sim[i].topk(min(k, n_t)).indices
        vis_i = x_vis[i].to(sim.device)
        score = model.vtm_score(l8_h[idx], add_mask[idx], vis_i[None].expand(len(idx), -1, -1))
        i2t_x[i, idx] = score.float()
    for j in range(n_t):
        idx = sim[:, j].topk(min(k, n_v)).indices
        pairs = l8_h[j][None].expand(len(idx), -1, -1)
        mask = add_mask[j][None].expand(len(idx), -1, -1, -1)
        score = model.vtm_score(pairs, mask, x_vis[idx].to(sim.device))
        t2i_x[j, idx] = score.float()
    return i2t_x.cpu().numpy(), t2i_x.cpu().numpy()


def _ranks(scores, gt):
    """Official itm_eval rank: position of the best ground-truth id under descending sort."""
    r = np.zeros(scores.shape[0])
    for i, s in enumerate(scores):
        truth = gt[i] if isinstance(gt[i], list) else [gt[i]]
        r[i] = min(np.where(np.argsort(s)[::-1] == g)[0][0] for g in truth)
    return r


def metrics(i2t, t2i, txt2img, img2txt):
    """tasks/retrieval_utils.itm_eval: txt_r* = V2T (per video, rank of its text),
    img_r* = T2V (per text, rank of its video)."""
    tr = _ranks(np.asarray(i2t), img2txt)
    ir = _ranks(np.asarray(t2i), txt2img)
    pct = lambda r, k: round(100.0 * float((r < k).mean()), 2)
    return dict(v2t_r1=pct(tr, 1), v2t_r5=pct(tr, 5), v2t_r10=pct(tr, 10),
                t2v_r1=pct(ir, 1), t2v_r5=pct(ir, 5), t2v_r10=pct(ir, 10))


def ranks_frame(sim_np, i2t_x, t2i_x, txt2img, img2txt):
    """Per-query ranks (1-based) for both protocols and both directions, for the parquet."""
    parts = []
    for name, i2t, t2i in (("vtc", sim_np, sim_np.T), ("vtm", i2t_x, t2i_x)):
        parts.append(pd.DataFrame(dict(
            protocol=name, direction="v2t", query_id=np.arange(len(i2t)),
            gt_id=[g[0] if isinstance(g, list) else g for g in img2txt],
            rank=_ranks(i2t, img2txt) + 1)))
        parts.append(pd.DataFrame(dict(
            protocol=name, direction="t2v", query_id=np.arange(len(t2i)),
            gt_id=list(txt2img), rank=_ranks(t2i, txt2img) + 1)))
    return pd.concat(parts, ignore_index=True)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = CFG["out_dir"]
    if CFG["limit"]:
        out = out.with_name(out.name + "_smoke")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    vids, texts, txt2img, img2txt = load_annotations(CFG["anno"], CFG["limit"])
    paths = resolve_videos(CFG["video_dir"], vids)
    print(f"{len(vids)} videos / {len(texts)} captions  anno={CFG['anno']}  device={CFG['device']}")
    model = build_umt25m(CFG["ckpt"], CFG["num_frames"], CFG["device"])
    l8_h, add_mask, tproj = encode_texts(model, tok, texts)
    x_vis, vproj = encode_videos(model, paths)
    sim = model.vtc_logits(vproj.to(CFG["device"]), tproj)
    sim_np = sim.float().cpu().numpy()
    m_vtc = metrics(sim_np, sim_np.T, txt2img, img2txt)
    i2t_x, t2i_x = rerank(model, l8_h, add_mask, x_vis, sim, CFG["k_test"])
    m_vtm = metrics(i2t_x, t2i_x, txt2img, img2txt)
    ranks = ranks_frame(sim_np, i2t_x, t2i_x, txt2img, img2txt)
    try:
        ranks.to_parquet(out / "ranks.parquet", index=False)
    except ImportError:   # local smoke envs without pyarrow; Isambard writes parquet
        ranks.to_csv(out / "ranks.csv", index=False)
    gate_val = m_vtm["t2v_r1"]
    summary = dict(
        cfg={k: str(v) for k, v in CFG.items()}, n_videos=len(vids), n_texts=len(texts),
        wall_min=round((time.time() - t0) / 60, 1), vtc=m_vtc, vtm=m_vtm,
        gate=dict(metric="vtm t2v_r1", ref=CFG["gate_ref"], margin=CFG["gate_margin"], value=gate_val,
                  passed=bool(abs(gate_val - CFG["gate_ref"]) <= CFG["gate_margin"])))
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(pd.DataFrame([dict(protocol="vtc-only", **m_vtc), dict(protocol="vtm-reranked", **m_vtm)]).to_string(index=False))
    print(f"GATE: VTM T2V R@1 {gate_val} vs 35.2 ±2 pt -> {'PASS' if summary['gate']['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
