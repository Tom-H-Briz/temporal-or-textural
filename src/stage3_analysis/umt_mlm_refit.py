"""UMT MLM head refit: plain-torch port of the full MLM text path, feature cache,
head training and evaluation, per brief mlm_head_refit_brief_151026.md.

Modes (uv run python src/stage3_analysis/umt_mlm_refit.py <mode>):
  port     local CPU: self-check the port vs notebooks/umt_mlm_ref.npz (xbert, both
           BERTs) and reproduce umt_mlm_graft_check.py (0.399 pretrain / 0.113 graft,
           null video) on its cached ids.
  extract  GPU/SLURM: cache per-blank layer-11 states h (fp16) for train R and
           val R/null/Rev (real video, one zero vision token, reversed frames).
  train    on the cache: refit the pretrain-initialised MLM head (623k params,
           decoder tied to the finetuned word embeddings, frozen) with AdamW.
  eval     val: metrics.csv (model x condition x word type), per_blank_val.parquet
           with probe-word log-probs, head_refit.pt alongside in train mode.

Usage: MAX_TRAIN_CLIPS=500 MAX_VAL_CLIPS=200 uv run python src/stage3_analysis/umt_mlm_refit.py all
"""
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src" / "stage3_analysis")]
from umt_vtm_gate import _attend, _fusion_layer, _kv, _lin   # verified plain-torch xbert ops

CFG = {
    "pretrain_ckpt":  Path("models/umt_ckpts/b16_25m.pth"),          # local, else HF
    "finetune_ckpt":  Path("models/umt_ckpts/ret_ssv2_tpl_b16_25m.pth"),
    "hf_repo":        "OpenGVLab/UMT",
    "hf_pretrain":    "multi_modality/b16_25m.pth",
    "hf_finetune":    "multi_modality/ret_ssv2_tpl_b16_25m.pth",
    "ref":            ROOT / "notebooks/umt_mlm_ref.npz",            # from umt_mlm_ref_cache.py
    "labels_path":    os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "train_path":     os.environ.get("TRAIN_PATH", str(ROOT / "data/ssv2/labels/train.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":      os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/videos")),
    "max_train_clips": int(os.environ.get("MAX_TRAIN_CLIPS", 0) or 0),
    "max_val_clips":   int(os.environ.get("MAX_VAL_CLIPS", 0) or 0),
    "max_txt_l":      25,                # the finetune's max_txt_l (exp/finetuning/ret_ssv2_tpl)
    "out_dir":        ROOT / "outputs/analysis/umt_mlm_refit",
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size":     8, "num_workers": 4,
    # training (D3): lr/batch picked for a 623k-param linear-ish head on ~1M rows
    "lr": 1e-3, "weight_decay": 0.01, "train_bs": 8192, "epochs": 60,
    "patience": 5, "holdout": 0.05, "seed": 0,
    "wandb_project":  "temporal-or-textural",
    # probe words (D4): dropped + logged if not a single word-piece
    "probe_words": ["up", "down", "left", "right", "closer", "away", "towards", "into",
                    "out", "open", "close", "behind", "front", "apart", "together",
                    "over", "under", "around", "across", "past", "onto", "through"],
    "eval_bs":        1024,
}


# ---- weights ----
@lru_cache(maxsize=None)
def load_sd(which):
    """Checkpoint state dict, local path first, else the HF copy (Isambard).
    Cached: bert_weights/head_weights call it repeatedly within one run."""
    p = CFG[f"{which}_ckpt"]
    if not p.exists():
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(CFG["hf_repo"], CFG[f"hf_{which}"])
    return torch.load(p, map_location="cpu", weights_only=False)


def bert_weights(which):
    """BERT weights as finetuned-layout 'text_encoder.*' keys (pretrain: strip
    'text_encoder.bert.'), float32, cpu."""
    sd = load_sd(which)
    pre = "text_encoder.bert." if which == "pretrain" else "text_encoder."
    w = {("text_encoder." + k[len(pre):]): v.float() for k, v in sd.items() if k.startswith(pre)}
    w.pop("text_encoder.embeddings.position_ids", None)
    return w


def head_weights():
    """Pretrain MLM head: transform dense + LayerNorm + bias (decoder weight is tied
    to word_embeddings at call time, so it is not stored here)."""
    sd = load_sd("pretrain")
    t = "text_encoder.cls.predictions.transform."
    hw = {k: sd[t + k].float() for k in ("dense.weight", "dense.bias",
                                         "LayerNorm.weight", "LayerNorm.bias")}
    hw["bias"] = sd["text_encoder.cls.predictions.bias"].float()
    return hw


# ---- plain-torch MLM text path (mirrors xbert BertModel, fusion_layer=9, post-LN) ----
def bert_embed(ids, w):
    """word + token_type(0) + position embeddings -> LayerNorm(768, eps 1e-12)."""
    p = "text_encoder.embeddings."
    e = w[p + "word_embeddings.weight"][ids] + w[p + "token_type_embeddings.weight"][0] \
        + w[p + "position_embeddings.weight"][: ids.shape[1]]
    return F.layer_norm(e, (768,), w[p + "LayerNorm.weight"], w[p + "LayerNorm.bias"], 1e-12)


def text_layer(h, add_mask, w, i):
    """Layers 0-8 ('text' mode): self-attention -> FFN, no cross-attention."""
    p = f"text_encoder.encoder.layer.{i}"
    h = _attend(h, *_kv(h, w, p + ".attention"), w, p + ".attention", add_mask)
    ff = _lin(F.gelu(_lin(h, w, p + ".intermediate.dense")), w, p + ".output.dense")
    return F.layer_norm(ff + h, (768,), w[p + ".output.LayerNorm.weight"],
                        w[p + ".output.LayerNorm.bias"], 1e-12)


def mlm_text(ids, attn, w):
    """Embeddings + layers 0-8 ('text' mode): video-independent half of the path.
    -> (h8, additive attention mask for the padding positions)."""
    h = bert_embed(ids, w)
    add = ((1.0 - attn.float()) * -10000.0)[:, None, None, :]
    for i in range(9):
        h = text_layer(h, add, w, i)
    return h, add


def mlm_fusion(h8, add, vis, w):
    """Layers 9-11: cross-attention from text to vis (1, n_vis, 768) -> layer-11 states."""
    cross = {i: _kv(vis, w, f"text_encoder.encoder.layer.{i}.crossattention") for i in (9, 10, 11)}
    for i in (9, 10, 11):
        h8 = _fusion_layer(h8, add, cross[i], w, i)
    return h8


def mlm_hidden(ids, attn, vis, w):
    """input_ids (N, L), attn (N, L) 0/1, vis (1, n_vis, 768) -> layer-11 states.
    All tensors must already be on the same device as w."""
    h, add = mlm_text(ids, attn, w)
    return mlm_fusion(h, add, vis, w)


class RefitHead(torch.nn.Module):
    """The MLM head (xbert BertLMPredictionHead): dense -> GELU -> LayerNorm ->
    decoder tied to word_embeddings + bias. As a port: weights given as the
    'text_encoder.cls.predictions.*' dict; as trained: only dense, LayerNorm and
    bias are parameters (~623k), the decoder weight is a frozen buffer."""

    def __init__(self, hw, word_emb):
        super().__init__()
        self.dense = torch.nn.Linear(768, 768)
        self.ln = torch.nn.LayerNorm(768, eps=1e-12)
        self.bias = torch.nn.Parameter(hw["bias"].clone())
        with torch.no_grad():
            self.dense.weight.copy_(hw["dense.weight"])
            self.dense.bias.copy_(hw["dense.bias"])
            self.ln.weight.copy_(hw["LayerNorm.weight"])
            self.ln.bias.copy_(hw["LayerNorm.bias"])
        self.register_buffer("word_emb", word_emb)

    def forward(self, h):
        return self.ln(F.gelu(self.dense(h))) @ self.word_emb.T + self.bias


# ---- D1: self-checks (local, CPU) ----
def get_tokenizer():
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained("bert-base-uncased")


def encode(tok, text, max_l):
    """UMT/xbert single-sequence convention: [CLS] + pieces, NO trailing [SEP]
    (vendored tokenization_bert builds '[CLS] X'). -> (ids, truncated)."""
    pieces = tok(text, add_special_tokens=False)["input_ids"]
    return [tok.cls_token_id] + pieces[: max_l - 1], len(pieces) > max_l - 1


def tokenizer_check(tok):
    """Pipeline-env BertTokenizer (via encode) vs xbert's cached ids: all 174
    templates, the ref captions' masked variants, and the rebuilt graft set."""
    c = np.load(CFG["ref"], allow_pickle=True)
    tpl = [" ".join(t.lower().replace("[", "").replace("]", "").split())
           for t in json.load(open(CFG["labels_path"]))]
    assert tok(tpl, add_special_tokens=False)["input_ids"] == [list(r) for r in c["tpl_ids"]]
    ids = []
    for cap in c["ref_caps"]:
        enc, _ = encode(tok, str(cap).lower(), 32)
        ids += [enc[:p] + [tok.mask_token_id] + enc[p + 1:] for p in range(1, len(enc) - 1)]
    assert ids == [r[r != 0].tolist() for r in c["ref_ids"]], "ref-caption ids differ"
    rows = json.load(open(CFG["validation_path"]))
    picked = np.random.default_rng(0).choice(rows, 300, replace=False)
    g_ids = []
    for r in picked:
        enc, _ = encode(tok, r["label"].lower(), 32)
        g_ids += [enc[:p] + [tok.mask_token_id] + enc[p + 1:] for p in range(1, len(enc) - 1)]
    assert g_ids == [r[r != 0].tolist() for r in c["graft_ids"]], "graft ids differ"
    print("tokenizer check: transformers-5.5 ids == xbert ids (templates + ref + graft)")


def port_check():
    """Port vs the xbert reference: MLM logits at the masked positions, both BERTs."""
    c = np.load(CFG["ref"])
    vis = torch.from_numpy(c["ref_vis"]).unsqueeze(0)
    pos_t = torch.from_numpy(c["ref_pos"])
    for name, which, key in [("pretrain", "pretrain", "ref_logits_pretrain"),
                             ("graft (finetuned bert)", "finetune", "ref_logits_graft")]:
        w, hw = bert_weights(which), head_weights()
        head = RefitHead(hw, w["text_encoder.embeddings.word_embeddings.weight"])
        h = mlm_hidden(torch.from_numpy(c["ref_ids"]), torch.from_numpy(c["ref_attn"]), vis, w)
        logits = head(h[torch.arange(len(h)), pos_t])
        err = (logits - torch.from_numpy(c[key])).abs().max().item()
        print(f"port self-check vs xbert, {name}: max |diff| = {err:.2e}")
        assert err < 1e-3, f"port does not reproduce xbert for {name}"


@torch.no_grad()
def graft_repro():
    """Reproduce umt_mlm_graft_check.py with the port: its cached ids, null video
    (one zero vision token), batch 64. Returns {model: top-1 over all blanks}."""
    c = np.load(CFG["ref"])
    gx = torch.from_numpy(c["graft_ids"])
    pos, true = c["graft_pos"], c["graft_true"]
    is_obj = c["graft_is_obj"].astype(bool)
    out = {}
    for name, which, target in [("pretrain", "pretrain", 0.399), ("graft", "finetune", 0.113)]:
        w, hw = bert_weights(which), head_weights()
        head = RefitHead(hw, w["text_encoder.embeddings.word_embeddings.weight"])
        hit, logp = [], []
        for s in range(0, len(gx), 64):
            xb = gx[s:s + 64]
            h = mlm_hidden(xb, (xb != 0).long(), torch.zeros(1, 1, 768), w)
            lp = head(h[torch.arange(len(h)), torch.tensor(pos[s:s + 64])]).log_softmax(-1)
            t_t = torch.tensor(true[s:s + 64])
            hit.append((lp.argmax(-1) == t_t).numpy())
            logp.append(lp[torch.arange(len(t_t)), t_t].numpy())
        hit, logp = np.concatenate(hit), np.concatenate(logp)
        for part, m in [("all", slice(None)), ("object", is_obj), ("other", ~is_obj)]:
            print(f"  {name:8s} {part:7s} top-1 {hit[m].mean():.3f}  mean log-p {logp[m].mean():.2f}")
        assert abs(hit.mean() - target) <= 0.002, f"{name} {hit.mean():.4f} != {target}"
        out[name] = float(hit.mean())
    return out


# ---- D2: feature extraction ----
def umt_pre_text():
    """UMT's caption cleaner (dataset/utils.py), by path. Its 'from utils.distributed
    import ...' needs multi_modality on sys.path (same as umt_vtm_text_cache's
    PYTHONPATH)."""
    import importlib.util
    sys.path[:0] = [str(ROOT / "models/unmasked_teacher/multi_modality")]
    spec = importlib.util.spec_from_file_location(
        "umt_ds_utils", ROOT / "models/unmasked_teacher/multi_modality/dataset/utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.pre_text


def caption_sets(split, tok):
    """Per split (JSON order, after MAX_*_CLIPS): one dict per clip with id, template,
    label, enc ids ([CLS]+pieces, no SEP, max_txt_l), is_obj per position, truncated.
    Clips whose webm is missing are dropped and counted."""
    path = CFG["train_path"] if split == "train" else CFG["validation_path"]
    rows = json.load(open(path))
    n_max = CFG["max_train_clips"] if split == "train" else CFG["max_val_clips"]
    if n_max:
        rows = rows[:n_max]
    pre_text = umt_pre_text()
    clips, dropped = [], 0
    for r in rows:
        if not (Path(CFG["video_dir"]) / f"{r['id']}.webm").exists():
            dropped += 1
            continue
        enc, trunc = encode(tok, pre_text(r["label"]), CFG["max_txt_l"])
        obj = set(tok(pre_text(" ".join(r["placeholders"])), add_special_tokens=False)["input_ids"])
        clips.append(dict(id=str(r["id"]), template=r["template"], label=r["label"], enc=enc,
                          is_obj=[enc[p] in obj for p in range(1, len(enc))], truncated=trunc))
    print(f"{split}: {len(clips):,} clips ({dropped} without video), "
          f"truncated {np.mean([c['truncated'] for c in clips]) if clips else 0:.4f}")
    return clips


def vision_loader(clips):
    """UMT vision tower + R loader: same dataset class, sampler and processor as
    umt_vtm_gate.load_model_and_data. Labels are the clip index into `clips`, so
    extract can tell which clip a batch item is after a decode retry moves on."""
    from ToT_utils import MODEL_REGISTRY, get_frame_sampler
    from perturb_accuracy_tf_kinetics import PerturbedKineticsDataset
    from umt_wrapper import UMTClassifier, UMTProcessor
    model = UMTClassifier.from_pretrained().to(CFG["device"]).eval()
    mcfg = MODEL_REGISTRY["umt"]
    paths = [Path(CFG["video_dir"]) / f"{c['id']}.webm" for c in clips]
    ds = PerturbedKineticsDataset(paths, [c["id"] for c in clips], list(range(len(clips))),
                                  UMTProcessor(), mcfg["num_frames"],
                                  get_frame_sampler("ssv2", mcfg), "R")
    loader = torch.utils.data.DataLoader(ds, batch_size=CFG["batch_size"],
                                         num_workers=CFG["num_workers"])
    return model, loader


@torch.no_grad()
def clip_variants(enc, mask_id):
    """All masked variants of one caption (positions 1..len-1, no [SEP] to skip):
    -> (ids (n, L), attn, positions). All variants share one length: no padding."""
    pos = list(range(1, len(enc)))
    ids = torch.tensor([[*enc[:p], mask_id, *enc[p + 1:]] for p in pos], device=CFG["device"])
    return ids, torch.ones_like(ids), torch.tensor(pos, device=CFG["device"])


def blank_layout(clips):
    """Row layout over all masked variants: blank start per clip + per-row meta
    (target, position, clip index, is_obj). Row order: clip-major, position ascending."""
    starts = np.concatenate([[0], np.cumsum([len(c["enc"]) - 1 for c in clips])]).astype(np.int64)
    meta = dict(target=np.concatenate([[c["enc"][p] for p in range(1, len(c["enc"]))]
                                       for c in clips]).astype(np.int32),
                pos=np.concatenate([np.arange(1, len(c["enc"])) for c in clips]).astype(np.int16),
                clip_idx=np.repeat(np.arange(len(clips)), np.diff(starts)).astype(np.int32),
                is_obj=np.concatenate([c["is_obj"] for c in clips]))
    return starts, meta


@torch.no_grad()
def extract(split, cache_dir):
    """Cache per-blank layer-11 states h (fp16): train -> R; val -> R, null, Rev.
    Row order: clip-major, masked position ascending; identical across conditions.
    The loader's returned label is the decoded clip's index (decode retries move
    on), so corrupt clips are detected, skipped and reported, never mispaired."""
    tok = get_tokenizer()
    clips = [c for c in caption_sets(split, tok) if len(c["enc"]) > 1]
    conds = ("R",) if split == "train" else ("R", "null", "Rev")
    starts, meta = blank_layout(clips)
    h_buf = {c: np.zeros((int(starts[-1]), 768), np.float16) for c in conds}
    model, loader = vision_loader(clips)
    w = {k: v.to(CFG["device"]) for k, v in bert_weights("finetune").items()}
    mask_id, dev = tok.mask_token_id, CFG["device"]
    null_vis = torch.zeros(1, 1, 768, device=dev)
    filled = np.zeros(len(clips), bool)
    n_dup, t_vis, t_txt = 0, 0.0, 0.0
    for pixel_values, ys in loader:
        _sync(); t0 = time.time()
        pv = pixel_values.to(dev)
        x_vis = model.vision(pv.permute(0, 2, 1, 3, 4), keep_temporal=True)[0]
        x_rev = model.vision(pv.flip(1).permute(0, 2, 1, 3, 4), keep_temporal=True)[0] \
            if "Rev" in conds else None
        _sync(); t_vis, t1 = t_vis + time.time() - t0, time.time()
        for b, y in enumerate(ys.tolist()):
            if filled[y]:                       # decode-retry duplicate of an
                n_dup += 1                      # already-written clip: skip
                continue
            c, sl = clips[y], slice(starts[y], starts[y + 1])
            ids, attn, pos_t = clip_variants(c["enc"], mask_id)
            h8, add = mlm_text(ids, attn, w)    # video-independent: once per clip
            for cond in conds:
                vis = null_vis if cond == "null" else \
                    (x_vis[b] if cond == "R" else x_rev[b]).unsqueeze(0)
                h = mlm_fusion(h8, add, vis, w)[torch.arange(len(pos_t)), pos_t]
                h_buf[cond][sl] = h.cpu().half().numpy()
            filled[y] = True
        _sync(); t_txt += time.time() - t1
        n_done = int(filled.sum())
        if n_done % 800 < CFG["batch_size"] or n_done == len(clips):
            print(f"  {split} {n_done:,}/{len(clips):,} clips  "
                  f"vision+decode {t_vis / max(n_done, 1) * 1e3:.0f} ms/clip  "
                  f"text {t_txt / max(n_done, 1) * 1e3:.0f} ms/clip", flush=True)
    save_cache(split, clips, conds, h_buf, meta, filled, n_dup, t_vis, t_txt, cache_dir)


def save_cache(split, clips, conds, h_buf, meta, filled, n_dup, t_vis, t_txt, cache_dir):
    """Write the caches: rows of undecoded clips dropped everywhere; clips parquet
    keeps every clip with an ok flag; report drops and the full-run projection."""
    corrupt = np.where(~filled)[0]
    if len(corrupt):
        print(f"WARNING {split}: {len(corrupt)} clips never decoded (rows dropped): "
              f"{[clips[y]['id'] for y in corrupt[:10]]}{'...' if len(corrupt) > 10 else ''}")
    keep = filled[meta["clip_idx"]]
    for cond in conds:
        np.savez(cache_dir / f"cache_{split}_{cond}.npz", h=h_buf[cond][keep],
                 **{k: v[keep] for k, v in meta.items()})
    pd.DataFrame([dict(clip_id=c["id"], template=c["template"], label=c["label"],
                       truncated=c["truncated"], n_blanks=len(c["enc"]) - 1,
                       enc=c["enc"], ok=bool(filled[j])) for j, c in enumerate(clips)]
                 ).to_parquet(cache_dir / f"clips_{split}.parquet", index=False)
    n_full = 168913 if split == "train" else 24777
    print(f"{split} done: {int(keep.sum()):,} blanks kept ({n_dup} duplicate loads), "
          f"cache {sum(h_buf[c].nbytes for c in conds) / 1e9:.2f} GB; "
          f"projected full split (compute only) "
          f"{(t_vis + t_txt) / max(int(filled.sum()), 1) * n_full / 3600:.1f} h")


def _sync():
    if CFG["device"] == "cuda":
        torch.cuda.synchronize()


# ---- D3: head training on the cache ----
@torch.no_grad()
def head_eval(head, h, tgt, probe=np.zeros(0, np.int64)):
    """Per-row reduction of head(h): pred, top1, logp of the true token, and
    log-probs of the probe token ids. Reduces on the GPU; only (N,) / (N, n_probe)
    results cross to the CPU."""
    N = len(h)
    out = dict(pred=np.zeros(N, np.int64), logp=np.zeros(N, np.float32), top1=np.zeros(N, bool),
               lp_probe=np.zeros((N, len(probe)), np.float32))
    for s in range(0, N, CFG["eval_bs"]):
        e = min(s + CFG["eval_bs"], N)
        lp = head(h[s:e].to(CFG["device"]).float()).log_softmax(-1)
        t = tgt[s:e].long().to(CFG["device"])
        out["pred"][s:e] = lp.argmax(-1).cpu().numpy()
        out["logp"][s:e] = lp.gather(1, t[:, None]).squeeze(1).cpu().numpy()
        out["top1"][s:e] = out["pred"][s:e] == t.cpu().numpy()
        if len(probe):
            out["lp_probe"][s:e] = lp[:, torch.from_numpy(probe).to(lp.device)].cpu().numpy()
    return out


def holdout_split(clip_idx):
    """5% of clips held out (by clip, seed 0) -> blank-level boolean mask."""
    clips = np.unique(clip_idx)
    rng = np.random.default_rng(CFG["seed"])
    held = rng.choice(clips, max(1, int(len(clips) * CFG["holdout"])), replace=False)
    return np.isin(clip_idx, held)


def train_epoch(head, opt, h, target, idx):
    """One pass over the shuffled blank indices. -> (mean CE loss, top-1) from the
    training batches themselves (no re-scoring of the training set)."""
    tot, hit, n = 0.0, 0, 0
    for s in range(0, len(idx), CFG["train_bs"]):
        b = idx[s:s + CFG["train_bs"]]
        logits = head(h[b].to(CFG["device"]).float())
        t = target[b].to(CFG["device"])
        loss = F.cross_entropy(logits, t)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
        hit += int((logits.argmax(-1) == t).sum().item())
        n += len(b)
    return tot / n, hit / n


def train_head(cache_dir, out_dir, smoke):
    """Refit on cached train-R h; early stopping on held-out loss. Val is never
    used here."""
    import wandb
    c = np.load(cache_dir / "cache_train_R.npz")
    h = torch.from_numpy(c["h"])
    target = torch.from_numpy(c["target"]).long()
    is_held = torch.from_numpy(holdout_split(c["clip_idx"]))
    emb_fin = bert_weights("finetune")["text_encoder.embeddings.word_embeddings.weight"]
    head = RefitHead(head_weights(), emb_fin).to(CFG["device"])
    n_par = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"trainable head params {n_par:,} (expect 622,650); "
          f"{int((~is_held).sum()):,} train / {int(is_held.sum()):,} held-out blanks "
          f"from {len(np.unique(c['clip_idx'])):,} clips")
    assert n_par == 622_650
    assert torch.equal(head.word_emb.cpu(), emb_fin)
    opt = torch.optim.AdamW(head.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    tr_idx, he_idx = torch.where(~is_held)[0].numpy(), torch.where(is_held)[0]
    try:
        wandb.init(project=CFG["wandb_project"], name=f"umt_mlm_refit{_smoke_tag(smoke)}",
                   config={k: str(v) for k, v in CFG.items() if k != "probe_words"})
    except Exception:
        wandb.init(mode="disabled")
    curve, best, best_ep, bad = [], float("inf"), -1, 0
    for ep in range(CFG["epochs"]):
        t0 = time.time()
        perm = np.random.default_rng(CFG["seed"] + ep).permutation(tr_idx)
        head.train()
        tr_loss, tr_top1 = train_epoch(head, opt, h, target, perm)
        m_he = head_eval(head.eval(), h[he_idx], target[he_idx])
        row = dict(epoch=ep + 1, train_loss=tr_loss, train_top1=tr_top1,
                   held_loss=float(-m_he["logp"].mean()),
                   held_top1=float(m_he["top1"].mean()))
        curve.append(row)
        wandb.log({**row, "epoch": ep + 1})
        print(f"  epoch {ep + 1}: held loss {row['held_loss']:.4f} top-1 {row['held_top1']:.3f} "
              f"train loss {row['train_loss']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if row["held_loss"] < best - 1e-4:
            best, best_ep, bad = row["held_loss"], ep + 1, 0
            torch.save({"head": head.state_dict(), "cfg": {k: v for k, v in CFG.items()
                       if isinstance(v, (int, float, str, bool))}, "curve": curve,
                       "best_epoch": best_ep}, out_dir / "head_refit.pt")
        else:
            bad += 1
            if bad >= CFG["patience"]:
                print(f"early stop at epoch {ep + 1} (best {best_ep})")
                break
    assert torch.equal(head.word_emb.cpu(), emb_fin), "decoder weight moved during training"
    wandb.finish()


# ---- D4: evaluation on val ----
def _smoke_tag(smoke):
    return "_smoke" if smoke else ""


def load_refit_head(out_dir):
    """Trained head from head_refit.pt (fp32; decoder buffer re-asserted tied)."""
    ck = torch.load(out_dir / "head_refit.pt", map_location="cpu", weights_only=False)
    emb_fin = bert_weights("finetune")["text_encoder.embeddings.word_embeddings.weight"]
    head = RefitHead(head_weights(), emb_fin)
    head.load_state_dict(ck["head"])
    assert torch.equal(head.word_emb, emb_fin)
    return head.to(CFG["device"]).eval(), ck


@torch.no_grad()
def pretrain_null_h(encs, mask_id):
    """Pretrain BERT layer-11 states at the masked position for every variant of
    every enc (cache row order), null video (one zero vision token)."""
    w = {k: v.to(CFG["device"]) for k, v in bert_weights("pretrain").items()}
    vis = torch.zeros(1, 1, 768, device=CFG["device"])
    rows = [[*e[:p], mask_id, *e[p + 1:]] for e in encs for p in range(1, len(e))]
    pos = [p for e in encs for p in range(1, len(e))]
    h_out = np.zeros((len(rows), 768), np.float32)
    for s in range(0, len(rows), CFG["eval_bs"]):
        chunk = rows[s:s + CFG["eval_bs"]]
        L = max(map(len, chunk))
        x = torch.tensor([r + [0] * (L - len(r)) for r in chunk], device=CFG["device"])
        h = mlm_hidden(x, (x != 0).long(), vis, w)
        h_out[s:s + len(chunk)] = h[torch.arange(len(chunk), device=h.device),
                                     torch.tensor(pos[s:s + len(chunk)], device=h.device)].cpu().numpy()
    return torch.from_numpy(h_out)


def probe_token_ids(tok):
    """CFG probe words -> [(word, token id)], single word-piece only; drops logged."""
    keep = []
    for wd in CFG["probe_words"]:
        ids = tok(wd, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            keep.append((wd, ids[0]))
        else:
            print(f"dropped probe word {wd!r}: {len(ids)} word-pieces")
    return keep


def eval_val(cache_dir, out_dir):
    """Val metrics for refit (R/null/Rev) + graft and pretrain on null; per-blank
    parquet with probe-word log-probs (refit head, all conditions)."""
    clips = pd.read_parquet(cache_dir / "clips_val.parquet")
    tok = get_tokenizer()
    head, _ = load_refit_head(out_dir)
    probes = probe_token_ids(tok)
    probe_arr = np.array([tid for _, tid in probes], np.int64)
    cache = {c: np.load(cache_dir / f"cache_val_{c}.npz") for c in ("R", "null", "Rev")}
    target = torch.from_numpy(cache["null"]["target"]).long()
    is_obj = cache["null"]["is_obj"]
    clip_id = clips.clip_id.to_numpy()[cache["null"]["clip_idx"]]
    pos = cache["null"]["pos"]
    emb_fin = bert_weights("finetune")["text_encoder.embeddings.word_embeddings.weight"]
    summ = {c: head_eval(head, torch.from_numpy(cache[c]["h"]), target, probe_arr)
            for c in cache}
    graft_head = RefitHead(head_weights(), emb_fin).to(CFG["device"]).eval()
    summ["graft"] = head_eval(graft_head, torch.from_numpy(cache["null"]["h"]), target)
    emb_pre = bert_weights("pretrain")["text_encoder.embeddings.word_embeddings.weight"]
    pre_head = RefitHead(head_weights(), emb_pre).to(CFG["device"]).eval()
    ok_encs = clips.loc[clips.ok, "enc"].tolist()   # cache rows only cover decoded clips
    summ["pretrain"] = head_eval(pre_head, pretrain_null_h(ok_encs, tok.mask_token_id), target)
    rows = []
    masks = {"all": np.ones(len(target), bool), "object": is_obj, "other": ~is_obj}
    for model, cond in [("refit", "R"), ("refit", "null"), ("refit", "Rev"),
                        ("graft", "null"), ("pretrain", "null")]:
        s = summ[model if model != "refit" else cond]
        for part, m in masks.items():
            rows.append(dict(model=model, condition=cond, word_type=part, n=int(m.sum()),
                             top1=float(s["top1"][m].mean()), mean_logp=float(s["logp"][m].mean())))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(metrics.round(4).to_string(index=False))
    frames = []
    for cond in ("R", "null", "Rev"):
        s = summ[cond]
        d = dict(clip_id=clip_id, position=pos, target=target.numpy(), is_obj=is_obj,
                 condition=cond, refit_pred=s["pred"], refit_top1=s["top1"], refit_logp=s["logp"])
        if cond == "null":
            for model in ("graft", "pretrain"):
                d[f"{model}_pred"] = summ[model]["pred"]
                d[f"{model}_top1"] = summ[model]["top1"]
                d[f"{model}_logp"] = summ[model]["logp"]
        for j, (wd, _) in enumerate(probes):
            d[f"lp_{wd}"] = s["lp_probe"][:, j]
        frames.append(pd.DataFrame(d))
    pd.concat(frames, ignore_index=True).to_parquet(out_dir / "per_blank_val.parquet", index=False)
    print(f"saved metrics.csv + per_blank_val.parquet "
          f"({len(frames[0]) * 3:,} rows, {len(probes)} probe words kept)")


def main():
    # TF32 off, as in umt_vtm_gate: the port's fp32 paths stay exact.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    smoke = bool(CFG["max_train_clips"] or CFG["max_val_clips"])
    out_dir = CFG["out_dir"].with_name(CFG["out_dir"].name + _smoke_tag(smoke))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.environ.get("CACHE_DIR") or out_dir)
    if smoke:                      # never let a smoke run overwrite the real cache
        cache_dir = cache_dir.with_name(cache_dir.name + "_smoke")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"mode={mode}  device={CFG['device']}  out={out_dir}  cache={cache_dir}  "
          f"max_train={CFG['max_train_clips'] or 'all'}  max_val={CFG['max_val_clips'] or 'all'}")
    if mode == "port":
        tok = get_tokenizer()
        tokenizer_check(tok)
        port_check()
        graft_repro()
        return
    t0 = time.time()
    if mode in ("extract", "all"):
        extract("train", cache_dir)
        extract("val", cache_dir)
    if mode in ("train", "all"):
        train_head(cache_dir, out_dir, smoke)
    if mode in ("eval", "all"):
        eval_val(cache_dir, out_dir)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
