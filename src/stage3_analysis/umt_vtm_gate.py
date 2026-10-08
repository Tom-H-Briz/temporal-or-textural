"""UMT VTM gate: does VTM (BERT fusion layers 9-11 + itm_head) add discriminative value over VTC?
Test 1: official rerank (VTM-only score over the VTC top-k). Test 2: two-way minimal pairs.

Stage 1 (GPU, SLURM): per clip, VTC logits (174) + VTM scores for all 174 templates -> per_clip.parquet.
Stage 2 (CPU): aggregate per_clip.parquet -> rerank_summary / rerank_per_class / pairs_2way CSVs.
Usage: python src/stage3_analysis/umt_vtm_gate.py [extract|aggregate|all]   (MAX_CLIPS=200 for smoke)
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "notebooks"), str(ROOT / "src" / "stage3_analysis")]
from umt_text_geometry import CFG as GEOM_CFG   # pair class ids identical to the geometry analysis

CFG = {
    "labels_path":     os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),   # = DATASET_REGISTRY default
    "max_clips":       int(os.environ.get("MAX_CLIPS", 0)),   # 0 = full val (24,777); >0 smoke -> *_smoke dir
    "text_cache":      ROOT / "notebooks/umt_ssv2_template_bert_l8.npz",  # from umt_vtm_text_cache.py
    "out_dir":         ROOT / "outputs/analysis/umt_vtm_gate",
    "batch_size":      8,
    "num_workers":     4,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "tpl_chunk":       58,                  # templates per fusion pass (174 = 3 chunks; bounds attn memory)
    "ks":              [2, 5, 10, 128],     # 128 = official k_test (exp/finetuning/ret_ssv2_tpl/b16_25m.py)
    "official_k":      128,
    "vtc_ref_top1":    15440 / 24777,       # per_class_accuracy_UMT_ssv2_R.csv (62.32%)
    "vtc_tol":         0.005,               # stop if |VTC top-1 - ref| > 0.5pp
    "n_boot":          1000,
    "seed":            0,
}
# Brief's pair sets, keyed by the contrast names in umt_text_geometry.CFG["pairs"].
# Geometry pairs not listed here (the push/pull cross-diagonals) are not part of Test 2.
PAIR_SETS = {
    "usable": ["cover vs uncover", "open vs close", "fold vs unfold", "camera up vs down",
               "move up vs down", "towards vs away camera", "closer vs away from",
               "closer vs apart (each other)"],
    "flip":   ["push LtR vs RtL", "pull LtR vs RtL", "camera left vs right"],
    "verb":   ["push vs pull, LtR", "push vs pull, RtL"],
}
EXTRA_PAIRS = []   # secondary "Pretending to X" vs "X": (class_a, class_b, name, "pretending") — ids pending Tom


def test2_pairs():
    """[(a, b, name, set)] for Test 2."""
    set_of = {name: s for s, names in PAIR_SETS.items() for name in names}
    pairs = [(a, b, n, set_of[n]) for a, b, n in GEOM_CFG["pairs"] if n in set_of]
    assert len(pairs) == sum(len(v) for v in PAIR_SETS.values()), "pair name mismatch with geometry CFG"
    return pairs + EXTRA_PAIRS


# ---- VTM: BERT fusion layers 9-11 + itm_head in plain torch (mirrors UMT xbert BertLayer, post-LN) ----
def _lin(x, w, name):
    return F.linear(x, w[name + ".weight"], w[name + ".bias"])


def _heads(x):   # (N, L, 768) -> (N, 12, L, 64)
    return x.view(*x.shape[:2], 12, 64).transpose(1, 2)


def _attend(x, k, v, w, pre, add_mask):
    """xbert BertAttention: q from x; k/v given (cross-attn k/v broadcast over N); add & LayerNorm."""
    s = _heads(_lin(x, w, pre + ".self.query")) @ k.transpose(-1, -2) / 8.0   # sqrt(64)
    if add_mask is not None:
        s = s + add_mask
    ctx = (s.softmax(-1) @ v).transpose(1, 2).reshape(x.shape)
    return F.layer_norm(_lin(ctx, w, pre + ".output.dense") + x, (768,),
                        w[pre + ".output.LayerNorm.weight"], w[pre + ".output.LayerNorm.bias"], 1e-12)


def _kv(x, w, pre):
    return _heads(_lin(x, w, pre + ".self.key")), _heads(_lin(x, w, pre + ".self.value"))


def _fusion_layer(h, add_mask, cross_kv, w, i):
    """Self-attn (text, padding-masked) -> cross-attn (vision, unmasked: UMT's mask is all ones) -> FFN."""
    p = f"text_encoder.encoder.layer.{i}"
    h = _attend(h, *_kv(h, w, p + ".attention"), w, p + ".attention", add_mask)
    h = _attend(h, *cross_kv, w, p + ".crossattention", None)
    ff = _lin(F.gelu(_lin(h, w, p + ".intermediate.dense")), w, p + ".output.dense")
    return F.layer_norm(ff + h, (768,), w[p + ".output.LayerNorm.weight"], w[p + ".output.LayerNorm.bias"], 1e-12)


@torch.no_grad()
def vtm_scores(vis, text, w):
    """vis (L, 768) vision tokens of one clip -> (174,) raw itm_head[:, 1] logits, one per template.
    Cross-attn k/v depend only on the video, so they're projected once and broadcast over templates
    (same maths as UMT's eval, which repeats the video k times)."""
    cross = {i: _kv(vis.unsqueeze(0), w, f"text_encoder.encoder.layer.{i}.crossattention") for i in (9, 10, 11)}
    out = []
    for s in range(0, len(text["hidden"]), CFG["tpl_chunk"]):
        h, m = text["hidden"][s:s + CFG["tpl_chunk"]], text["add_mask"][s:s + CFG["tpl_chunk"]]
        for i in (9, 10, 11):
            h = _fusion_layer(h, m, cross[i], w, i)
        out.append(_lin(h[:, 0], w, "itm_head")[:, 1])
    return torch.cat(out)


def load_vtm(sd, device):
    """Fusion weights (layers 9-11, itm_head) from the checkpoint + cached layer-0-8 template states.
    Asserts the plain-torch fusion reproduces xbert's reference ITM scores before anything runs."""
    keep = tuple(f"text_encoder.encoder.layer.{i}." for i in (9, 10, 11)) + ("itm_head.",)
    w = {k: v.float().to(device) for k, v in sd.items() if k.startswith(keep)}
    c = np.load(CFG["text_cache"])
    mask = torch.from_numpy(c["mask"]).float()
    text = {"hidden": torch.from_numpy(c["hidden"]).to(device),
            "add_mask": ((1.0 - mask) * -10000.0)[:, None, None, :].to(device)}   # xbert convention
    ref = torch.stack([vtm_scores(v.to(device), text, w) for v in torch.from_numpy(c["ref_vision"])])
    err = (ref.cpu() - torch.from_numpy(c["ref_itm"])).abs().max().item()
    print(f"VTM self-check vs xbert reference: max |diff| = {err:.2e}")
    assert err < 1e-3, "plain-torch fusion does not reproduce xbert"
    return text, w


# ---- Stage 1: extraction (same model, clip set, sampler and preprocessing as perturb_accuracy_umt.py R) ----
def load_model_and_data():
    from huggingface_hub import hf_hub_download
    from ToT_utils import MODEL_REGISTRY, get_frame_sampler
    from perturb_accuracy_tf_kinetics import PerturbedKineticsDataset
    from perturb_accuracy_vm_ssv2 import load_ssv2_clips
    from umt_wrapper import CFG as UMT_CFG, UMTClassifier, UMTProcessor
    model = UMTClassifier.from_pretrained().to(CFG["device"]).eval()
    sd = torch.load(hf_hub_download(UMT_CFG["hf_repo"], UMT_CFG["hf_filename"]), map_location="cpu", weights_only=False)
    paths, clip_ids, labels, _ = load_ssv2_clips(CFG)
    if CFG["max_clips"]:
        paths, clip_ids, labels = (x[: CFG["max_clips"]] for x in (paths, clip_ids, labels))
    ds = PerturbedKineticsDataset(paths, clip_ids, labels, UMTProcessor(), MODEL_REGISTRY["umt"]["num_frames"],
                                  get_frame_sampler("ssv2", MODEL_REGISTRY["umt"]), "R")
    loader = torch.utils.data.DataLoader(ds, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])
    return model, sd, loader, clip_ids


def _sync():
    if CFG["device"] == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def vision_and_vtc(model, pixel_values):
    """-> vision tokens after encoder.norm (B, 12*196, 768) = UMT's cross-attn input, and VTC logits
    (B, 174) computed exactly as UMTClassifier.forward (per-frame cosine, mean over frames, / temp)."""
    x_vis, pooled, _ = model.vision(pixel_values.permute(0, 2, 1, 3, 4), keep_temporal=True)
    frames = F.normalize(model.vision_proj(pooled), dim=-1)
    vtc = torch.einsum("btd,cd->bc", frames, model.text_emb) / frames.shape[1] / model.temp
    return x_vis, vtc


def extract(out_dir):
    """Raw per-clip data before any aggregation: VTC logits and VTM scores for all 174 templates
    (a superset of the brief's top-k / true / partner fields; those are derived in aggregate())."""
    model, sd, loader, clip_ids = load_model_and_data()
    text, w = load_vtm(sd, CFG["device"])
    rows, t_vis, t_vtm, i = [], 0.0, 0.0, 0
    for pixel_values, labels in loader:
        _sync(); t0 = time.time()
        x_vis, vtc = vision_and_vtc(model, pixel_values.to(CFG["device"]))
        _sync(); t1 = time.time()
        vtm = torch.stack([vtm_scores(v, text, w) for v in x_vis])
        _sync(); t_vis, t_vtm = t_vis + t1 - t0, t_vtm + time.time() - t1
        for v_c, v_m, y in zip(vtc.cpu().numpy(), vtm.cpu().numpy(), labels.tolist()):
            rows.append(dict(clip_id=clip_ids[i], true_class=y, vtc_logits=v_c, vtm_scores=v_m)); i += 1
        if len(rows) % 800 < CFG["batch_size"]:
            print(f"  {len(rows):,} clips  vision {t_vis / len(rows) * 1e3:.1f} ms/clip  vtm {t_vtm / len(rows) * 1e3:.1f} ms/clip", flush=True)
    pd.DataFrame(rows).to_parquet(out_dir / "per_clip.parquet", index=False)
    n_full = 24777
    print(f"done {len(rows):,} clips: vision {t_vis / len(rows) * 1e3:.1f} ms/clip, vtm {t_vtm / len(rows) * 1e3:.1f} ms/clip; "
          f"projected full run (compute only, excl. decoding) {(t_vis + t_vtm) / len(rows) * n_full / 60:.1f} min")


# ---- Stage 2: aggregation over per_clip.parquet ----
def load_per_clip(out_dir):
    df = pd.read_parquet(out_dir / "per_clip.parquet")
    return np.stack(df.vtc_logits), np.stack(df.vtm_scores), df.true_class.to_numpy()


def rerank_order(V, M, k):
    """Official rule (retrieval_utils.evaluation): VTC picks the top-k, VTM alone orders them.
    Returns (N, k) class ids, best first."""
    topk = np.argsort(-V, axis=1)[:, :k]
    vtm_k = np.take_along_axis(M, topk, axis=1)
    return np.take_along_axis(topk, np.argsort(-vtm_k, axis=1), axis=1)


def mcnemar_p(n_b, n_c):
    """Exact two-sided McNemar on the discordant counts."""
    from scipy.stats import binomtest
    return 1.0 if n_b + n_c == 0 else binomtest(n_b, n_b + n_c, 0.5).pvalue


def rerank_summary(V, M, y):
    """One row per k. rerank_top5 is NaN for k < 5 (the official rule leaves ranks > k at -100, unordered)."""
    vtc_rank = np.argsort(-V, axis=1)
    vtc_ok = vtc_rank[:, 0] == y
    rows = []
    for k in CFG["ks"]:
        order = rerank_order(V, M, k)
        rr_ok = order[:, 0] == y
        n_fixed, n_broken = int((rr_ok & ~vtc_ok).sum()), int((~rr_ok & vtc_ok).sum())
        rows.append(dict(
            k=k, combination_rule="vtm_only_over_vtc_topk (official)", primary=k == CFG["official_k"],
            vtc_top1=vtc_ok.mean(), vtc_recall_at_k=(vtc_rank[:, :k] == y[:, None]).any(1).mean(),
            rerank_top1=rr_ok.mean(),
            rerank_top5=(order[:, :5] == y[:, None]).any(1).mean() if k >= 5 else np.nan,
            n_changed=int((order[:, 0] != vtc_rank[:, 0]).sum()), n_fixed=n_fixed, n_broken=n_broken,
            mcnemar_p=mcnemar_p(n_fixed, n_broken)))
    return pd.DataFrame(rows)


def rerank_per_class(V, M, y):
    """Official-k rerank top-1 per class, same columns as per_class_accuracy_UMT_ssv2_R.csv + VTC alongside."""
    import json
    id2name = {int(v): k for k, v in json.load(open(CFG["labels_path"])).items()}
    rr_ok = rerank_order(V, M, CFG["official_k"])[:, 0] == y
    vtc_ok = V.argmax(1) == y
    df = pd.DataFrame(dict(class_id=y, rr=rr_ok, vtc=vtc_ok)).groupby("class_id").agg(
        correct=("rr", "sum"), total=("rr", "size"), vtc_correct=("vtc", "sum")).reset_index()
    df.insert(1, "template", df.class_id.map(id2name))
    df["accuracy"] = df.correct / df.total
    df["vtc_accuracy"] = df.vtc_correct / df.total
    df["delta"] = df.accuracy - df.vtc_accuracy
    return df.sort_values("accuracy", ascending=False)


def pairs_2way(V, M, y):
    """Per pair, over clips of class A or B: correct iff score(true) > score(partner), for VTC and VTM.
    Bootstrap CI (over clips) on VTM - VTC accuracy; paired exact McNemar."""
    rng, rows = np.random.default_rng(CFG["seed"]), []
    for a, b, name, pset in test2_pairs():
        idx = np.where((y == a) | (y == b))[0]
        true, partner = y[idx], np.where(y[idx] == a, b, a)
        m_vtc = V[idx, true] - V[idx, partner]
        m_vtm = M[idx, true] - M[idx, partner]
        ok_c, ok_m = m_vtc > 0, m_vtm > 0
        boot = rng.integers(0, len(idx), (CFG["n_boot"], len(idx)))
        diffs = ok_m[boot].mean(1) - ok_c[boot].mean(1)
        rows.append(dict(
            pair=name, set=pset, n_A=int((y == a).sum()), n_B=int((y == b).sum()),
            vtc_2way_acc=ok_c.mean(), vtm_2way_acc=ok_m.mean(), diff=ok_m.mean() - ok_c.mean(),
            diff_ci_low=np.percentile(diffs, 2.5), diff_ci_high=np.percentile(diffs, 97.5),
            mcnemar_p=mcnemar_p(int((ok_m & ~ok_c).sum()), int((~ok_m & ok_c).sum())),
            vtc_margin_median=np.median(m_vtc), vtm_margin_median=np.median(m_vtm),
            vtc_margin_mean=m_vtc.mean(), vtm_margin_mean=m_vtm.mean()))
    return pd.DataFrame(rows)


def aggregate(out_dir, smoke):
    V, M, y = load_per_clip(out_dir)
    top1 = (V.argmax(1) == y).mean()
    print(f"VTC top-1 {top1:.4%} on {len(y):,} clips (ref {CFG['vtc_ref_top1']:.4%})")
    if not smoke and abs(top1 - CFG["vtc_ref_top1"]) > CFG["vtc_tol"]:
        sys.exit("VTC top-1 deviates > 0.5pp from per_class_accuracy_UMT_ssv2_R.csv: stopping (brief).")
    summ, pairs = rerank_summary(V, M, y), pairs_2way(V, M, y)
    summ.to_csv(out_dir / "rerank_summary.csv", index=False)
    rerank_per_class(V, M, y).to_csv(out_dir / "rerank_per_class.csv", index=False)
    pairs.to_csv(out_dir / "pairs_2way.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.drop(columns="combination_rule").round(4).to_string(index=False))
    print(pairs[["pair", "set", "n_A", "n_B", "vtc_2way_acc", "vtm_2way_acc", "diff", "diff_ci_low",
                 "diff_ci_high", "mcnemar_p"]].round(4).to_string(index=False))
    u = pairs[pairs.set == "usable"]
    print(f"usable-set median 2-way: VTC {u.vtc_2way_acc.median():.4f}  VTM {u.vtm_2way_acc.median():.4f}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    smoke = CFG["max_clips"] > 0
    out_dir = CFG["out_dir"].with_name(CFG["out_dir"].name + "_smoke") if smoke else CFG["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"mode={mode}  device={CFG['device']}  max_clips={CFG['max_clips'] or 'all'}  out={out_dir}")
    t0 = time.time()
    if mode in ("extract", "all"):
        extract(out_dir)
        print(f"extract wall time {(time.time() - t0) / 60:.1f} min (incl. decoding)", flush=True)
    if mode in ("aggregate", "all"):
        aggregate(out_dir, smoke)


if __name__ == "__main__":
    main()
