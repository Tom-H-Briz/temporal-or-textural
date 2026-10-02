"""
Temporal position-embedding shuffle vs raw accuracy — the slot-identity arm.

Existing perturbations (A/B/C/C1/C2) destroy content order at the input; none
touch the positional embeddings. Here content stays in true order and the
model's slot-identity signal is scrambled instead: per clip, a fresh
non-identity random permutation of the 8 temporal embedding blocks (slot i
<- embed(perm[i]); spatial 196 vectors per slot intact). Reference anchors:
R = 67.3%, C (full frame shuffle) = 7.1% on the same 35 SL classes.

Population: 500 (seed 0, manifest order) of the 919 SL-subset clips with
cached per-clip baseline_correct (spliced_accuracy_sweep_per_clip.parquet,
held-out split, ~26/class). Baseline is joined, not re-run, guarded by a
20-clip parity gate (in-run baseline must match cached 20/20 or abort).

Resume/extension: all 919 clips are ranked deterministically into
clip_order_manifest.csv; rerun with --n-clips 919 to add the remaining 419.
Done clip_ids are skipped, results accumulate in the one parquet.

Usage:
    uv run python src/stage3_analysis/pos_embed_shuffle_accuracy.py --n-clips 5
    uv run python src/stage3_analysis/pos_embed_shuffle_accuracy.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from stage3_analysis.dfa_engine import _preprocess_clip
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY

CFG = {
    "model_flag": "videomae",
    "dataset": "ssv2",
    "n_clips": 500,
    "seed": 0,
    "n_parity": 20,          # first N manifest clips get an in-run baseline pass
    "checkpoint_every": 25,  # parquet rewrites — crash/resume loses <= this many
    "device": "cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "T": 8,                  # temporal tubelet slots; asserted against the model
    "SPATIAL": 196,          # spatial tokens per slot; asserted (8*196 = 1568)
    "baseline_parquet": ROOT / "outputs/analysis/spliced_accuracy_sweep/"
                                  "spliced_accuracy_sweep_per_clip.parquet",
    "baseline_config": "VM_ssv2_L5_k64",
    "sl_csv": ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv",
    "anchors_csv": ROOT / "outputs/Laura_SL/per_class_accuracy_VM_all.csv",
    "video_dir": ROOT / "data/ssv2/20bn-something-something-v2",
    "out_dir": ROOT / "outputs/analysis/pos_embed_shuffle",
}


def load_baseline(cfg: dict) -> pd.DataFrame:
    """919 SL-subset clips with cached per-clip baseline correctness. The
    baseline pass is SAE-free, so every ssv2 config must agree on it — checked,
    then canonical rows are taken from the reference config."""
    all_rows = pd.read_parquet(cfg["baseline_parquet"])
    ssv2 = all_rows[all_rows.dataset == "ssv2"]
    per_clip = ssv2.groupby("clip_id")["baseline_correct"].nunique()
    assert per_clip.max() == 1, "baseline_correct differs across configs — join unsafe"
    base = ssv2[ssv2.config == cfg["baseline_config"]][
        ["clip_id", "class_id", "baseline_correct"]].copy()
    base["clip_id"] = base.clip_id.astype(str).str.removesuffix(".webm")
    return base


def build_manifest(base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join SL labels and per-class totals, then rank ALL clips deterministically
    (seed 0). The run is a rank cutoff, so extending later = --n-clips 919."""
    sl = pd.read_csv(cfg["sl_csv"])[["class_id", "category", "total"]]
    sl = sl.rename(columns={"category": "sl_label", "total": "class_n"})
    pool = base.merge(sl, on="class_id", how="inner")
    print(f"  pool: {len(pool)} SL clips, {pool.class_id.nunique()} classes, "
          f"baseline acc {pool.baseline_correct.mean():.3f}")
    rng = np.random.default_rng(cfg["seed"])
    pool = pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)
    pool["rank"] = pool.index
    pool.to_csv(cfg["out_dir"] / "clip_order_manifest.csv", index=False)
    return pool


def draw_perm(cfg: dict, clip_id: str) -> np.ndarray:
    """Non-identity temporal permutation, seeded per clip so reruns and resumes
    assign the same perm to the same clip regardless of execution order."""
    rng = np.random.default_rng([cfg["seed"], int(clip_id)])
    perm = rng.permutation(cfg["T"])
    while list(perm) == list(range(cfg["T"])):
        perm = rng.permutation(cfg["T"])
    return perm


def apply_temporal_perm(model, perm: np.ndarray, cfg: dict) -> None:
    """Slot i <- embed(perm[i]): permute temporal blocks, spatial intact."""
    pos = model.videomae.embeddings.position_embeddings
    assert tuple(pos.shape[1:]) == (cfg["T"] * cfg["SPATIAL"], pos.shape[2])
    blocks = pos.data.view(cfg["T"], cfg["SPATIAL"], -1)
    pos.data = blocks[torch.as_tensor(perm)].reshape(1, -1, pos.shape[2]).contiguous()


def forward_stats(model, pixel_values: torch.Tensor, class_id: int) -> tuple[float, int]:
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits.squeeze(0)
    return float(logits[class_id]), int(logits.argmax())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=CFG["n_clips"],
                        help="take the first N clips of the manifest order "
                             "(919 = add the remaining 419)")
    args = parser.parse_args()
    cfg = CFG
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    model_cfg = MODEL_REGISTRY[cfg["model_flag"]]
    checkpoint = CHECKPOINT_REGISTRY[(cfg["model_flag"], cfg["dataset"])]
    processor = model_cfg["processor_class"].from_pretrained(checkpoint)
    model = model_cfg["model_class"].from_pretrained(checkpoint).to(cfg["device"]).eval()
    model.requires_grad_(False)
    pos = model.videomae.embeddings.position_embeddings
    original = pos.data.clone()
    print(f"Device: {cfg['device']}  pos table {tuple(pos.shape)}")

    pool = build_manifest(load_baseline(cfg), cfg)
    out_path = cfg["out_dir"] / "pos_embed_shuffle_results.parquet"
    done: set[str] = set()
    if out_path.exists():
        done = set(pd.read_parquet(out_path, columns=["clip_id"]).clip_id)
        print(f"  resuming: {len(done)} clips already done")
    todo = pool[pool["rank"] < args.n_clips]
    todo = todo[~todo.clip_id.isin(done)]
    print(f"  running {len(todo)} of first {args.n_clips} manifest clips "
          f"({args.n_clips - len(todo)} skipped)")

    gate = todo.head(min(cfg["n_parity"], len(todo)))
    for _, r in gate.iterrows():
        pv = _preprocess_clip(cfg["video_dir"] / f"{r.clip_id}.webm",
                              model_cfg["num_frames"], processor, cfg["device"])
        _, pred = forward_stats(model, pv, int(r.class_id))   # baseline pass
        if bool(pred == int(r.class_id)) != bool(r.baseline_correct):
            raise SystemExit(f"parity gate FAILED at {r.clip_id}: in-run baseline "
                             f"{pred == int(r.class_id)} vs cached {r.baseline_correct}"
                             " — decode differs, cached join unsafe")
    print(f"  parity gate passed ({len(gate)}/{len(gate)} match cached baseline)")

    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(todo.iterrows()):
        clip_path = cfg["video_dir"] / f"{r.clip_id}.webm"
        pv = _preprocess_clip(clip_path, model_cfg["num_frames"], processor,
                              cfg["device"])
        perm = draw_perm(cfg, r.clip_id)
        apply_temporal_perm(model, perm, cfg)
        shuf_logit, shuf_pred = forward_stats(model, pv, int(r.class_id))
        pos.data = original                                   # restore per clip
        rows.append({
            "clip_id": r.clip_id, "class_id": int(r.class_id),
            "sl_label": r.sl_label, "class_n": int(r.class_n),
            "perm": "".join(map(str, perm)), "baseline_correct": bool(r.baseline_correct),
            "shuf_logit": shuf_logit, "shuf_pred": shuf_pred,
            "shuf_correct": shuf_pred == int(r.class_id),
        })
        if (i + 1) % cfg["checkpoint_every"] == 0 or i == len(todo) - 1:
            merged = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
            merged = pd.concat([merged, pd.DataFrame(rows)]).drop_duplicates("clip_id")
            merged.to_parquet(out_path, index=False)
            rows = []
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            print(f"  [{i+1}/{len(todo)}]  {(time.time()-t0)/(i+1):.1f}s/clip")

    assert torch.equal(pos.data, original), "embeddings not restored"
    summarise(pd.read_parquet(out_path), cfg)


def group_stats(g: pd.DataFrame) -> dict:
    """Paired baseline-vs-shuffled stats for one group of clips."""
    b = g.baseline_correct.astype(float)
    s = g.shuf_correct.astype(float)
    d = b - s
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {"n_clips": len(g), "baseline_acc": b.mean(), "shuffled_acc": s.mean(),
            "drop": d.mean(), "ci95_lo": d.mean() - 1.96 * se,
            "ci95_hi": d.mean() + 1.96 * se,
            "flip_correct_to_wrong": float((b * (1 - s)).mean()),
            "flip_wrong_to_correct": float(((1 - b) * s).mean())}


def summarise(df: pd.DataFrame, cfg: dict) -> None:
    rows = [{"scope": "overall", **group_stats(df)}]
    for label in ("temporal", "static"):
        sub = df[df.sl_label == label]
        if len(sub):
            rows.append({"scope": label, **group_stats(sub)})
    pc = pd.DataFrame([group_stats(g) for _, g in df.groupby("class_id")])
    macro = pc.drop(columns=["n_clips"]).mean()
    rows.append({"scope": "macro", "n_clips": df.class_id.nunique(),
                 **{k: macro[k] for k in macro.index}})
    w = df.class_n / df.groupby("class_id").clip_id.transform("count")
    b, s = df.baseline_correct.astype(float), df.shuf_correct.astype(float)
    rows.append({"scope": "class-reweighted", "n_clips": len(df),
                 "baseline_acc": (w * b).sum() / w.sum(),
                 "shuffled_acc": (w * s).sum() / w.sum(),
                 "drop": (w * (b - s)).sum() / w.sum()})
    summary = pd.DataFrame(rows)
    summary.to_csv(cfg["out_dir"] / "pos_embed_shuffle_summary.csv", index=False)

    if cfg["anchors_csv"].exists():
        a = pd.read_csv(cfg["anchors_csv"])
        a = a[a.class_id.isin(df.class_id)]          # same SL classes as the run
        for cond, tag in [("R", "unperturbed R"), ("C", "full frame shuffle C")]:
            acc = a[f"correct_{cond}"].sum() / a[f"total_{cond}"].sum()
            print(f"  anchor {tag}: {acc:.3f}")
    print(f"  {len(df)} clips done of {len(pd.read_csv(cfg['out_dir'] / 'clip_order_manifest.csv'))} in manifest")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
