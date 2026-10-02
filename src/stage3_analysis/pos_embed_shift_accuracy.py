"""
Temporal position-embedding one-slot offset vs raw accuracy.

Not a shuffle: three deterministic assignments, each perturbing the slot->time
map in one specific way —

    lag  [0,0,1,2,3,4,5,6]  embed lags content by one slot; embed(0) doubled,
                            embed(7) dropped
    lead [1,2,3,4,5,6,7,7]  embed leads content by one slot; embed(7) doubled,
                            embed(0) dropped
    reverse [7,6,5,4,3,2,1,0]  the progression mirrored: adjacency and spacing
                            intact, direction of the ramp flipped (the real-clip
                            version of the pre-registered synthetic reversal)

Content order untouched. This is the minimal-displacement companion to the
full shuffle: if a 1-slot misalignment already moves accuracy substantially,
the slot->time mapping is used with near-zero slack.

Same design as pos_embed_shuffle_accuracy.py: first 500 ranks of the same
clip_order_manifest.csv (identical clips, comparable numbers), cached
baseline_correct joined, 20-clip parity gate, resume-safe parquet. One decode,
one forward per arm per clip.

Usage:
    uv run python src/stage3_analysis/pos_embed_shift_accuracy.py --n-clips 5
    uv run python src/stage3_analysis/pos_embed_shift_accuracy.py
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from stage3_analysis.dfa_engine import _preprocess_clip
from stage3_analysis.pos_embed_shuffle_accuracy import (
    apply_temporal_perm, forward_stats, group_stats,
)
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY

CFG = {
    "model_flag": "videomae",
    "dataset": "ssv2",
    "n_clips": 500,
    "n_parity": 20,
    "checkpoint_every": 25,
    "device": "cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "T": 8, "SPATIAL": 196,
    "arms": {"lag": [0, 0, 1, 2, 3, 4, 5, 6],
             "lead": [1, 2, 3, 4, 5, 6, 7, 7],
             "reverse": [7, 6, 5, 4, 3, 2, 1, 0]},
    "manifest": ROOT / "outputs/analysis/pos_embed_shuffle/clip_order_manifest.csv",
    "anchors_csv": ROOT / "outputs/Laura_SL/per_class_accuracy_VM_all.csv",
    "video_dir": ROOT / "data/ssv2/20bn-something-something-v2",
    "out_dir": ROOT / "outputs/analysis/pos_embed_shift",
}


def load_pool(cfg: dict, n_clips: int) -> pd.DataFrame:
    """First n_clips ranks of the shuffle run's manifest — same clips, and the
    manifest already carries class ids, SL labels, class sizes, cached baseline."""
    m = pd.read_csv(cfg["manifest"], dtype={"clip_id": str})
    pool = m[m["rank"] < n_clips].copy()
    assert len(pool) == n_clips, f"manifest gave {len(pool)} clips, wanted {n_clips}"
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=CFG["n_clips"])
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

    pool = load_pool(cfg, args.n_clips)
    out_path = cfg["out_dir"] / "pos_embed_shift_results.parquet"
    done: set[tuple] = set()
    if out_path.exists():
        prev = pd.read_parquet(out_path, columns=["clip_id", "arm"])
        done = set(zip(prev.clip_id, prev.arm))
        print(f"  resuming: {len(done)} (clip, arm) pairs done")
    todo = pool[~pool.apply(
        lambda r: all((r.clip_id, arm) in done for arm in cfg["arms"]), axis=1)]
    print(f"Device: {cfg['device']}  running {len(todo)} clips x 2 arms "
          f"({len(pool) - len(todo)} fully done)")

    gate = todo.head(min(cfg["n_parity"], len(todo)))
    for _, r in gate.iterrows():
        pv = _preprocess_clip(cfg["video_dir"] / f"{r.clip_id}.webm",
                              model_cfg["num_frames"], processor, cfg["device"])
        _, pred = forward_stats(model, pv, int(r.class_id))   # baseline pass
        if bool(pred == int(r.class_id)) != bool(r.baseline_correct):
            raise SystemExit(f"parity gate FAILED at {r.clip_id}")
    print(f"  parity gate passed ({len(gate)}/{len(gate)})")

    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(todo.iterrows()):
        pv = _preprocess_clip(cfg["video_dir"] / f"{r.clip_id}.webm",
                              model_cfg["num_frames"], processor, cfg["device"])
        for arm, assign in cfg["arms"].items():
            if (r.clip_id, arm) in done:
                continue
            apply_temporal_perm(model, assign, cfg)
            logit, pred = forward_stats(model, pv, int(r.class_id))
            pos.data = original
            rows.append({"clip_id": r.clip_id, "class_id": int(r.class_id),
                         "sl_label": r.sl_label, "class_n": int(r.class_n),
                         "arm": arm, "assignment": "".join(map(str, assign)),
                         "baseline_correct": bool(r.baseline_correct),
                         "shuf_logit": logit, "shuf_pred": pred,
                         "shuf_correct": pred == int(r.class_id)})
        if (i + 1) % cfg["checkpoint_every"] == 0 or i == len(todo) - 1:
            merged = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
            merged = pd.concat([merged, pd.DataFrame(rows)])
            merged = merged.drop_duplicates(["clip_id", "arm"])
            merged.to_parquet(out_path, index=False)
            rows = []
        if (i + 1) % 25 == 0 or i == len(todo) - 1:
            print(f"  [{i+1}/{len(todo)}]  {(time.time()-t0)/(i+1):.1f}s/clip")

    assert torch.equal(pos.data, original), "embeddings not restored"
    summarise(pd.read_parquet(out_path), cfg)


def summarise(df: pd.DataFrame, cfg: dict) -> None:
    out = []
    for arm in cfg["arms"]:
        d = df[df.arm == arm]
        rows = [{"arm": arm, "scope": "overall", **group_stats(d)}]
        for label in ("temporal", "static"):
            sub = d[d.sl_label == label]
            if len(sub):
                rows.append({"arm": arm, "scope": label, **group_stats(sub)})
        pc = pd.DataFrame([group_stats(g) for _, g in d.groupby("class_id")])
        macro = pc.drop(columns=["n_clips"]).mean()
        rows.append({"arm": arm, "scope": "macro", "n_clips": d.class_id.nunique(),
                     **{k: macro[k] for k in macro.index}})
        out += rows
    summary = pd.DataFrame(out)
    summary.to_csv(cfg["out_dir"] / "pos_embed_shift_summary.csv", index=False)
    if cfg["anchors_csv"].exists():
        a = pd.read_csv(cfg["anchors_csv"])
        a = a[a.class_id.isin(df.class_id)]
        for cond, tag in [("R", "unperturbed R"), ("C", "full frame shuffle C")]:
            print(f"  anchor {tag}: {a[f'correct_{cond}'].sum() / a[f'total_{cond}'].sum():.3f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
