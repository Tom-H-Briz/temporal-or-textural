"""
Phase A of the position-embedding reversal test (CC brief, 16 Sep 2026).

Hypothesis: position-locked SAE features are detectors for a specific embedding
VECTOR, not for "this is temporal slot N". Intervention: reverse the
embedding-to-slot assignment before the forward pass — slot i (0-indexed)
receives the embedding normally assigned to slot 7-i (brief's 1-indexed
9-i: 1<->8, 2<->7, 3<->6, 4<->5) — while content stays in true order.

This synthetic phase is the break point before any Isambard spend: the same 18
content-free videos as the characterisation probe (white / black / noise x 16
seeds), whose every tubelet position carries identical content — so the
embedding vector is the ONLY thing that moved. Pre-registered readout:

  argmax moves to 7 - original_lock  -> features track the embedding vector
                                        (hypothesis confirmed at intervention
                                        level; near-certain preview of the
                                        real-clip outcome)
  argmax stays at original slot      -> slot-index stickers — the pre-registered
                                        falsification (other position-linked
                                        signal in the architecture)
  diffuse / inactive                 -> signal collapsed under the OOD
                                        intervention — ambiguous; the parked
                                        pairwise single-swap variant becomes
                                        the discriminator

Scope: VideoMAE, ssv2 (16 members) + kinetics400 (26 members), layers 3/5/7 —
all 42 strict scaffold members, k64 x8 job7ep SAEs. z-metric only (primary, per
decision); DFA has no role in synthetic mode.

Output: outputs/analysis/pos_embed_reversal/pos_embed_reversal_synthetic.csv

Usage: uv run python notebooks/pos_embed_reversal_synthetic.py  (CPU, ~minutes)
"""

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "stage3_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import (
    CHECKPOINT_REGISTRY, MODEL_REGISTRY, N_SPATIAL, gather_by_position,
    get_processor, resolve_sae_checkpoint,
)
from dfa_engine import DFAEngine
from pos_lock_synthetic_probe import make_frames, scaffold_members, synthetic_videos

CFG = {
    "model_name": "videomae",
    "sae_k":      64,
    "job_label":  "7ep",
    "configs": [("ssv2", 3), ("ssv2", 5), ("ssv2", 7),
                ("kinetics400", 3), ("kinetics400", 5), ("kinetics400", 7)],
    "output_dir": ROOT / "outputs" / "analysis" / "pos_embed_reversal",
}


def reverse_position_embeddings(engine: DFAEngine) -> torch.Tensor:
    """Slot i <- embed(7-i), 0-indexed (brief's 1-indexed 9-i). Content order
    untouched — only the embedding-to-slot assignment moves. Returns the
    original data for restoration."""
    pos = engine._model.videomae.embeddings.position_embeddings  # (1, 1568, 768)
    original = pos.data.clone()
    blocks = pos.data.view(8, N_SPATIAL, -1)
    pos.data = blocks.flip(0).reshape(1, 8 * N_SPATIAL, -1).contiguous()
    # guard: the intervention did what the brief specifies — slot 0 now carries
    # old slot 7's vector, and the block multiset is unchanged (pure permutation)
    flipped = pos.data.view(8, N_SPATIAL, -1)
    assert torch.equal(flipped[0], blocks[7]) and torch.equal(flipped[7], blocks[0])
    return original


def main() -> None:
    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = MODEL_REGISTRY[CFG["model_name"]]
    rows = []

    for dataset_name, layer in CFG["configs"]:
        members = scaffold_members(dataset_name, layer)
        resolved = resolve_sae_checkpoint(CFG["model_name"], layer, dataset_name,
                                          CFG["sae_k"], CFG["job_label"])
        processor = get_processor(model_cfg,
                                  CHECKPOINT_REGISTRY[(CFG["model_name"], dataset_name)])
        with DFAEngine(CFG["model_name"], resolved["sae_path"], resolved["dim_mean_path"],
                       layer=layer, device=device, sae_k=resolved["sae_k"],
                       dataset_name=dataset_name) as engine:
            original = reverse_position_embeddings(engine)


            for content, seed in synthetic_videos():
                pv = processor(make_frames(content, seed),
                               return_tensors="pt")["pixel_values"].to(device)
                z = engine.get_z_pixels(pv)  # label-free, (1568, dict_size)
                per_pos = gather_by_position(z, CFG["model_name"]).sum(dim=1)  # (8, D)
                for feat, locked in members:
                    col = per_pos[:, feat]
                    total = float(col.sum())
                    active = total > 1e-8  # the pipeline's own activity gate
                    predicted = 7 - locked  # pre-registered: embed(9-p) lands at slot p
                    argmax = int(col.argmax()) if active else -1
                    rows.append({
                        "dataset": dataset_name, "layer": layer,
                        "feature_idx": feat, "original_lock": locked,
                        "predicted_lock": predicted,
                        "content": content, "seed": seed, "active": active,
                        "argmax_position": argmax,
                        "share_at_predicted": round(float(col[predicted] / total), 4) if active else 0.0,
                        "relock_hit": bool(active and argmax == predicted),
                        "stays_at_original_slot": bool(active and argmax == locked),
                    })
            engine._model.videomae.embeddings.position_embeddings.data = original
        print(f"{dataset_name} L{layer}: {len(members)} members x {len(synthetic_videos())} videos, reversed")


    df = pd.DataFrame(rows)
    out_path = out_dir / "pos_embed_reversal_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    summary = (df.groupby(["dataset", "layer", "content"])
                 .agg(relock_rate=("relock_hit", "mean"),
                      active_rate=("active", "mean"),
                      mean_share_at_predicted=("share_at_predicted", "mean"))
                 .round(4).reset_index())
    print("\nReversal readout (pre-registered: argmax should move to 7 - original_lock):")
    print(summary.to_string(index=False))

    deviants = df[~df["relock_hit"]]
    if len(deviants):
        print("\nDEVIATIONS from the pre-registered prediction (feature, config, outcome):")
        for (d, l, f), g in deviants.groupby(["dataset", "layer", "feature_idx"]):
            print(f"  {d} L{l} feature {f}: relock {g.relock_hit.mean():.2f}, "
                  f"stays {g.stays_at_original_slot.mean():.2f}, active {g.active.mean():.2f}")
    else:
        print("\nNo deviations: every active member relocked at 7 - original_lock on every video.")


if __name__ == "__main__":
    main()
