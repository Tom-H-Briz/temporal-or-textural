"""
Guard tests for ToT_utils.gather_by_position — brief §7.

1. Synthetic gather test: fabricate tokens in each model's true on-disk token
   order (VM: temporal-major, TF: patch-major/frame-minor) with values that
   encode their true (position, patch) origin, and assert gather_by_position
   recovers the correct (position, patch) grouping for both.
2. Count assert: verify each of the 8 output positions is fed by exactly 196
   unique source-token indices, no overlap, full 1568-token coverage.

Run once, unit-test style — not part of the scan itself.

Usage:
    uv run python notebooks/test_gather_by_position.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from ToT_utils import N_SPATIAL, gather_by_position

NUM_POSITIONS = 8
NUM_TOKENS = NUM_POSITIONS * N_SPATIAL


def value(position: int, patch: int) -> int:
    return position * 10_000 + patch


def test_synthetic_recovery(model_flag: str, true_token_idx) -> None:
    """true_token_idx(position, patch) -> flat index in the model's real token order."""
    raw = torch.zeros(NUM_TOKENS, 1)
    for p in range(NUM_POSITIONS):
        for s in range(N_SPATIAL):
            raw[true_token_idx(p, s), 0] = value(p, s)

    grouped = gather_by_position(raw, model_flag)  # (num_positions, N_SPATIAL, 1)
    assert grouped.shape == (NUM_POSITIONS, N_SPATIAL, 1)
    for p in range(NUM_POSITIONS):
        for s in range(N_SPATIAL):
            got = int(grouped[p, s, 0].item())
            assert got == value(p, s), (
                f"{model_flag}: position {p} patch {s} — expected {value(p, s)}, got {got}"
            )
    print(f"  {model_flag}: synthetic recovery OK ({NUM_POSITIONS}x{N_SPATIAL} grouping correct)")


def test_count_coverage(model_flag: str) -> None:
    """Every output position is fed by exactly N_SPATIAL unique source indices,
    no overlap, full NUM_TOKENS coverage."""
    idx = torch.arange(NUM_TOKENS).reshape(NUM_TOKENS, 1)
    grouped = gather_by_position(idx, model_flag).reshape(NUM_POSITIONS, N_SPATIAL)
    seen = set()
    for p in range(NUM_POSITIONS):
        position_indices = grouped[p].tolist()
        assert len(position_indices) == N_SPATIAL
        assert len(set(position_indices)) == N_SPATIAL, f"{model_flag}: duplicate indices within position {p}"
        seen.update(position_indices)
    assert seen == set(range(NUM_TOKENS)), f"{model_flag}: coverage mismatch — {NUM_TOKENS - len(seen)} tokens missing"
    print(f"  {model_flag}: count/coverage OK ({NUM_POSITIONS} x {N_SPATIAL} = {NUM_TOKENS}, no overlap)")


def main() -> None:
    print("VM (temporal-major: token_idx = position*N_SPATIAL + patch)")
    test_synthetic_recovery("videomae", lambda p, s: p * N_SPATIAL + s)
    test_count_coverage("videomae")

    print("TF (patch-major/frame-minor: token_idx = patch*NUM_POSITIONS + position)")
    test_synthetic_recovery("timesformer", lambda p, s: s * NUM_POSITIONS + p)
    test_count_coverage("timesformer")

    print("\nAll guards passed.")


if __name__ == "__main__":
    main()
