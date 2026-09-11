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

# model_flag -> (num_positions, token layout note)
CASES = {
    "videomae":   (8,  "temporal-major: token_idx = position*N_SPATIAL + patch"),
    "timesformer": (8, "patch-major/frame-minor: token_idx = patch*num_positions + position"),
    "vivit":      (16, "temporal-major: token_idx = position*N_SPATIAL + patch"),
}


def value(position: int, patch: int) -> int:
    return position * 10_000 + patch


def test_synthetic_recovery(model_flag: str, true_token_idx, num_positions: int) -> None:
    """true_token_idx(position, patch) -> flat index in the model's real token order."""
    num_tokens = num_positions * N_SPATIAL
    raw = torch.zeros(num_tokens, 1)
    for p in range(num_positions):
        for s in range(N_SPATIAL):
            raw[true_token_idx(p, s), 0] = value(p, s)

    grouped = gather_by_position(raw, model_flag)  # (num_positions, N_SPATIAL, 1)
    assert grouped.shape == (num_positions, N_SPATIAL, 1)
    for p in range(num_positions):
        for s in range(N_SPATIAL):
            got = int(grouped[p, s, 0].item())
            assert got == value(p, s), (
                f"{model_flag}: position {p} patch {s} — expected {value(p, s)}, got {got}"
            )
    print(f"  {model_flag}: synthetic recovery OK ({num_positions}x{N_SPATIAL} grouping correct)")


def test_count_coverage(model_flag: str, num_positions: int) -> None:
    """Every output position is fed by exactly N_SPATIAL unique source indices,
    no overlap, full num_tokens coverage."""
    num_tokens = num_positions * N_SPATIAL
    idx = torch.arange(num_tokens).reshape(num_tokens, 1)
    grouped = gather_by_position(idx, model_flag).reshape(num_positions, N_SPATIAL)
    seen = set()
    for p in range(num_positions):
        position_indices = grouped[p].tolist()
        assert len(position_indices) == N_SPATIAL
        assert len(set(position_indices)) == N_SPATIAL, f"{model_flag}: duplicate indices within position {p}"
        seen.update(position_indices)
    assert seen == set(range(num_tokens)), f"{model_flag}: coverage mismatch — {num_tokens - len(seen)} tokens missing"
    print(f"  {model_flag}: count/coverage OK ({num_positions} x {N_SPATIAL} = {num_tokens}, no overlap)")


def main() -> None:
    for model_flag, (num_positions, layout) in CASES.items():
        print(f"{model_flag} ({layout})")
        if model_flag in ("videomae", "vivit"):
            idx_fn = lambda p, s: p * N_SPATIAL + s
        else:
            idx_fn = lambda p, s: s * num_positions + p
        test_synthetic_recovery(model_flag, idx_fn, num_positions)
        test_count_coverage(model_flag, num_positions)

    print("\nAll guards passed.")


if __name__ == "__main__":
    main()
