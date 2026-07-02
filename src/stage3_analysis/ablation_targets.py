"""
Ablation target definitions — single source of truth.

Imported by run_ablation.py (writer) and ablation_summary.py (reader).
run_ablation.py also dumps this to JSON at run-start for manual inspection.
"""

TARGETS: dict[str, list[int]] = {
    "single_1842": [1842],
    "single_5578": [5578],
    "single_1996": [1996],
    "single_3513": [3513],
    "single_1990": [1990],
    "single_3558": [3558],
    "single_5552": [5552],
    "iso4061":     [4061],
    "clean7":      [1842, 5578, 1996, 3513, 1990, 3558, 5552],
    "all8":        [1842, 5578, 1996, 3513, 1990, 3558, 4061, 5552],
}

SINGLETON_TARGETS = [k for k in TARGETS if k.startswith("single_")]
