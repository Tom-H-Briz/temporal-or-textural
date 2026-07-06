"""
Ablation target definitions — single source of truth.

Imported by run_ablation.py (writer) and ablation_summary.py (reader).
run_ablation.py also dumps this to JSON at run-start for manual inspection.
"""

TARGETS: dict[str, list[int]] = {
    "single_6135": [6135],
    "single_2197": [2197],
    "single_1535": [1535],
    "single_308": [308],
    "all4":      [6135, 2197, 1535, 308],
    
}

SINGLETON_TARGETS = [k for k in TARGETS if k.startswith("single_")]
