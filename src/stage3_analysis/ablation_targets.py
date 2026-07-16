"""
Ablation target definitions — current run only (overwritten per run, not an
accumulating registry). run_ablation.py dumps this to JSON at run-start, and
the output parquet's run_tag is the permanent record of which targets a given
result used — see ablation_summary_7.py for how a past run's targets are
recovered after this file has moved on.

L5 clean8 — from scaffold_selection_consolidated.py's L5_x8k64_VM.csv (15/07/26),
all 8 gate-passed on both DFA and z, no soft-lock exclusion needed (unlike L7's
4061). CC brief: L5 Scaffold Ablation (clean8), Relative-Damage-Only, R + C1.
"""

TARGETS: dict[str, list[int]] = {
    "single_1394": [1394],
    "single_1784": [1784],
    "single_1919": [1919],
    "single_2468": [2468],
    "single_2577": [2577],
    "single_3246": [3246],
    "single_3325": [3325],
    "single_6006": [6006],
    "clean8":      [1394, 1784, 1919, 2468, 2577, 3246, 3325, 6006],
}

SINGLETON_TARGETS = [k for k in TARGETS if k.startswith("single_")]
GROUP_TARGETS     = [k for k in TARGETS if k not in set(SINGLETON_TARGETS)]
