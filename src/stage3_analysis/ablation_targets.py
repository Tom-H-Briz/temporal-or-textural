"""
Ablation target definitions — current run only (overwritten per run, not an
accumulating registry). run_ablation.py dumps this to JSON at run-start, and
the output parquet's run_tag is the permanent record of which targets a given
result used — see ablation_summary_7.py for how a past run's targets are
recovered after this file has moved on.

L5 all7 — from scaffold_selection_consolidated.py's L5_x8k64_VM.csv (31/07/26,
post-30/07 bias-fix, job7ep/k64), all 7 gate-passed on both DFA and z. Replaces
the pre-fix "clean8" set this file held previously (1394/1784/1919/2468/2577/
3246/3325/6006) — that was from the legacy job64 SAE, a different dictionary;
those feature indices have no relationship to the current SAE's indices.
"""

TARGETS: dict[str, list[int]] = {
    "single_358":  [358],
    "single_449":  [449],
    "single_917":  [917],
    "single_2093": [2093],
    "single_3516": [3516],
    "single_3938": [3938],
    "single_5004": [5004],
    "all7":        [358, 449, 917, 2093, 3516, 3938, 5004],
}

SINGLETON_TARGETS = [k for k in TARGETS if k.startswith("single_")]
GROUP_TARGETS     = [k for k in TARGETS if k not in set(SINGLETON_TARGETS)]
