"""
Ablation target definitions — current run only (overwritten per run, not an
accumulating registry). run_ablation.py dumps this to JSON at run-start, and
the output parquet's run_tag is the permanent record of which targets a given
result used — see ablation_summary_7.py for how a past run's targets are
recovered after this file has moved on.

L7 all4 — from scaffold_selection_consolidated.py's L7_x8k64_VM_all_features.csv
(31/07/26, post-30/07 bias-fix, job7ep/k64): 6021/6032/5165 gate-passed on both
DFA and z; 3347 is the near-miss — position-locked (tubelet 7, stable across
R/C1/A) but fails the strict gate on DFA share/consistency at C1 specifically
(0.891/0.994, just under the 0.90/1.0 thresholds). Included deliberately per
Tom (31/07/26) to see if it behaves like a member or like a true non-member.

L5's "all7" set (358/449/917/2093/3516/3938/5004) was run and analyzed
31/07/26 — see ablation_results_long_l5_job7ep_k64.parquet — not repeated here
since this file holds one layer's targets at a time (see docstring above).
"""

TARGETS: dict[str, list[int]] = {
    "single_3347": [3347],
    "single_5165": [5165],
    "single_6021": [6021],
    "single_6032": [6032],
    "all4":        [3347, 5165, 6021, 6032],
}

SINGLETON_TARGETS = [k for k in TARGETS if k.startswith("single_")]
GROUP_TARGETS     = [k for k in TARGETS if k not in set(SINGLETON_TARGETS)]
