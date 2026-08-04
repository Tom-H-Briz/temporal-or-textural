"""
Ablation target definitions — accumulating registry keyed by (dataset, layer).

Reverses the earlier "current run only, overwritten per run, not an
accumulating registry" convention (04/08, per Tom, explicit trade-off accepted):
that convention assumed strictly sequential runs, one hand-edit between each.
It broke down once K400 L5/L7 ablation and the L5+L7 dual ablation
(ablation_cross_l5_l7.py's separate L5_INDICES/L7_INDICES constants — same
problem, different file) needed to run in parallel — there's no safe window
to edit a shared "current" dict between two jobs starting close together.
Every config now has its own permanent entry; nothing overwrites anything else.

run_ablation.py dumps the resolved (dataset, layer) entry to
ablation_targets.json at run-start — that + the output parquet's run_tag is
still the permanent per-run record, same as before.

"all_members" replaces the old count-in-the-key convention ("all4"/"all7") —
member count varies per config, so a fixed key is simpler for callers
(run_ablation.py, ablation_summary.py, ablation_cross_l5_l7.py) to look up
generically regardless of which config they're running.
"""


def _make_targets(indices: list[int], group_key: str = "all_members") -> dict[str, list[int]]:
    """group_key defaults to "all_members" for new configs, but the two ssv2
    entries below pass their original "all7"/"all4" — those names are baked
    into ablation_target columns in already-run historical parquets, and
    ablation_summary.py can't re-analyze that data if the key it looks up no
    longer matches what's actually in the file (caught by running
    ablation_summary.py against the real l7_job7ep_k64 parquet, not just
    import-checking the registry)."""
    targets = {f"single_{i}": [i] for i in indices}
    targets[group_key] = list(indices)
    return targets


# (dataset, layer) -> target set. Provenance per entry:
TARGETS_BY_CONFIG: dict[tuple[str, int], dict[str, list[int]]] = {
    # 31/07/26 — L5's SSv2 "all7": scaffold_selection_consolidated.py members.
    ("ssv2", 5): _make_targets([358, 449, 917, 2093, 3516, 3938, 5004], group_key="all7"),
    # 31/07/26 — L7's SSv2 set: 6021/6032/5165 gate-passed on both DFA and z;
    # 3347 is the near-miss (position-locked, fails strict gate at C1 only,
    # 0.891/0.994 vs 0.90/1.0) — included deliberately per Tom to see if it
    # behaves like a member or a true non-member.
    ("ssv2", 7): _make_targets([3347, 5165, 6021, 6032], group_key="all4"),
    # 04/08/26 — K400 L5: position_lock_extraction.py's 03/08 manifest-based
    # re-run confirmed all 10, unchanged from the prior provisional set.
    ("kinetics400", 5): _make_targets([647, 729, 3003, 3421, 4152, 4534, 4551, 4887, 5248, 5377]),
    # 04/08/26 — K400 L7: same re-run, confirmed all 6, unchanged.
    ("kinetics400", 7): _make_targets([661, 1116, 1348, 4391, 4817, 4853]),
    # K400 L9 deliberately absent — 0 scaffold members, nothing to ablate.
}


def get_targets(dataset: str, layer: int) -> dict[str, list[int]]:
    key = (dataset, layer)
    if key not in TARGETS_BY_CONFIG:
        raise KeyError(f"No ablation targets registered for dataset={dataset!r} layer={layer}")
    return TARGETS_BY_CONFIG[key]


def singleton_targets(targets: dict[str, list[int]]) -> list[str]:
    return [k for k in targets if k.startswith("single_")]


def group_targets(targets: dict[str, list[int]]) -> list[str]:
    return [k for k in targets if k not in set(singleton_targets(targets))]
