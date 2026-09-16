# DFA/activation extraction — too many versions, flagged for a rework

Moved here 03/09/26 as a group, not fixed. `dfa_engine.py` (the shared engine — splice hook, `run_pixels`/`run_ablated`, the actual gradient×activation formula) deliberately **stays** at `src/stage3_analysis/dfa_engine.py`, not in this folder — it's imported via `from stage3_analysis.dfa_engine import DFAEngine` by ~24 files across the whole pipeline (`run_ablation.py`, `run_ablation_tf.py`, several notebooks, etc.), so moving it would ripple far outside this cleanup for no benefit. Everything in this folder is a *caller* of that engine, and the callers are where the duplication lives.

Contents, and how they overlap:

- `dfa_extraction_vm.py` / `dfa_extraction_tf.py` — the earliest per-clip DFA extractor, R-only, 32 SL classes. **Already confirmed superseded**: `build_interim_results.py`'s own audit found their output pool's "only consumer anywhere in src/notebooks is a narrow, unused validation notebook."
- `dfa_mass_delta_vm.py` / `dfa_mass_delta.py` — the current R vs C1(or C) vs A extractor per backbone. Still live — everything in `outputs/analysis/dfa_mass_delta*` traces back to these.
- `dfa_mass_delta_plots.py` — plotting for the above.
- `position_lock_extraction.py` — the current per-position DFA+z extractor. Its own docstring says it replaces two now-deleted scripts (`dfa_per_tubelet_mass.py`, `z_position_lock_extraction.py`, removed in the 31/07/26 refactor) — i.e. this is already the *second* consolidation of this exact idea, not the first.
- `position_lock_summary.py` — an earlier, different-thresholded gate over the same z-based columns that `scaffold_selection_consolidated.py` now gates independently. Superseded but not deleted.
- `cumulative_mass_diagnostic.py` / `cumulative_mass_diagnostic_tf.py` and `cumulative_mass_explore.py` / `cumulative_mass_explore_tf.py` — a diagnostic/exploratory pair per backbone, checking whether DFA reliability regime predicts top-10 magnitude collapse. Only the TF diagnostic's *output file* is still read (by `clip_shuffle_disruption.py`, as an optional reliability parquet) — none of these four scripts themselves are known to be re-run.

**The pattern worth fixing**: this is at least three generations of "extract DFA per clip, per condition, per backbone" (`dfa_extraction_*` → `dfa_mass_delta_*` → `position_lock_extraction`, with `position_lock_extraction` itself already a second-generation merge), each solving the same core problem — R-correct clip loop, per-condition frame perturbation, SAE splice, gradient×activation, write parquet — with its own copy of that loop. A rework should settle on one current extraction path per backbone and either delete or clearly archive the rest, rather than leaving every generation live in the same directory the active pipeline reads from.

Every file's `ROOT` path was updated (`.parent` × 4, not 3) to account for this folder being one level deeper than `src/stage3_analysis/`. `dfa_engine` imports are untouched — they're absolute (`stage3_analysis.dfa_engine`), so they resolve correctly regardless of where the importing file lives. No other logic was changed by the move itself.
