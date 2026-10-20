# Brief: push vs pull through the MLM contrast → groups → ablation chain, with a hand-feature check (20 Oct 2026)

**For:** the agent building the scripts. **Review:** checked against §7, with stops at §6. The user confirms after review. Read `CLAUDE.md` first (§5).

**Background:** `findings/umt_mlm_feature_contrast_findings_191026.md`, `mlm_feature_contrast_brief_191026.md`, `mlm_group_ablation_brief_201026.md`. This extends `src/stage3_analysis/umt_mlm_feature_contrast.py` and `src/stage3_analysis/umt_mlm_group_ablation.py` to two new pairs. **Their existing outputs must not change.**

## 0. Goal

Run push vs pull through the same chain the three clean pairs went through:
- feature contributions to the MLM pair word under R / C / Rev;
- the gated / flip / no-change groups;
- group, control and single-feature ablations on held-out clips, measured on the pair word and on the other words.

Then check whether the gated group is made of **hand features**.

**Why (context, not for you to interpret):**
- **Up/down** is decided by gated features (direction detectors); **open/close** by flips (shared, ubiquitous features read with a different sign).
- **Push and pull** share the same motion and differ in where the hand is relative to the object. **User's prediction:** if the shared features encode generic change or motion, they can't separate push from pull, and push/pull needs **gated hand-object features**.
- The hand-feature overlap tests that prediction directly.

## 1. Decisions already made by the user (do not revisit)

| Item | Decision |
|---|---|
| Pairs | **Both push/pull pairs:** (93, 86) "Pushing / Pulling something from left to right" and (94, 87) "… from right to left". word_a / word_b = **pushing / pulling** (single word-pieces; the captions differ only in the verb) |
| Method | **Identical to the clean-pair runs** at every step: L9 SAE, Phase 3's gradient convention, T_ab log-odds at the masked verb, R / C / Rev, the selection table, the §2 group rules of `mlm_group_ablation_brief_201026.md` (5 largest of each type, whatever their rank), a mass-matched control per group, singles, ablation only (dose 1.0), the pair word plus every other word, held-out clips under R and Rev |
| Hand features | **Added as an explicit check** (§3 D4) |
| No VTM comparison | Push/pull is not in Phase 3; skip the contrast script's `compare` stage |

**Val clip counts:** 93: 250, 86: 170, 94: 236, 87: 125 (781 in total, before mask skips).

## 2. Clips and split (new: push/pull isn't in Phase 3)

- **Clip list:** every val clip of the two pairs' four classes, built the way `umt_vision_pair_contrast.pair_clips` does (template normalisation, sorted by clip id).
- **Held-out split: Phase 3's rule,** reused so the method matches: `umt_bridge_phase3.sample_g34_clips`. **Per pair, the first 37 clips by clip id** (Phase 3 used 300 // 8 = 37 per pair) are `g34` (held out); the rest are `selection`. Import the function; don't re-implement it. It takes a total and divides by the number of pairs, so call it with **n_total = 74** (2 pairs × 37), not 300. Write the split to `split.csv`.
- Masked verb: first occurrence of "pushing" / "pulling", with missing or repeated words skipped and counted, exactly as `clip_texts` / `masked_pos` do.

## 3. Deliverables (extend the existing scripts; don't fork them)

**Generalise rather than copy.** Add config options to the two existing scripts so they can run these pairs, with the defaults reproducing the current behaviour exactly:
- **`umt_mlm_feature_contrast.py`:** a clip source option (Phase 3 files, the current default, or "val + Phase 3 split rule" for new pairs); push/pull added to the pairs dict; an output folder option. `compare` stays Phase 3-only.
- **`umt_mlm_group_ablation.py`:** the pairs and the contrast folder from config. Keep the `EXPECTED` assert for the three clean pairs. For new pairs, print the groups and stop at checkpoint §6.1 for review, since there's no expected table.

### D1. Contrast extraction → `outputs/analysis/umt_mlm_feature_contrast_pushpull/`
Same outputs as the clean-pair run: `sparse/{R,C,Rev}`, `per_clip.parquet`, `skipped_clips.csv`; then `select` → `mlm_pair_selection.csv`, `selection_table.parquet`.

### D2. Groups → `outputs/analysis/umt_mlm_group_ablation_pushpull/groups.csv`
§2 rules of the ablation brief, unchanged. **If a type has fewer than 5 qualifying features, take what exists and report the count.** Don't relax the thresholds.

### D3. Ablation + summary → the same folder
Identical to the clean-pair run: 22 edits per clip-condition, R and Rev, on the held-out clips. Same output files and summary columns, including `pred_sep` / `pred_bias` and `sep_over_pred` / `bias_over_pred`.

### D4. Hand-feature check → `hand_overlap.csv`
Source: `outputs/analysis/hand_tracker_alignment/umt_L9.csv` (columns `feature, overlap, track, dist, coact, dfa_share`; one row per L9 feature).
- **Hand feature** = `coact ≥ 0.5` **and** `overlap > 0.4`: the definition used in `findings/phase3_layer_grid_hand_features_findings_151026.md` §6b. Report how many features meet it. Also report #2782, the template.
- **Per group member, for the two push/pull pairs *and* the three clean pairs:** overlap, track, coact, and whether it's a hand feature.
- **Per group:** the count of hand features, and their share of the group's importance.
- **Enrichment:** the fraction of hand features among each pair's top 1% by importance, against the fraction among all active features.

The hand alignment was computed on **move up/down clips**. Note that in the output.

## 4. Running
- One SLURM chain: contrast `extract` + `select` (781 clips × 3 conditions, about 79 ms each, so a few minutes), then `groups`, then **stop** (§6.1). After review, `check` + `ablate` + `summarise` + the hand check.
- `MODE` stages as in the existing scripts, sized time limits (≤ 1 hour each), **no smoke run**.
- The decode-determinism check stays report-only.

## 5. Conventions
- `uv`; functional Python; a CFG dict at the top; functions of 30–50 lines; simple code.
- **No more than 20 lines written at a time; a bullet-point rationale with every code change.**
- **Import, don't copy;** generalise the existing scripts with defaulted options. **Regression check:** run the existing scripts' default paths on 2 clips and show the outputs are unchanged.
- fp32, TF32 off. Don't commit, push or chmod; give the git command and its risks. No Co-Authored-By trailer.
- No findings or interpretations in files or comments. **If something isn't covered here, stop and ask.**

## 6. Checkpoints (stop and report at each)
1. **After contrast + groups:** `split.csv` counts, skipped clips, R/C/Rev pair accuracy (from `per_clip.parquet`), the groups with ranks, shares, activation ratios and evidence (and any group with fewer than 5), the controls with mass gaps, and `hand_overlap.csv` for the group members.
2. **After ablation:** the outputs. Numbers only.

## 7. Acceptance checklist
1. **Defaults unchanged:** with default config, both existing scripts reproduce their current outputs (2-clip regression check, |diff| < 1e-6, within one decode).
2. **Split:** made by Phase 3's `sample_g34_clips` (imported), 37 per pair; held-out clips never used for selection or groups.
3. **Masks:** the masked token is "pushing" or "pulling" on every kept clip; skips counted.
4. **Orientation:** under R, mean T_ab > 0 on push clips and < 0 on pull clips, for both pairs. Report rather than fix if not.
5. **Groups:** the §2 rules applied unchanged; any shortfall reported.
6. **Ablation:** dose 1.0 asserted; empty edit = baseline; separation and bias predictions present.
7. **Hand check:** the hand-feature definition applied as stated; reported for all five pairs.
8. Accuracy and gap measures in separate tables.

## 8. Out of scope
- The cross pairs (push LtR vs pull RtL, etc.) and push LtR vs RtL.
- Text-leak pairs; other layers; VTM / VTC.
- Recomputing the hand alignment on push/pull clips (a possible follow-up).
- Any interpretation or write-up.
