"""
TF-K400 vs TF-SSv2 collation — the three comparisons requested 03/10:

  1. Perturbed-accuracy table  (R/C/A overall top-1, both datasets)
  2. Four-bucket noise-vs-sign-flip (ssv2_tf L7 vs k400_tf L5/L7/L9)
  3. Position locking under the canonical position_lock_summary.py 5-criterion
     definition (z-based), plus relaxed-threshold context

Inputs are all on local disk (synced per sync manifest 03/10). SSv2 position-lock
uses the pre-merge legacy z_position_lock scores (same quantities, old column
names); K400 uses the merged position_lock_extraction.py scores. Numbers as
found — no interpretation here.

Usage: uv run python src/stage3_analysis/tf_k400_vs_tf_ssv2_collation.py
Output: findings/tf_k400_vs_tf_ssv2_findings_031026.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

CFG = {
    "acc": {
        "k400": {c: ROOT / f"outputs/stage1_class_selection_TF_kinetics/per_class_accuracy_TF_kinetics_{c}.csv"
                 for c in ("R", "C", "A")},
        "ssv2": {"R": ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF.csv",
                 "C": ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF_C.csv",
                 "A": ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF_A.csv",
                 "B": ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF_B.csv"},
    },
    "bucket_dir": ROOT / "outputs/analysis/shuffle_reduction_composition",
    "bucket_files": {"ssv2_tf L7": "ssv2_tf_clip_shuffle_disruption.csv",
                     "k400_tf L5": "k400_tf_l5_clip_shuffle_disruption.csv",
                     "k400_tf L7": "k400_tf_l7_clip_shuffle_disruption.csv",
                     "k400_tf L9": "k400_tf_l9_clip_shuffle_disruption.csv"},
    "pos_lock": {"k400": {L: ROOT / f"outputs/analysis/position_lock/position_lock_scores_timesformer_kinetics400_l{L}_job7ep_k64.csv"
                          for L in (5, 7, 9)},
                 "ssv2": {L: ROOT / f"outputs/analysis/z_position_lock/z_position_lock_scores_timesformer_l{L}.csv"
                          for L in (5, 7, 9)}},
    "mass": {  # full-vector bucket mass: parquet + R-acc eligibility per config
        ("k400", L): (ROOT / f"outputs/analysis/dfa_mass_delta_tf/dfa_mass_delta_tf_kinetics400_l{L}_job7ep_k64.parquet",
                      ROOT / "outputs/stage1_class_selection_TF_kinetics/per_class_accuracy_TF_kinetics_R.csv")
        for L in (5, 7, 9)
    } | {("ssv2", 7): (ROOT / "outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet",
                       ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF.csv")},
    "out_md": ROOT / "findings/tf_k400_vs_tf_ssv2_findings_031026.md",
}


def accuracy_section() -> list[str]:
    lines = ["## 1. Perturbed accuracy — TF K400 vs TF SSv2 (full val, overall top-1)", ""]
    overall = {}
    for ds, conds in CFG["acc"].items():
        overall[ds] = {}
        for cond, path in conds.items():
            df = pd.read_csv(path)
            overall[ds][cond] = df["correct"].sum() / df["total"].sum()
    lines += ["| dataset | R (real) | C (shuffle) | A (still) | ΔC−R | ΔA−R |", "|---|---|---|---|---|---|"]
    for ds in ("ssv2", "k400"):
        r, c, a = overall[ds]["R"], overall[ds]["C"], overall[ds]["A"]
        lines.append(f"| TF {ds} | {r:.4f} | {c:.4f} | {a:.4f} | {c-r:+.4f} | {a-r:+.4f} |")
    if "B" in overall["ssv2"]:
        b = overall["ssv2"]["B"]
        lines += ["", f"SSv2-only extra as found: B (first/last) = {b:.4f} "
                      f"(ΔB−R {b-overall['ssv2']['R']:+.4f}). K400 ran no B condition."]
    return lines


BUCKET_COLS = ["frac_noise", "frac_sign_flip", "frac_decrease", "frac_increase"]


def bucket_section() -> list[str]:
    lines = ["## 2. Four-bucket — invariant (noise) vs sign-flip", "",
             "Pooled = clip-mean. Class-weighted = mean-per-class first, then equal-weighted "
             "across classes (B1 weighting).", ""]
    for name, fname in CFG["bucket_files"].items():
        df = pd.read_csv(CFG["bucket_dir"] / fname)
        per_class = df.groupby("class_id")[BUCKET_COLS].mean()
        pooled = {c: df[c].mean() for c in BUCKET_COLS}
        cw = {c: per_class[c].mean() for c in BUCKET_COLS}
        lines.append(f"### {name} — {len(df):,} clips, {df['class_id'].nunique()} classes, "
                     f"P(correct|shuffle) = {df['correct_under_shuffle'].mean():.4f}")
        lines.append("| weighting | noise | sign_flip | decrease | increase |")
        lines.append("|---|---|---|---|---|")
        lines.append("| pooled | " + " | ".join(f"{pooled[c]:.4f}" for c in BUCKET_COLS) + " |")
        lines.append("| class-weighted | " + " | ".join(f"{cw[c]:.4f}" for c in BUCKET_COLS) + " |")
        lines.append("")
    return lines


# Canonical position_lock_summary.py thresholds — copied verbatim, not reinvented.
MIN_SHARE, MIN_FRAC, MIN_TOTAL, MIN_TOP = 0.90, 0.90, 0.05, 0.02

_RENAME = {"n_clips": "n_clips_z", "total_abs_R": "total_abs_z_R", "top_abs_R": "top_abs_z_R",
           "pos_consistent": "pos_consistent_z", "mean_per_clip_share_R": "mean_per_clip_share_z_R",
           "frac_clips_matching_mode_R": "frac_clips_matching_mode_z_R",
           "mode_frame_R": "mode_frame_z_R"}


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["mean_per_clip_share_z_R"] >= MIN_SHARE)
              & (df["frac_clips_matching_mode_z_R"] >= MIN_FRAC)
              & df["pos_consistent_z"].astype(bool)
              & (df["total_abs_z_R"] >= MIN_TOTAL)
              & (df["top_abs_z_R"] >= MIN_TOP)]


def pos_lock_section() -> list[str]:
    lines = ["## 3. Position locking — canonical 5-criterion definition (z-based)", "",
             "Criteria (position_lock_summary.py, verbatim): mean_per_clip_share_R >= 0.90, "
             "frac_clips_matching_mode >= 0.90, mode consistent across R/C/A, "
             "total_abs_R >= 0.05, top_abs_R >= 0.02. Counts are (class, feature) pairs; "
             "'features' = unique feature indices among those pairs.", ""]
    lines += ["| dataset | layer | active pairs | canonical-locked pairs | unique locked features | "
              "max frac_mode (active) | relaxed ≥0.70 pairs |", "|---|---|---|---|---|---|---|"]
    for ds in ("k400", "ssv2"):
        for L, path in CFG["pos_lock"][ds].items():
            df = pd.read_csv(path).rename(columns=_RENAME)
            active = df[df["total_abs_z_R"] > 0]
            locked = _canonical(active)
            relaxed = active[active["frac_clips_matching_mode_z_R"] >= 0.70]
            lines.append(f"| TF {ds} | {L} | {len(active):,} | {len(locked):,} | "
                         f"{locked['feature_idx'].nunique():,} | "
                         f"{active['frac_clips_matching_mode_z_R'].max():.3f} | {len(relaxed):,} |")
    lines += ["", "Readings behind the table (verified, not assumed):",
              "- SSv2's ≥0.70 tail is NOT small-n granularity: those pairs have n_clips "
              "min 47 / median 111 (L7). But their mean_per_clip_share_R is only ~0.20-0.26 — "
              "the modal frame is consistent yet carries a minority of each clip's mass, "
              "which is why the canonical share≥0.90 bar yields 0 pairs on SSv2 too.",
              "- K400's maximum frac_mode across all 393,216 active pairs is 0.469 (L7): no "
              "pair even reaches 0.5, with n_clips 45-50 per class — the bar was reachable "
              "in principle and simply never met. No frame locking on TF-K400, weak-or-none "
              "on TF-SSv2."]
    return lines


def _clip_buckets(abs_r: np.ndarray, abs_c: np.ndarray, s_r: np.ndarray, s_c: np.ndarray):
    """Vectorised top10_detail taxonomy over ALL recruited features: noise-first,
    sign-flip-second, decrease/increase residual (same locked order, 5% band)."""
    recruited = abs_r > 0
    rel = np.where(recruited, (abs_c - abs_r) / np.where(recruited, abs_r, 1.0), 0.0)
    noise = recruited & (np.abs(rel) <= 0.05)
    sf = recruited & ~noise & (np.sign(s_r) != np.sign(s_c))
    dec = recruited & ~noise & ~sf & (rel < 0)
    inc = recruited & ~noise & ~sf & (rel > 0)
    return recruited, noise, sf, dec, inc


def mass_bucket_section() -> list[str]:
    """Fraction of total DFA R-mass per bucket over the FULL feature set, plus the
    recruitment question: how much mass the non-top-10 tail holds at all, and how
    much shuffle-only mass exists in never-recruited features."""
    lines = ["## 4. Bucket composition by DFA mass, full feature set (not top-10, not counts)", ""]
    for (ds, L), (pq_path, acc_path) in CFG["mass"].items():
        df = pd.read_parquet(pq_path)
        acc = pd.read_csv(acc_path)
        eligible = set(acc.loc[acc["accuracy"] >= 0.40, "class_id"])
        df = df[df["class_id"].isin(eligible)]
        r = np.stack(df["signed_vec_R"].to_numpy()).astype(np.float32)
        c = np.stack(df["signed_vec_C"].to_numpy()).astype(np.float32)
        rec, noise, sf, dec, inc = _clip_buckets(np.abs(r), np.abs(c), r, c)
        tot = np.abs(r * rec).sum(axis=1)
        lines.append(f"### TF {ds} L{L} — {len(df):,} clips")
        lines += _mass_rows(r, c, rec, noise, sf, dec, inc, tot)
    lines += ["Readings (measurements, not claims):",
              "- Cross-check: the top-10 row divided by its mass share reproduces "
              "clip_shuffle_disruption's top-10 fractions (e.g. K400 L7 0.019/0.169=0.112 "
              "vs 0.109 reported) — two independent implementations agree.",
              "- Recruitment is near-universal: 17-77 of 6,144 features per clip have zero "
              "R DFA mass; shuffle-only recruitment carries ~0.01-0.03% of mass.",
              "- Top-10 features hold 13-17% of clip R-mass on K400, 7.5% on SSv2; the "
              "tail holds 83-93%.",
              "- Count vs mass: 20-26% of recruited features flip sign (6-11% of mass); "
              "decrease holds 0.47-0.52 of mass on both datasets.",
              "- K400 by layer: sign-flip mass 0.110 (L5) -> 0.082 (L7) -> 0.058 (L9); "
              "noise mass 0.071 -> 0.090."]
    return lines


def _mass_rows(r, c, rec, noise, sf, dec, inc, tot) -> list[str]:
    abs_r, abs_c = np.abs(r), np.abs(c)
    # boolean-masked rows flatten on fancy indexing — multiply instead, keep rows
    mass = lambda m: ((abs_r * m).sum(axis=1) / tot).mean()
    cnt = lambda m: (m.sum(axis=1) / rec.sum(axis=1).clip(min=1)).mean()
    # top-10 by |R| per clip — the recruitment head
    head_idx = np.argsort(-abs_r, axis=1)[:, :10]
    head_mask = np.zeros_like(rec)
    np.put_along_axis(head_mask, head_idx, True, axis=1)
    tail = rec & ~head_mask
    shuffle_only = (~rec) & (abs_c > 0)
    rows = ["| slice | noise | sign_flip | decrease | increase | share of clip R-mass |",
            "|---|---|---|---|---|---|",
            f"| all recruited (count view) | {cnt(noise):.4f} | {cnt(sf):.4f} | {cnt(dec):.4f} | {cnt(inc):.4f} | 1.0 |",
            f"| all recruited (mass view) | {mass(noise):.4f} | {mass(sf):.4f} | {mass(dec):.4f} | {mass(inc):.4f} | 1.0 |",
            f"| top-10 by R-mass | {mass(noise & head_mask):.4f} | {mass(sf & head_mask):.4f} | "
            f"{mass(dec & head_mask):.4f} | {mass(inc & head_mask):.4f} | {((abs_r * head_mask).sum(axis=1)/tot).mean():.4f} |",
            f"| tail (recruited, non-top-10) | {mass(noise & tail):.4f} | {mass(sf & tail):.4f} | "
            f"{mass(dec & tail):.4f} | {mass(inc & tail):.4f} | {((abs_r * tail).sum(axis=1)/tot).mean():.4f} |"]
    rows.append(f"never-recruited under R: {int((~rec).sum(axis=1).mean())}/6144 features avg; "
                f"of those, shuffle fires {int(shuffle_only.sum(axis=1).mean())} avg, carrying "
                f"{((abs_c * shuffle_only).sum(axis=1)/tot).mean():.4f} of a clip's R-mass as new C-mass")
    rows.append("")
    return rows


def top_locked_listing() -> list[str]:
    """Canonical bar yields nothing on either dataset, so show the strongest
    evidence that DOES exist: top pairs by frac_mode (SSv2) vs the ceiling (K400)."""
    lines = ["### Strongest position-lock evidence available, L7 (canonical bar: 0 pairs both datasets)", ""]
    for ds in ("ssv2", "k400"):
        df = pd.read_csv(CFG["pos_lock"][ds][7]).rename(columns=_RENAME)
        active = df[df["total_abs_z_R"] > 0]
        top = active.nlargest(10, "frac_clips_matching_mode_z_R")
        lines.append(f"**TF {ds} L7** — top 10 active pairs by frac_mode:")
        lines.append("| class | feature | mode_frame | frac_mode | mean_share | n_clips |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in top.iterrows():
            lines.append(f"| {int(r['class_id'])} | {int(r['feature_idx'])} | {int(r['mode_frame_z_R'])} "
                         f"| {r['frac_clips_matching_mode_z_R']:.2f} | {r['mean_per_clip_share_z_R']:.2f} "
                         f"| {int(r['n_clips_z'])} |")
        lines.append("")
    return lines


def dec_inc_section() -> list[str]:
    """Decrease vs increase under C, differentiated on three axes: net mass
    balance (new vs redistributed mass), stillness (A) retention relative to the
    clip average, and cross-clip identity stability of the bucket assignment."""
    lines = ["## 6. Decrease vs increase — what differentiates them", ""]
    for (ds, L), (pq_path, acc_path) in CFG["mass"].items():
        df = pd.read_parquet(pq_path)
        acc = pd.read_csv(acc_path)
        df = df[df["class_id"].isin(set(acc.loc[acc["accuracy"] >= 0.40, "class_id"]))]
        r = np.stack(df["signed_vec_R"].to_numpy()).astype(np.float32)
        c = np.stack(df["signed_vec_C"].to_numpy()).astype(np.float32)
        a = np.stack(df["signed_vec_A"].to_numpy()).astype(np.float32)
        ar, ac, aa = np.abs(r), np.abs(c), np.abs(a)
        rec, noise, sf, dec, inc = _clip_buckets(ar, ac, r, c)
        lines.append(f"### TF {ds} L{L}")
        lines += _balance_rows(rec, dec, inc, ar, ac)
        lines += _retention_rows(rec, noise, sf, dec, inc, ar, aa, r, a)
        lines += _stability_rows(rec, noise, sf, dec, inc)
    return lines


def _balance_rows(rec, dec, inc, ar, ac) -> list[str]:
    tot_r = (ar * rec).sum(axis=1)
    tot_c = (ac * rec).sum(axis=1)
    net = ((tot_c - tot_r) / tot_r).mean()
    c_dec = ((ac * dec).sum(axis=1) / tot_c).mean()
    c_inc = ((ac * inc).sum(axis=1) / tot_c).mean()
    lost = ((ar * dec).sum(axis=1) / tot_r).mean()
    return [f"- Net per-clip causal-mass change under shuffle: C/R = {net:+.4f}",
            f"- Decrease-bucket features: {lost:.3f} of R mass -> {c_dec:.3f} of C mass",
            f"- Increase-bucket features: {c_inc:.3f} of C mass"]


def _retention_rows(rec, noise, sf, dec, inc, ar, aa, r_s, a_s) -> list[str]:
    """Per C-bucket under stillness (A): (i) |A|/|R| magnitude ratio relative to
    the clip average, and (ii) mass-weighted fraction of retained |A| whose SIGN
    agrees with R. Reported jointly because magnitude alone cannot indicate
    functional stability — a feature may retain magnitude while its signed
    contribution (the logit it favours) reverses."""
    base = (aa * rec).sum() / (ar * rec).sum()
    parts = []
    for n, m in (("noise", noise), ("sign_flip", sf), ("decrease", dec), ("increase", inc)):
        mag = ((aa * m).sum() / (ar * m).sum()) / base
        m_a = m & (aa > 0)
        agree = (np.sign(r_s) == np.sign(a_s)) & m_a
        sa = (aa * agree).sum() / (aa * m_a).sum()
        parts.append(f"{n} mag {mag:.2f} / sign-agree {sa:.3f}")
    return ["- Under stillness (A), per C-bucket: " + "; ".join(parts),
            "- Note: the two columns are separate quantities. Magnitude retention is not "
            "functional stability. Sign-flip bucket ratios rest on 6-11% of clip mass (§4)."]


def _stability_rows(rec, noise, sf, dec, inc, min_clips: int = 10) -> list[str]:
    """Is a feature's bucket the same across clips, or per-clip idiosyncratic?"""
    n_rec = rec.sum(axis=0)                      # per-feature recruitment count
    votes = np.stack([noise.sum(axis=0), sf.sum(axis=0), dec.sum(axis=0), inc.sum(axis=0)])
    stable_feats = n_rec >= min_clips
    dom = votes[:, stable_feats].argmax(axis=0)
    stability = votes[:, stable_feats].max(axis=0) / n_rec[stable_feats]
    names = ("noise", "sign_flip", "decrease", "increase")
    parts = [f"{names[b]}: {int((dom == b).sum())} feats "
             f"({int(((dom == b) & (stability >= 0.7)).sum())} stable≥0.7)"
             for b in range(4)]
    return [f"- Identity across clips ({int(stable_feats.sum())} features recruited in ≥{min_clips} clips): "
            + "; ".join(parts)]


def sign_flip_focus_section() -> list[str]:
    """Per-clip sign-flip mass: distributional test K400 vs SSv2 (L7), plus the
    behavioural meaning from the already-computed logit coefficients."""
    from scipy import stats
    lines = ["## 5. Sign-flip focus — amount vs behavioural meaning (L7)", ""]
    sf = {}
    for ds, (pq, acc_csv) in (("k400", CFG["mass"][("k400", 7)]), ("ssv2", CFG["mass"][("ssv2", 7)])):
        df = pd.read_parquet(pq)
        acc = pd.read_csv(acc_csv)
        df = df[df["class_id"].isin(set(acc.loc[acc["accuracy"] >= 0.40, "class_id"]))]
        r = np.stack(df["signed_vec_R"].to_numpy()).astype(np.float32)
        c = np.stack(df["signed_vec_C"].to_numpy()).astype(np.float32)
        ar, ac = np.abs(r), np.abs(c)
        rec = ar > 0
        rel = (ac - ar) / np.where(rec, ar, 1.0)
        flip = rec & ~(np.abs(rel) <= 0.05) & (np.sign(r) != np.sign(c))
        sf[ds] = (ar * flip).sum(axis=1) / (ar * rec).sum(axis=1)
    k, s = sf["k400"], sf["ssv2"]
    u = stats.mannwhitneyu(k, s, alternative="two-sided")
    pooled_sd = np.sqrt(((len(k)-1)*k.var() + (len(s)-1)*s.var()) / (len(k)+len(s)-2))
    lines += [f"- Amount: K400 mean {k.mean():.4f} (sd {k.std():.4f}) vs SSv2 {s.mean():.4f} "
              f"(sd {s.std():.4f}); Mann-Whitney p={u.pvalue:.1e}, Cohen's d={(k.mean()-s.mean())/pooled_sd:+.2f}.",
              "- Association with P(correct|shuffle), frac_sign_flip term, from the "
              "established clip_shuffle_disruption fits:",
              "  - SSv2: −8.23 pooled / −6.01 class-FE (negative).",
              "  - K400: +2.49 pooled / +2.73 class-FE (positive).",
              "- These are associations; direction of causation is not established here."]
    return lines


TOP12_SSV2 = [3029, 1517, 2090, 2057, 2156, 1588, 4590, 3813, 6029, 622, 1371, 4134]  # run_ablation_tf.py


def _feature_rates(pq_path, acc_path):
    """Per-feature behaviour rates across recruited clips: flip / noise / dec /
    inc rate, mean mass share, recruitment count. The clip-to-clip stability lens."""
    df = pd.read_parquet(pq_path)
    acc = pd.read_csv(acc_path)
    df = df[df["class_id"].isin(set(acc.loc[acc["accuracy"] >= 0.40, "class_id"]))]
    r = np.stack(df["signed_vec_R"].to_numpy()).astype(np.float32)
    c = np.stack(df["signed_vec_C"].to_numpy()).astype(np.float32)
    ar, ac = np.abs(r), np.abs(c)
    rec, noise, sf, dec, inc = _clip_buckets(ar, ac, r, c)
    n = rec.sum(axis=0)
    tot = (ar * rec).sum(axis=1)
    share = (ar * rec / tot[:, None]).sum(axis=0)          # summed share of clip mass
    return pd.DataFrame({"feature_idx": np.arange(ar.shape[1]), "n_clips": n,
                         "flip_rate": np.divide(sf.sum(0), n, out=np.zeros_like(n, float), where=n > 0),
                         "noise_rate": np.divide(noise.sum(0), n, out=np.zeros_like(n, float), where=n > 0),
                         "dec_rate": np.divide(dec.sum(0), n, out=np.zeros_like(n, float), where=n > 0),
                         "inc_rate": np.divide(inc.sum(0), n, out=np.zeros_like(n, float), where=n > 0),
                         "mass_share_sum": share})


def signflip_candidates_section() -> list[str]:
    """Confirm/deny: sign-flip behaviour is clip-to-clip STABLE for a subset of
    features (the routed-mass hypothesis), validate the ranking against the
    known SSv2 TOP12, and list the TF-K400 equivalents per layer for ablation."""
    lines = ["## 7. Sign-flip stability and ablation candidates", ""]
    out_rows = []
    ssv2 = _feature_rates(*CFG["mass"][("ssv2", 7)])
    lines += _stability_summary(ssv2, "TF ssv2 L7 (validation set)")
    top_mass = ssv2[ssv2["n_clips"] >= 100].nlargest(12, "mass_share_sum")
    overlap = sorted(set(top_mass["feature_idx"]) & set(TOP12_SSV2))
    top12_stats = ssv2[ssv2["feature_idx"].isin(TOP12_SSV2)]
    lines += [f"- Method check: the registered TOP12 are the dictionary's heaviest features — "
              f"top-12 by mass_share_sum recovers {len(overlap)}/12 -> {overlap}; their mass "
              f"ranks sit at the very top of 6,144, and their flip rates (0.10-0.23) are "
              f"unremarkable vs population median "
              f"{ssv2.loc[ssv2['n_clips'] >= 100, 'flip_rate'].median():.2f}. The stable, "
              "targetable property is MASS DOMINANCE, not bucket role.", ""]
    lines += _table(top_mass, "TF ssv2 L7 — top 12 by mass (the TOP12 criterion, as found)")
    out_rows = []
    for L in (5, 7, 9):
        fr = _feature_rates(*CFG["mass"][("k400", L)])
        lines += _stability_summary(fr, f"TF k400 L{L}")
        top = fr[fr["n_clips"] >= 100].nlargest(12, "mass_share_sum")
        out_rows += [{"layer": L, **row} for _, row in top.iterrows()]
        lines += _table(top, f"TF k400 L{L} — top 12 by mass_share_sum (ablation candidates)")
    pd.DataFrame(out_rows).to_csv(ROOT / "outputs/tf_k400/signflip_ablation_candidates.csv", index=False)
    lines += ["", "Candidate list persisted -> outputs/tf_k400/signflip_ablation_candidates.csv "
                  "(k64/x8 dictionary indices — layer-specific, never comparable across configs).",
                  "- Findings as measured: (i) no feature on either dataset flips in >=40% of "
                  "its recruited clips — max flip_rate 0.37-0.38; per-clip bucket role is not a "
                  "stable feature property; (ii) noise (invariant) behaviour likewise has no "
                  "stable population — max noise_rate 0.08-0.11, zero noise-dominant features; "
                  "(iii) what IS stable clip-to-clip is which features carry the mass."]
    return lines


def _stability_summary(fr: pd.DataFrame, name: str) -> list[str]:
    """Is one behaviour rate per feature more concentrated than the others?"""
    rates = fr[["flip_rate", "noise_rate", "dec_rate", "inc_rate"]]
    dom = rates.idxmax(axis=1).str.replace("_rate", "")
    dom_rate = rates.max(axis=1)
    ok = fr["n_clips"] >= 100
    parts = [f"{b}: n={int((dom[ok] == b).sum())}, median rate {dom_rate[ok][dom[ok] == b].median():.2f}"
             for b in ("flip", "noise", "dec", "inc")]
    return [f"- {name} — dominant behaviour per feature (n_clips>=100): " + "; ".join(parts)
            + f"; max flip_rate {fr.loc[ok, 'flip_rate'].max():.2f}, "
              f"max noise_rate {fr.loc[ok, 'noise_rate'].max():.2f}"]


def _table(top: pd.DataFrame, title: str) -> list[str]:
    lines = [f"**{title}**", "| feature | flip_rate | noise | dec | inc | n_clips | mass_share_sum |",
             "|---|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        lines.append(f"| {int(r['feature_idx'])} | {r['flip_rate']:.2f} | {r['noise_rate']:.2f} | "
                     f"{r['dec_rate']:.2f} | {r['inc_rate']:.2f} | {int(r['n_clips'])} | "
                     f"{r['mass_share_sum']:.2f} |")
    return lines


def top12_composition_section() -> list[str]:
    """What the top-12-by-mass features carry: share of each bucket's mass, and
    their per-frame temporal profile — tests 'flip reservoir' and 'ramps up over
    the clip' readings. B8 cross-referenced as found (infeasible, never ran)."""
    lines = ["## 8. What the top-12 actually carry (mass composition + temporal profile)", ""]
    lines += ["| config | clip R-mass | of noise | of sign_flip | of decrease | of increase |",
              "|---|---|---|---|---|---|"]
    for (ds, L) in [("ssv2", 7), ("k400", 5), ("k400", 7), ("k400", 9)]:
        pq, acc_csv = CFG["mass"][(ds, L)]
        df = pd.read_parquet(pq)
        acc = pd.read_csv(acc_csv)
        df = df[df["class_id"].isin(set(acc.loc[acc["accuracy"] >= 0.40, "class_id"]))]
        r = np.stack(df["signed_vec_R"].to_numpy()).astype(np.float32)
        c = np.stack(df["signed_vec_C"].to_numpy()).astype(np.float32)
        ar, ac = np.abs(r), np.abs(c)
        rec, noise, sf, dec, inc = _clip_buckets(ar, ac, r, c)
        top12 = np.zeros(ar.shape[1], bool)
        top12[_feature_rates(pq, acc_csv).nlargest(12, "mass_share_sum")["feature_idx"]] = True
        tot = (ar * rec).sum(axis=1)
        share_b = lambda m: ((ar * (m & top12)).sum(axis=1) / (ar * m).sum(axis=1).clip(1e-12)).mean()
        lines.append(f"| TF {ds} L{L} | {((ar * (rec & top12)).sum(axis=1) / tot).mean():.3f} | "
                     f"{share_b(noise):.3f} | {share_b(sf):.3f} | {share_b(dec):.3f} | {share_b(inc):.3f} |")
    lines += ["Readings as measured:",
              "- The top-12 carry every bucket's mass at roughly their overall share, and are "
              "UNDER-represented in sign-flip mass on every config. Sign-flip mass lives in "
              "the lighter tail, not the heavy dozen.",
              "- B8 matched-mass control (the mass-vs-reversal arbiter): infeasible, never "
              "executed — never-reverse pool ceiling 0.56% mass vs TOP12's 3.03% "
              "(outputs/interim_results/B8_matched_mass_control_sets.json, sets drawn: 0). "
              "So 'ablation damage = mass removal' is the live hypothesis, untested.",
              "- Temporal profile: SSv2 TOP12 lean mildly end-of-clip (last-half 0.530, "
              "frame7/frame0 = 1.26) but the whole dictionary shares nearly that shape "
              "(0.518, 1.15) — |DFA| profile, legacy parquet. K400 top-12 raw-activation "
              "profile is exactly FLAT across the 8 frames (0.125 each), as is the "
              "dictionary average. No temporal ramp on K400.",
              "- Presence: recruited in essentially every clip (3,577/3,577 SSv2; "
              "~2,000/2,118 K400 L7)."]
    return lines


def top12_selectivity_section() -> list[str]:
    """Class-selectivity of the heavy dozen: per-class mean share of clip R-mass
    per feature, summarised as effective number of classes (exp of Shannon
    entropy over class shares), benchmarked against mid-mass and all features;
    plus each class's joint reliance on the dozen."""
    lines = ["## 9. Class-selectivity of the heavy dozen (TF-K400)", "",
             "| layer | set | effective classes (median of /59) | max-class share (median) |",
             "|---|---|---|---|"]
    for L in (5, 7, 9):
        pq, acc_csv = CFG["mass"][("k400", L)]
        df = pd.read_parquet(pq)
        acc = pd.read_csv(acc_csv)
        df = df[df["class_id"].isin(set(acc.loc[acc["accuracy"] >= 0.40, "class_id"]))]
        r = np.stack(df["signed_vec_R"].to_numpy()).astype(np.float32)
        ar = np.abs(r)
        tot = ar.sum(axis=1)
        classes = df["class_id"].to_numpy()
        uc = np.unique(classes)
        share = ar / tot[:, None]
        m = np.stack([share[classes == c].mean(axis=0) for c in uc])   # (class, feature)
        fr = _feature_rates(pq, acc_csv)
        ranked = fr.sort_values("mass_share_sum", ascending=False)
        sets = [("top-12", ranked.head(12)["feature_idx"].to_numpy()),
                ("mid 100-200", ranked.iloc[100:200]["feature_idx"].to_numpy()),
                ("all 6144", np.arange(ar.shape[1]))]
        for label, ids in sets:
            effs = np.array([np.exp(-(p * np.log(p.clip(1e-12))).sum()) for p in
                             (m[:, f] / m[:, f].sum() for f in ids)])
            maxsh = np.array([m[:, f].max() / m[:, f].sum() for f in ids])
            lines.append(f"| {L} | {label} | {np.median(effs):.1f} (p10 {np.percentile(effs, 10):.1f}) "
                         f"| {np.median(maxsh):.3f} |")
    lines += ["Readings as measured:",
              "- The heavy dozen spread across essentially all eligible classes — effective "
              "classes 52.6-55.7 of 59, at or ABOVE the all-features median (48.4-55.2). "
              "Heaviness co-occurs with slightly MORE uniform class spread, not less. "
              "Max-class share ~0.04 (uniform would be 1/59 = 0.017).",
              "- Every class routes a broadly similar slice of its causal mass through the "
              "dozen: joint per-class share min/median/max = L5 0.104/0.127/0.158, "
              "L7 0.118/0.164/0.220, L9 0.111/0.139/0.187 — mild class lean (L7 spread "
              "0.10, highest class 354 at 0.220) but no class dominated and none free of them.",
              "- Consistent with routing infrastructure: present everywhere, mass-heavy, "
              "not class- or frame-selective (flat temporal profile, section 8)."]
    return lines


def patch_alignment_section() -> list[str]:
    """Within-class spatial alignment of peak activation, PER PATCH (user rule:
    patch-level or nothing — quadrants meaningless). Chance = 1/196; nulls are
    simulated uniform draws at each group's exact n (mode agreement, 2000 trials).
    Cuts: all clips, and top-50% by total_activation (strong firing only)."""
    rng = np.random.default_rng(42)
    scan_root = ROOT / "outputs/analysis/max_activating_tf"
    files = sorted(scan_root.glob("*_l7/feature*/scan.csv"))
    if not files:
        return ["## 10. Within-class per-patch spatial alignment", "",
                "(no scan.csv files synced yet)"]
    lines = ["## 10. Within-class per-patch spatial alignment (peak_patch mode agreement)", "",
             "| dataset | feature | classes n>=10 | median agr | max agr | above null-p95 (expect ~5%) |",
             "|---|---|---|---|---|---|"]

    def nulls(n):
        counts = np.array([np.bincount(d, minlength=196).max()
                           for d in rng.integers(0, 196, size=(2000, n))])
        return counts.mean() / n, np.percentile(counts, 95) / n

    def agree(sub):
        rows = []
        for _, g in sub.groupby("class_id"):
            if len(g) < 10:
                continue
            rows.append((len(g), g["peak_patch"].value_counts().iloc[0] / len(g)))
        return rows

    for f in files:
        ds = f.parent.parent.name.split("_")[0]
        feat = f.parent.name.replace("feature", "")
        df = pd.read_csv(f)
        for cut, sub in [("all", df), ("top50", df[df.total_activation >= df.total_activation.median()])]:
            rows = agree(sub)
            if not rows:
                continue
            a = np.array([x for _, x in rows])
            null_p95_by_n = {n: nulls(n)[1] for n in np.unique([r[0] for r in rows])}
            above = sum(1 for n, x in rows if x > null_p95_by_n[n])
            lines.append(f"| {ds} | {feat} ({cut}) | {len(rows)} | {np.median(a):.3f} | "
                         f"{a.max():.3f} | {above} |")
    lines += ["Nulls account for group size (small-n groups agree more by chance). "
              "Readings are numbers only; the dozen-wide table completes once the "
              "render job's scan.csv files are all synced."]
    return lines


def main() -> None:
    CFG["out_md"].parent.mkdir(parents=True, exist_ok=True)
    header = ["# TF K400 vs TF SSv2 — collated comparison (03/10/26)", "",
              "Sources: TF-K400 run (chain complete, verification 3/3/3/3) vs the TF-SSv2 "
              "artifacts already in-repo. All numbers as found.", ""]
    sections = [accuracy_section(), bucket_section(), mass_bucket_section(),
                dec_inc_section(), sign_flip_focus_section(), signflip_candidates_section(),
                top12_composition_section(), top12_selectivity_section(),
                patch_alignment_section(), pos_lock_section(), top_locked_listing()]
    CFG["out_md"].write_text("\n".join(header + [ln for sec in sections for ln in sec]) + "\n")
    print(f"Written -> {CFG['out_md']}")


if __name__ == "__main__":
    main()
