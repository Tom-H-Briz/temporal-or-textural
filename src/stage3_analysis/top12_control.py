"""
TOP12 matched-mass control (CC brief 25/08/26, "Matched-Mass Control for TF
TOP12 Ablation") — tests whether TOP12's ablation damage follows from DFA
mass or from reversal (sign-flip) behaviour specifically. Draws mass-matched
control sets from TF-L7's 135 never-reverse features and ablates them under
the same protocol as B5 (run_ablation_tf.py's TOP12 run).

Two stages, run separately given the compute cost of stage 2 — each
additional ablation target is one FULL model forward pass per clip per
condition in DFAEngine.run_ablated (no cheap decoder-only replay exists),
so 200 targets is ~29x run_ablation_tf.py's existing 7-target job:
  draw   (default) — brief steps 1-3: report TOP12's mass, build the
           never-reverse pool, draw up to 200 matched sets, or report the
           shortfall if none exist. Fast, no model/video needed.
  ablate (--ablate) — brief steps 4-5: ablate TOP12 + every drawn set,
           aggregate. Heavy — intended for Isambard, not local.

Outputs:
    outputs/interim_results/B8_matched_mass_control_sets.json  (draw stage —
        matched sets + feasibility report; --ablate reads this back rather
        than redrawing, so the two stages can run on different machines)
    outputs/interim_results/B8_matched_mass_control.csv         (--ablate;
        per target/condition/stratum, same column convention as B5)
    outputs/interim_results/B8_matched_mass_control_distribution.csv
        (--ablate; brief step 5's distribution-across-sets + TOP12 percentile
         — a second file since it's a different shape to the long table above;
         the brief's Output section names only the one CSV, flagging this as
         an addition rather than silently folding it in)

Usage:
    uv run python src/stage3_analysis/top12_control.py             # draw only
    uv run python src/stage3_analysis/top12_control.py --ablate --dry-run
    uv run python src/stage3_analysis/top12_control.py --ablate
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.dfa_engine import DFAEngine, _preprocess_clip
from run_ablation_tf import CFG as TF_CFG, load_clips, preprocess_c
from ToT_utils import resolve_sae_checkpoint

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Same list as run_ablation_tf.py's TARGETS["TOP12"] — duplicated literally,
# not re-derived, so this control is matched against the exact set B5 ablated.
TOP12 = [3029, 1517, 2090, 2057, 2156, 1588, 4590, 3813, 6029, 622, 1371, 4134]

CFG = {
    "mass_delta_path":  ROOT / "outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet",
    "full_detail_path": ROOT / "outputs/analysis/shuffle_reduction_composition/ssv2_tf_full_detail.parquet",
    "out_dir":          ROOT / "outputs/interim_results",
    "sets_path":        ROOT / "outputs/interim_results/B8_matched_mass_control_sets.json",
    "out_path":         ROOT / "outputs/interim_results/B8_matched_mass_control.csv",
    "dist_path":        ROOT / "outputs/interim_results/B8_matched_mass_control_distribution.csv",
    "tolerance":     0.10,
    "draw_size":     12,
    "n_target_sets": 200,
    "seed":          42,
    "max_attempts":  200_000,
}


def combined_mass_pct(mat: np.ndarray, total_abs: np.ndarray, features: list[int]) -> float:
    """Self-consistent scaffold-mass metric — same formula as
    scaffold_selection_consolidated.py's _combined_mass_pct_r: per-clip
    Sigma|signed_vec_R[features]| / Sigma|signed_vec_R[all]|, mean across clips."""
    subset_abs = np.abs(mat[:, features]).sum(axis=1)
    return float((subset_abs / total_abs).mean())


def per_feature_contribution(mat: np.ndarray, total_abs: np.ndarray) -> np.ndarray:
    """Per-feature mean contribution to combined_mass_pct. Additive: mean and
    sum-over-a-fixed-feature-set commute, so combined_mass_pct(S) == sum of
    this vector over S — lets set-matching work off one (6144,) vector
    instead of recomputing the ratio for every candidate draw."""
    return (np.abs(mat) / total_abs[:, None]).mean(axis=0)


def never_reverse_pool(full_detail_path: Path) -> tuple[list[int], int]:
    df = pd.read_parquet(full_detail_path)
    flip_frac = df.groupby("feature_id")["bucket"].apply(lambda s: (s == "sign_flip").mean())
    pool = sorted(int(f) for f in flip_frac[flip_frac == 0.0].index)
    return pool, len(flip_frac)


def draw_matched_sets(contrib: np.ndarray, eligible: np.ndarray, target_mass: float, cfg: dict) -> dict:
    """Rejection-samples draw_size-feature sets from `eligible` whose summed
    contribution falls within +/-tolerance of target_mass. Feasibility is
    checked first: if even the highest-mass draw_size features in the pool
    can't reach the band, no sampling budget will find one either (brief step
    3 — stop and report, don't tune the tolerance to force a match)."""
    lo, hi = target_mass * (1 - cfg["tolerance"]), target_mass * (1 + cfg["tolerance"])
    ceiling = float(np.sort(contrib[eligible])[::-1][:cfg["draw_size"]].sum())
    if ceiling < lo:
        return {"feasible": False, "ceiling_mass": ceiling, "target_lo": lo, "target_hi": hi, "sets": []}

    rng = np.random.default_rng(cfg["seed"])
    found: dict[frozenset, float] = {}
    attempts = 0
    while len(found) < cfg["n_target_sets"] and attempts < cfg["max_attempts"]:
        attempts += 1
        draw = rng.choice(eligible, size=cfg["draw_size"], replace=False)
        mass = float(contrib[draw].sum())
        if lo <= mass <= hi:
            found[frozenset(int(f) for f in draw)] = mass
    sets = [{"features": sorted(k), "mass_pct": v} for k, v in found.items()]
    return {"feasible": True, "ceiling_mass": ceiling, "target_lo": lo, "target_hi": hi,
            "attempts": attempts, "sets": sets}


def run_draw_stage() -> dict:
    mass_df = pd.read_parquet(CFG["mass_delta_path"])
    mat = np.stack(mass_df["signed_vec_R"].to_numpy()).astype(np.float32)
    total_abs = mass_df["total_abs_R"].to_numpy()

    top12_mass_frac = combined_mass_pct(mat, total_abs, TOP12)
    contrib = per_feature_contribution(mat, total_abs)
    assert abs(contrib[TOP12].sum() - top12_mass_frac) < 1e-4, "additivity check failed"
    log.info(f"TOP12 aggregate DFA mass (mean per-clip, R, L7, n_clips={len(mass_df)}): "
             f"{top12_mass_frac * 100:.2f}%  (drafts cited 3.03%, no artifact found for that figure)")

    pool, n_total = never_reverse_pool(CFG["full_detail_path"])
    log.info(f"Never-reverse pool: {len(pool)} of {n_total} ever-top10 L7 features "
             f"(brief expected 135 of 183)")

    result = draw_matched_sets(contrib, np.array(pool), top12_mass_frac, CFG)
    if not result["feasible"]:
        log.warning(f"INFEASIBLE — best {CFG['draw_size']}-feature draw from the never-reverse "
                    f"pool reaches {result['ceiling_mass'] * 100:.2f}%, target band is "
                    f"[{result['target_lo'] * 100:.2f}%, {result['target_hi'] * 100:.2f}%]. "
                    f"Mass and reversal are not separable in this dictionary.")
    else:
        log.info(f"Drew {len(result['sets'])} matched sets in {result['attempts']:,} attempts "
                 f"(requested {CFG['n_target_sets']})")

    payload = {"top12_mass_pct": top12_mass_frac * 100, "top12_features": TOP12,
               "never_reverse_pool_size": len(pool), "ever_top10_total": n_total, **result}
    CFG["sets_path"].parent.mkdir(parents=True, exist_ok=True)
    CFG["sets_path"].write_text(json.dumps(payload, indent=2))
    log.info(f"  -> {CFG['sets_path']}")
    return payload


def run_clip(engine: DFAEngine, clip_id: str, class_id: int, sl_label: str, clip_path: Path,
             device: str, targets: dict) -> list[dict]:
    """Same protocol as run_ablation_tf.py's run_clip (R-eligibility gate,
    then R + C for every target), parameterized on `targets` instead of that
    module's fixed TARGETS global — TOP12 plus every matched control set."""
    pv_r = _preprocess_clip(clip_path, engine._num_frames, engine._processor, device)
    z_r = engine.get_z_pixels(pv_r)
    base_logit_r, _, correct_r, _ = engine.run_ablated(pv_r, class_id, [], z_r)
    if not correct_r:
        return []

    pv_c = preprocess_c(clip_path, clip_id, engine._num_frames, engine._processor, device)
    rows = []
    for cond, pv, z_cache, base_logit in [("R", pv_r, z_r, base_logit_r),
                                           ("C", pv_c, engine.get_z_pixels(pv_c), None)]:
        if base_logit is None:
            base_logit, _, _, _ = engine.run_ablated(pv, class_id, [], z_cache)
        for target_name, indices in targets.items():
            abl_logit, pred, correct, _ = engine.run_ablated(pv, class_id, indices, z_cache)
            rows.append({
                "clip_id": clip_id, "class_id": class_id, "sl_label": sl_label,
                "perturbation_condition": cond, "ablation_target": target_name,
                "baseline_logit": base_logit, "ablated_logit": abl_logit,
                "delta": base_logit - abl_logit,
                "predicted_class_ablated": pred, "correct_ablated": correct,
            })
    return rows


def aggregate_target_condition_stratum(df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    """Same column convention as B5_top12_ablation.csv."""
    mtime = pd.Timestamp(source_file.stat().st_mtime, unit="s").date().isoformat()
    rows = []
    for (target, cond), sub in df.groupby(["ablation_target", "perturbation_condition"]):
        strata = [("static", sub[sub.sl_label == "static"]),
                  ("temporal", sub[sub.sl_label == "temporal"]),
                  ("overall", sub)]
        for stratum, ssub in strata:
            n = len(ssub)
            rows.append({
                "ablation_target": target, "condition": cond, "stratum": stratum, "n_clips": n,
                "flip_rate": 1 - ssub["correct_ablated"].mean() if n else np.nan,
                "mean_logit_damage": ssub["delta"].mean() if n else np.nan,
                "median_logit_damage": ssub["delta"].median() if n else np.nan,
                "mean_baseline_logit": ssub["baseline_logit"].mean() if n else np.nan,
                "median_baseline_logit": ssub["baseline_logit"].median() if n else np.nan,
                "pool": "SL-35", "unit": "clip", "weighting": "n/a",
                "source_file": str(source_file), "source_mtime": mtime,
                "status": "OK" if n else "MISSING",
            })
    return pd.DataFrame(rows)


def distribution_summary(agg: pd.DataFrame) -> pd.DataFrame:
    """Brief step 5: distribution across matched sets + TOP12's percentile
    within it, overall stratum, per condition. Lower flip_rate/logit_damage
    is less damage, so percentile = share of matched sets TOP12 exceeds."""
    matched = agg[agg["ablation_target"].str.startswith("matched_") & (agg["stratum"] == "overall")]
    top12 = agg[(agg["ablation_target"] == "TOP12") & (agg["stratum"] == "overall")]
    rows = []
    for cond, sub in matched.groupby("condition"):
        t = top12[top12["condition"] == cond].iloc[0]
        for metric in ["flip_rate", "mean_logit_damage", "median_logit_damage"]:
            rows.append({
                "condition": cond, "metric": metric, "n_matched_sets": len(sub),
                "matched_median": sub[metric].median(), "matched_mean": sub[metric].mean(),
                "matched_min": sub[metric].min(), "matched_max": sub[metric].max(),
                "top12_value": t[metric],
                "top12_percentile": float((sub[metric] < t[metric]).mean() * 100),
            })
    return pd.DataFrame(rows)


def run_ablate_stage(payload: dict, dry_run: bool) -> None:
    if not payload.get("feasible", False):
        log.warning("Draw stage reported infeasible — nothing to ablate.")
        return

    targets = {"TOP12": payload["top12_features"]}
    for i, s in enumerate(payload["sets"]):
        targets[f"matched_{i:03d}"] = s["features"]
    log.info(f"Ablating {len(targets)} targets (TOP12 + {len(targets) - 1} matched sets)")

    resolved = resolve_sae_checkpoint(TF_CFG["model_flag"], TF_CFG["layer"],
                                      dataset_name="ssv2", sae_k=TF_CFG["sae_k"])
    cfg = {**TF_CFG, **resolved}
    clips = load_clips(cfg)
    if dry_run:
        clips = clips[:10]
        log.info("DRY RUN — 10 clips only")

    all_rows = []
    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"], layer=cfg["layer"],
                   device=cfg["device"], sae_k=cfg["sae_k"], dataset_name="ssv2") as engine:
        for i, (clip_id, class_id, sl_label, clip_path) in enumerate(clips):
            try:
                all_rows.extend(run_clip(engine, clip_id, class_id, sl_label, clip_path,
                                          cfg["device"], targets))
            except Exception as exc:
                log.warning(f"SKIP {clip_id}: {exc}")
            log.info(f"[{i + 1}/{len(clips)}] clip {clip_id}  rows: {len(all_rows):,}")

    df = pd.DataFrame(all_rows)
    tag = "dry_run_" if dry_run else ""
    long_path = CFG["out_dir"] / f"{tag}top12_control_results_long.parquet"
    df.to_parquet(long_path, index=False)
    log.info(f"  {len(df):,} rows -> {long_path}")

    agg = aggregate_target_condition_stratum(df, long_path)
    agg.to_csv(CFG["out_path"], index=False)
    log.info(f"  -> {CFG['out_path']}")

    dist = distribution_summary(agg)
    dist.to_csv(CFG["dist_path"], index=False)
    print(dist.to_string(index=False))
    log.info(f"  -> {CFG['dist_path']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablate", action="store_true", help="run the heavy ablation stage (steps 4-5)")
    parser.add_argument("--dry-run", action="store_true", help="with --ablate, 10 clips only")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_draw_stage()
    if args.ablate:
        run_ablate_stage(payload, args.dry_run)
    else:
        log.info("Draw stage only — rerun with --ablate to run the heavy ablation "
                 "(recommended on Isambard: each matched set costs one more full "
                 "forward pass per clip per condition, ~29x run_ablation_tf.py's cost).")


if __name__ == "__main__":
    main()
