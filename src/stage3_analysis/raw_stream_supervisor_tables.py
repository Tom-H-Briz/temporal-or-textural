"""
Supervisor summary tables for the final raw-stream ablation run (24/09).

Writes, to outputs/analysis/raw_stream_ablation/supervisor_tables/ (no p-values):
    00_notes.csv               — definitions and population, one row each
    01_overall_summary.csv     — per intervention x target: logit damage, accuracy, flip rate
    02_top10_classes.csv       — top-10 most damaged classes per intervention, with SL label
    03_flip_rates.csv          — prediction flip rate: overall / temporal / static
    04_temporal_vs_static.csv  — logit damage by SL label, scaffold vs control

Usage:
    uv run python src/stage3_analysis/raw_stream_supervisor_tables.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

CFG = {
    "in_path": ROOT / "outputs/analysis/raw_stream_ablation/raw_stream_ablation_l5_n4450.parquet",
    "labels_path": ROOT / "data/ssv2/labels/labels.json",
    "sl_path": ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv",
    "out_dir": ROOT / "outputs/analysis/raw_stream_ablation/supervisor_tables",
    "min_class_clips": 30,
    "top_n": 10,
}
ARMS = {
    "recon_minus": "SAE ablation",
    "raw_minus": "Subtraction (raw)",
    "raw_meanproject": "Mean projection (raw)",
}
TARGETS = {
    "all7": "Scaffold (7 features)",
    "rand_dict7": "Control: 7 random SAE features (mass-matched)",
    "rand_iso7": "Control: 7 random directions",
}


def load(cfg: dict) -> pd.DataFrame:
    """One row per (clip, arm, target). base_* are the arm's own un-ablated pass
    (raw arms vs the untouched model, SAE arm vs the unablated splice);
    in_correct_set marks clips the unmodified model classifies correctly."""
    df = pd.read_parquet(cfg["in_path"])
    base = df[df.ablation_target == "none"].pivot(
        index="clip_id", columns="mode", values=["correct", "correct_class_logit"])
    x = df[df["mode"].isin(ARMS) & df.ablation_target.isin(TARGETS)].copy()
    is_raw = x["mode"].str.startswith("raw")
    for col in ("correct", "correct_class_logit"):
        x[f"base_{col}"] = np.where(is_raw, x.clip_id.map(base[(col, "raw")]),
                                    x.clip_id.map(base[(col, "recon")]))
    x["base_correct"] = x.base_correct.astype(bool)
    x["correct"] = x.correct.astype(bool)
    x["in_correct_set"] = x.clip_id.map(base[("correct", "raw")]).astype(bool)
    x["logit_damage"] = x.base_correct_class_logit.astype(float) - x.correct_class_logit.astype(float)
    labels = {int(v): k for k, v in json.load(open(cfg["labels_path"])).items()}
    sl = pd.read_csv(cfg["sl_path"]).set_index("class_id")["category"]
    x["class_name"] = x.class_id.map(labels)
    x["temporal_or_static"] = x.class_id.map(sl)
    x["intervention"] = x["mode"].map(ARMS)
    x["target"] = x.ablation_target.map(TARGETS)
    return x


def overall_summary(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, tgt), g in x.groupby(["mode", "ablation_target"]):
        c = g[g.in_correct_set]
        se = c.logit_damage.std() / np.sqrt(len(c))
        rows.append({
            "intervention": ARMS[arm], "target": TARGETS[tgt],
            "n_clips_all": len(g),
            "accuracy_before_all_clips_%": g.base_correct.mean() * 100,
            "accuracy_after_all_clips_%": g.correct.mean() * 100,
            "accuracy_drop_pp": (g.base_correct.mean() - g.correct.mean()) * 100,
            "n_clips_correct_set": len(c),
            "mean_logit_damage": c.logit_damage.mean(),
            "logit_damage_95ci_low": c.logit_damage.mean() - 1.96 * se,
            "logit_damage_95ci_high": c.logit_damage.mean() + 1.96 * se,
            "prediction_flip_rate_%": (1 - c.correct.mean()) * 100,
        })
    out = pd.DataFrame(rows)
    out["_a"], out["_t"] = out.intervention.map(list(ARMS.values()).index), out.target.map(list(TARGETS.values()).index)
    return out.sort_values(["_a", "_t"]).drop(columns=["_a", "_t"])


def class_table(x: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per (arm, class) for the scaffold, plus the same-arm control's damage."""
    rows = []
    for (arm, cls), g in x[x.ablation_target == "all7"].groupby(["mode", "class_id"]):
        c = g[g.in_correct_set]
        if len(c) < cfg["min_class_clips"]:
            continue
        ctl = x[(x["mode"] == arm) & (x.class_id == cls) & (x.ablation_target == "rand_dict7") & x.in_correct_set]
        rows.append({
            "mode": arm, "intervention": ARMS[arm], "class_id": cls,
            "class_name": g.class_name.iloc[0], "temporal_or_static": g.temporal_or_static.iloc[0],
            "n_clips_correct_set": len(c),
            "mean_logit_damage": c.logit_damage.mean(),
            "control_mean_logit_damage": ctl.logit_damage.mean(),
            "prediction_flip_rate_%": (1 - c.correct.mean()) * 100,
            "accuracy_before_all_clips_%": g.base_correct.mean() * 100,
            "accuracy_after_all_clips_%": g.correct.mean() * 100,
        })
    t = pd.DataFrame(rows)
    t["rank"] = t.groupby("mode").mean_logit_damage.rank(ascending=False, method="first").astype(int)
    return t


def top10_classes(t: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Top-N per arm, with each class's rank under every arm so alignment is visible."""
    ranks = t.pivot(index="class_id", columns="mode", values="rank")
    top = t[t["rank"] <= cfg["top_n"]].copy()
    for arm, name in ARMS.items():
        top[f"rank_under_{name}"] = top.class_id.map(ranks[arm])
    order = list(ARMS)
    top["_a"] = top["mode"].map(order.index)
    return top.sort_values(["_a", "rank"]).drop(columns=["_a", "mode"])


def flip_rates(x: pd.DataFrame) -> pd.DataFrame:
    c = x[x.in_correct_set]
    rows = []
    for (arm, tgt), g in c.groupby(["mode", "ablation_target"]):
        flip = lambda s: (1 - s.correct.mean()) * 100
        rows.append({
            "intervention": ARMS[arm], "target": TARGETS[tgt],
            "flip_rate_all_%": flip(g),
            "flip_rate_temporal_%": flip(g[g.temporal_or_static == "temporal"]),
            "flip_rate_static_%": flip(g[g.temporal_or_static == "static"]),
        })
    out = pd.DataFrame(rows)
    out["_a"], out["_t"] = out.intervention.map(list(ARMS.values()).index), out.target.map(list(TARGETS.values()).index)
    return out.sort_values(["_a", "_t"]).drop(columns=["_a", "_t"])


def temporal_vs_static(x: pd.DataFrame) -> pd.DataFrame:
    """Gap = temporal minus static mean logit damage. Net gap pairs each clip's
    scaffold damage with its own control damage before splitting by label."""
    c = x[x.in_correct_set]
    rows = []
    for arm in ARMS:
        a = c[c["mode"] == arm].pivot(index="clip_id", columns="ablation_target", values="logit_damage")
        lab = (c[c["mode"] == arm].drop_duplicates("clip_id").set_index("clip_id")
               .temporal_or_static.reindex(a.index))
        T, S = a[lab == "temporal"], a[lab == "static"]
        net = a.all7 - a.rand_dict7
        rows.append({
            "intervention": ARMS[arm],
            "n_temporal_clips": len(T), "n_static_clips": len(S),
            "scaffold_damage_temporal": T.all7.mean(), "scaffold_damage_static": S.all7.mean(),
            "scaffold_gap_T_minus_S": T.all7.mean() - S.all7.mean(),
            "control_damage_temporal": T.rand_dict7.mean(), "control_damage_static": S.rand_dict7.mean(),
            "control_gap_T_minus_S": T.rand_dict7.mean() - S.rand_dict7.mean(),
            "scaffold_gap_net_of_control": net[lab == "temporal"].mean() - net[lab == "static"].mean(),
        })
    return pd.DataFrame(rows)


def notes(x: pd.DataFrame) -> pd.DataFrame:
    n_all, n_ok = x.clip_id.nunique(), x[x.in_correct_set].clip_id.nunique()
    return pd.DataFrame({"item": [
        "model / layer", "population", "correct set", "interventions", "scaffold", "controls",
        "logit damage", "prediction flip rate", "accuracy columns", "class tables",
    ], "definition": [
        "VideoMAE finetuned on SSv2, layer 5, SAE k64 x8 (job7ep)",
        f"{n_all} SSv2 clips from the 32 SL classes (the existing SAE-ablation population)",
        f"{n_ok} clips classified correctly by the unmodified model; logit damage and flip rates use these",
        "SAE ablation = zero the 7 latents in the SAE reconstruction; Subtraction = remove the 7 features' "
        "contribution from the model's own activations; Mean projection = set the activations along the 7 "
        "feature directions to their dataset mean",
        "7 position-locked SAE features (358, 449, 917, 2093, 3516, 3938, 5004)",
        "7 random SAE features matched per clip to the scaffold's activation mass (within 10%); "
        "7 random directions (mean projection only)",
        "drop in the correct-class logit versus the same model without the intervention",
        "% of correctly classified clips whose prediction changes to a wrong class",
        "accuracy over all clips in the population, before and after the intervention",
        f"classes with at least {CFG['min_class_clips']} clips in the correct set; "
        "rank 1 = largest mean logit damage",
    ]})


def main() -> None:
    cfg = CFG
    out = cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    x = load(cfg)
    tables = {
        "00_notes": notes(x),
        "01_overall_summary": overall_summary(x),
        "02_top10_classes": top10_classes(class_table(x, cfg), cfg),
        "03_flip_rates": flip_rates(x),
        "04_temporal_vs_static": temporal_vs_static(x),
    }
    for name, t in tables.items():
        t.round(3).to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(t)} rows")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
