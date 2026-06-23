"""
Pair contrast: class 36 vs class 37 — order-sensitive, class-discriminating SAE features.

Class 36: "Moving something and something away from each other"
Class 37: "Moving something and something closer to each other"

delta(f)      = mean_s_R(f) − mean_s_C(f)   [per class, on class-level means]
cross_delta(f) = delta_36(f) − delta_37(f)   [asymmetry of causal-mass shift; ranking signal]

Outputs: outputs/stage3_analysis/pair_contrast_tf/pair_contrast_tf_<cids>.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(ROOT / "notebooks"))

from ToT_utils import MODEL_REGISTRY, _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

_LAYER = 7

CFG = {
    "model_flag":      "timesformer",
    "layer":           _LAYER,
    "class_ids":       [36, 37],
    "labels_path":     str(ROOT / "data/ssv2/labels/labels.json"),
    "validation_path": str(ROOT / "data/ssv2/labels/validation.json"),
    "video_dir":       str(ROOT / "data/ssv2/20bn-something-something-v2"),
    "output_dir":      str(ROOT / "outputs/stage3_analysis/pair_contrast_tf"),
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
}


def _resolve_cfg(cfg: dict) -> dict:
    sae_dir   = ROOT / "outputs" / "sae"
    layer     = cfg["layer"]
    matches   = list(sae_dir.glob(f"sae_tf_k*_x*_l{layer}_job{layer}_best.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected 1 TF checkpoint for layer {layer}, found: {matches}")
    ckpt_path = matches[0]
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae_k     = ckpt.get("sae_k")
    if sae_k is None:
        raise ValueError(f"sae_k not in checkpoint {ckpt_path}")
    dim_mean  = sae_dir / f"tf_layer{layer}_dim_mean.pt"
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {"sae_path": str(ckpt_path), "dim_mean_path": str(dim_mean), "sae_k": sae_k}


def load_clips(cfg: dict) -> dict[int, list[tuple[str, Path, Path]]]:
    label_map, clips, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir   = Path(cfg["video_dir"])
    perturb_dir = ROOT / "data" / "perturbed"
    target      = set(cfg["class_ids"])
    result: dict[int, list[tuple[str, Path, Path]]] = {c: [] for c in target}
    for c in clips:
        cid = label_map.get(_strip_brackets(c["template"]))
        if cid not in target:
            continue
        path_r = video_dir   / f"{c['id']}.webm"
        path_c = perturb_dir / "C" / f"{c['id']}C.webm"
        if path_r.exists() and path_c.exists():
            result[cid].append((str(c["id"]), path_r, path_c))
    for cid, lst in result.items():
        print(f"  Class {cid}: {len(lst)} clips on disk")
    return result


def collect_scores(engine: DFAEngine, clips: list[tuple[str, Path, Path]],
                   class_id: int) -> list[dict]:
    records = []
    for clip_id, path_r, path_c in clips:
        r_result = engine.run(path_r, class_id)
        if not r_result.correct:
            continue
        c_result = engine.run(path_c, class_id)
        records.append({"clip_id": clip_id, "class_id": class_id,
                        "dfa": r_result.per_feature_summary.numpy(),
                        "s_R": r_result.signed_feature_summary.numpy(),
                        "s_C": c_result.signed_feature_summary.numpy()})
    print(f"  Class {class_id}: {len(records)} R-correct clips used")
    return records


def aggregate(records_36: list[dict], records_37: list[dict]) -> pd.DataFrame:
    ms_dfa_36 = np.stack([r["dfa"] for r in records_36]).mean(axis=0)
    ms_dfa_37 = np.stack([r["dfa"] for r in records_37]).mean(axis=0)
    ms_r_36   = np.stack([r["s_R"] for r in records_36]).mean(axis=0)
    ms_c_36   = np.stack([r["s_C"] for r in records_36]).mean(axis=0)
    ms_r_37   = np.stack([r["s_R"] for r in records_37]).mean(axis=0)
    ms_c_37   = np.stack([r["s_C"] for r in records_37]).mean(axis=0)

    delta_36    = ms_r_36 - ms_c_36
    delta_37    = ms_r_37 - ms_c_37
    cross_delta = delta_36 - delta_37

    return pd.DataFrame({
        "feature_idx":  np.arange(len(ms_dfa_36)),
        "mean_dfa_36":  ms_dfa_36,
        "mean_dfa_37":  ms_dfa_37,
        "max_dfa":      np.maximum(ms_dfa_36, ms_dfa_37),
        "mean_s_R_36":  ms_r_36,
        "mean_s_R_37":  ms_r_37,
        "mean_s_C_36":  ms_c_36,
        "mean_s_C_37":  ms_c_37,
        "delta_36":     delta_36,
        "delta_37":     delta_37,
        "cross_delta":  cross_delta,
    })


def save_outputs(records_36, records_37, df: pd.DataFrame, cfg: dict) -> None:
    out_dir   = Path(cfg["output_dir"])
    class_str = "_".join(str(c) for c in sorted(cfg["class_ids"]))
    csv_name  = f"pair_contrast_tf_{class_str}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = df.sort_values("max_dfa", ascending=False).reset_index(drop=True)
    cols = ["feature_idx", "mean_dfa_36", "mean_dfa_37", "max_dfa",
            "mean_s_R_36", "mean_s_R_37", "mean_s_C_36", "mean_s_C_37",
            "delta_36", "delta_37", "cross_delta"]
    out      = ranked[cols].copy()
    num_cols = [c for c in cols if c != "feature_idx"]
    total_row = {c: out[c].sum() for c in num_cols}
    total_row["feature_idx"] = "total"
    out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)
    out.to_csv(out_dir / csv_name, index=False)

    with open(out_dir / "run_summary.txt", "w") as f:
        f.write(f"class 36: {len(records_36)} R-correct clips used\n")
        f.write(f"class 37: {len(records_37)} R-correct clips used\n")
        f.write(f"features total: {len(df)}\n")
        f.write(f"max_dfa range: [{ranked['max_dfa'].min():.4f}, {ranked['max_dfa'].max():.4f}]\n")
        top10 = ranked.head(10)[["feature_idx", "max_dfa", "cross_delta"]].values.tolist()
        f.write(f"top-10 by max_dfa: {top10}\n")
    print(f"  Outputs → {out_dir / csv_name}")


def main() -> None:
    cfg = {**CFG, **_resolve_cfg(CFG)}
    print(f"Device: {cfg['device']}  Layer: {cfg['layer']}")
    print(f"SAE: {Path(cfg['sae_path']).name}  sae_k={cfg['sae_k']}")

    clips_by_class = load_clips(cfg)

    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"],
                   sae_k=cfg["sae_k"]) as engine:
        print("Collecting class 36 (apart)...")
        records_36 = collect_scores(engine, clips_by_class[36], 36)
        print("Collecting class 37 (closer)...")
        records_37 = collect_scores(engine, clips_by_class[37], 37)

    print("Aggregating and contrasting...")
    df = aggregate(records_36, records_37)
    save_outputs(records_36, records_37, df, cfg)
    print("Done.")


if __name__ == "__main__":
    main()
