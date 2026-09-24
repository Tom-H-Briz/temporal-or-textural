"""
Raw-stream ablation of individual L5 SAE features — the reconstruction-free check.

An SAE feature's contribution to the residual stream is exactly rank-1:
    delta_i = z[:, i] outer d_i        (d_i = dictionary row i, see sae/dictionary.py)
so subtracting it from the model's OWN activations needs no reconstruction at
all. Because h_raw = x_hat + eps, `h_raw - delta_i` is identical to ablating
feature i from the reconstruction while adding the SAE error back — the
error-term-preserving ablation. This removes SAE reconstruction fidelity from
the argument entirely: the SAE only supplies the direction to cut.

Conditions per clip, all at L5, same cached z:
    raw               — untouched forward pass, no hook intervention
    raw_minus         — h_raw - delta_i, surgical removal from the real stream
    raw_meanproject   — h_raw with the span(d_i) component set to the dataset mean
    recon             — full SAE reconstruction spliced in (the existing baseline)
    recon_minus       — reconstruction with z[:, i] zeroed (what run_ablation.py did)
    recon_meanproject — reconstruction with the span(d_i) component set to the mean

minus vs meanproject brackets the contribution: because the dictionary is
non-orthogonal, ablating a feature leaves mass along its own direction that
overlapping atoms re-supply (measured: up to 43% for feature 3516). Subtraction
therefore under-removes and mean projection removes all variation along the
directions, so the feature's causal contribution lies between them — in
superposition it is an interval, not a point. Zero projection was dropped
(24/09): no precedent, and it adds an off-distribution shift the controls
depend on.

raw vs raw_minus is the skeptic-facing comparison (no reconstruction anywhere).
recon vs recon_minus bridges back to ablation_summary_l5_job7ep_k64.csv.

Keeps h_raw and the sparse z per clip so any later question about the
fingerprint is answerable without re-running the forward passes.

Usage:
    uv run python src/stage3_analysis/raw_stream_feature_ablation.py
    uv run python src/stage3_analysis/raw_stream_feature_ablation.py --n-clips 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from stage3_analysis.ablation_targets import get_targets
from stage3_analysis.dfa_engine import _preprocess_clip
from stage3_analysis.l5_ablation_l7_feature_impact import load_sae
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY

CFG = {
    "model_flag": "videomae",
    "dataset": "ssv2",
    "layer": 5,
    "sae_k": 64,
    "n_clips": 100,
    "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "video_dir": ROOT / "data/ssv2/20bn-something-something-v2",
    "source_parquet": ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_l5_job7ep_k64.parquet",
    "out_dir": ROOT / "outputs/analysis/raw_stream_ablation",
    # rand_dict7 control: per clip, 7 non-scaffold features drawn from the clip's
    # top-N by activation mass, rejection-sampled to match the scaffold's total
    # mass (method of select_control_features.py). Top-50 matched 149/150 clips
    # within 10% in a pilot; top-200 only 108/150.
    "control_pool_n": 50,
    "control_mass_tol": 0.10,
    "control_max_tries": 2000,
}


def sample_clips(cfg: dict, n_clips: int) -> list[tuple[str, int, Path]]:
    """Random sample from the same R-correct population run_ablation.py used, so
    every number here is directly comparable to ablation_summary_l5_job7ep_k64.csv
    rather than to a differently-filtered pool."""
    df = pd.read_parquet(cfg["source_parquet"], columns=["clip_id", "class_id"]).drop_duplicates("clip_id")
    video_dir = Path(cfg["video_dir"])
    present = [(str(r["clip_id"]), int(r["class_id"]), video_dir / f"{r['clip_id']}.webm")
               for _, r in df.iterrows()
               if (video_dir / f"{r['clip_id']}.webm").exists()]
    if n_clips <= 0:                       # full population, no sampling
        print(f"  {len(present):,} R-correct clips on disk -> using all")
        return present
    rng = np.random.default_rng(cfg["seed"])
    picked = rng.choice(len(present), size=min(n_clips, len(present)), replace=False)
    print(f"  {len(present):,} R-correct clips on disk -> sampling {len(picked)}")
    return [present[i] for i in sorted(picked)]


def make_l5_hook(sae, dim_mean, dictionary, cls_offset: int, state: dict, capture: dict):
    """One hook, four modes, read from `state` each call so a single registration
    serves every condition (pattern from l5_ablation_l7_feature_impact.make_l5_hook).
    Captures h_raw and z every pass — they are the minimal sufficient statistic for
    reconstructing any feature's mark offline."""
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        cls, patches = hidden[:, :cls_offset], hidden[:, cls_offset:]
        B, T, D = patches.shape
        flat = patches.reshape(B * T, D).float()
        with torch.no_grad():
            _, z = sae.encode(flat - dim_mean)
        capture["h_raw"], capture["z"] = flat.detach(), z.detach()

        mode, idx = state["mode"], state["indices"]
        if mode == "raw":
            return None                                    # untouched forward pass
        if mode == "raw_minus":
            # sum of the targets' rank-1 contributions; one index or seven, same form
            new = flat - z[:, idx] @ dictionary[idx]
        elif mode == "recon":
            new = sae.decode(z) + dim_mean
        elif mode == "recon_minus":
            z_abl = z.clone()
            z_abl[:, idx] = 0.0
            new = sae.decode(z_abl) + dim_mean
        elif mode in ("raw_meanproject", "recon_meanproject"):
            # Mean projection (Dobrzeniecka et al. 2025): set the component along
            # span(d_idx) to the overall dataset mean's (dim_mean), removing all
            # variation there including what overlapping atoms re-supply. Overall,
            # not per-tubelet: a per-position mean would restore the positional
            # signal. state["dirs"] overrides the dictionary rows for rand_iso7,
            # which has no feature indices of its own.
            base = flat if mode.startswith("raw") else sae.decode(z) + dim_mean
            Dk = state["dirs"] if state.get("dirs") is not None else dictionary[idx]
            coef = torch.linalg.solve(Dk @ Dk.T, ((base - dim_mean) @ Dk.T).T).T
            new = base - coef @ Dk
        else:
            raise ValueError(f"unknown mode: {mode}")
        new = new.to(hidden.dtype).reshape(B, T, D)
        out = torch.cat([cls, new], dim=1) if cls_offset else new
        return (out,) + output[1:] if isinstance(output, tuple) else out
    return hook_fn


def build_conditions(features: list[int]) -> list[tuple[str, str, list[int]]]:
    """(mode, target_name, indices). Target names match ablation_targets.py's
    single_*/all7 convention so these rows join straight onto the existing
    ablation_summary_l5_job7ep_k64.csv. The group carries the large effect
    (reference delta 0.487 vs ~0.02-0.05 per feature) — the singles are below
    the noise floor at 100 clips."""
    conds: list[tuple[str, str, list[int]]] = [("raw", "none", []), ("recon", "none", [])]
    modes = ("raw_minus", "recon_minus", "raw_meanproject", "recon_meanproject")
    for target, idx in [(f"single_{f}", [f]) for f in features] + [("all7", features)]:
        conds += [(m, target, idx) for m in modes]
    # Controls. rand_dict7 = 7 mass-matched SAE features, run under every
    # operation the scaffold gets, so subtraction and projection share one
    # control. rand_iso7 = 7 arbitrary directions: projection only, since
    # subtraction needs a z_i that arbitrary directions do not have.
    conds += [(m, "rand_dict7", []) for m in modes]
    conds += [(m, "rand_iso7", []) for m in ("raw_meanproject", "recon_meanproject")]
    return conds


def draw_mass_matched(z: torch.Tensor, scaffold: list[int], rng: np.random.Generator,
                      cfg: dict) -> tuple[list[int], float]:
    """7 non-scaffold features whose total activation mass in THIS clip is within
    tolerance of the scaffold's. Unmatched random features carry ~4% of the
    scaffold's mass, so subtracting them would be a trivially weak control.
    Falls back to the closest draw; the achieved ratio is returned for audit."""
    mass = z.abs().sum(dim=0).cpu()
    target = float(mass[scaffold].sum())
    ranked = mass.clone()
    ranked[scaffold] = -1.0
    pool = torch.argsort(ranked, descending=True)[:cfg["control_pool_n"]].numpy()
    best, best_ratio = None, None
    for _ in range(cfg["control_max_tries"]):
        pick = rng.choice(pool, size=len(scaffold), replace=False)
        ratio = float(mass[pick].sum()) / target
        if best is None or abs(ratio - 1) < abs(best_ratio - 1):
            best, best_ratio = pick, ratio
        if abs(ratio - 1) <= cfg["control_mass_tol"]:
            break
    return best.tolist(), best_ratio


def draw_control(target: str, z: torch.Tensor, dictionary: torch.Tensor,
                 scaffold: list[int], rng: np.random.Generator, cfg: dict) -> dict:
    """Drawn once per clip and shared by every operation, so subtraction and
    projection are tested against the same random features. rand_iso7 = 7
    arbitrary orthonormal directions (the floor — any 7-of-768 subspace)."""
    if target == "rand_dict7":
        idx, ratio = draw_mass_matched(z, scaffold, rng, cfg)
        return {"indices": idx, "dirs": None, "mass_ratio": ratio}
    gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
    dirs = torch.linalg.qr(torch.randn(dictionary.shape[1], len(scaffold), generator=gen)).Q.T
    return {"indices": [], "dirs": dirs.to(dictionary.device, dictionary.dtype), "mass_ratio": None}


def run_clip(model, pixel_values: torch.Tensor, class_id: int,
             conditions: list[tuple[str, str, list[int]]], state: dict, capture: dict,
             dictionary: torch.Tensor, scaffold: list[int],
             rng: np.random.Generator) -> tuple[list[dict], dict]:
    """h_raw and z are identical across all conditions (L5's input depends only on
    layers 0-4, which nothing here touches), so the snapshot is taken once."""
    rows, snapshot, drawn = [], {}, {}
    for mode, target, indices in conditions:
        # controls come after the "raw" pass, so capture["z"] already holds this
        # clip's latents (identical every pass — L5's input is never touched)
        if target.startswith("rand_") and target not in drawn:
            drawn[target] = draw_control(target, capture["z"], dictionary, scaffold, rng, CFG)
        ctl = drawn.get(target, {})
        state["mode"] = mode
        state["indices"] = ctl.get("indices", indices)
        state["dirs"] = ctl.get("dirs")
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits.squeeze(0)
        if not snapshot:
            z = capture["z"]
            nz = z.nonzero(as_tuple=False)
            snapshot = {"h_raw": capture["h_raw"].half().cpu(),
                        "z_indices": nz.to(torch.int32).cpu(),
                        "z_values": z[nz[:, 0], nz[:, 1]].half().cpu()}
        pred = int(logits.argmax())
        rows.append({"mode": mode, "ablation_target": target,
                     "correct_class_logit": float(logits[class_id]),
                     "predicted_class": pred, "correct": pred == class_id,
                     "control_mass_ratio": ctl.get("mass_ratio")})
    return rows, snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=CFG["n_clips"],
                        help="0 or less uses the full population")
    parser.add_argument("--no-activations", action="store_true",
                        help="skip the per-clip h_raw/z dump — 3.5MB/clip, only "
                             "needed for fingerprint work, not for logit deltas")
    args = parser.parse_args()
    cfg, device = CFG, CFG["device"]

    features = get_targets(cfg["dataset"], cfg["layer"])["all7"]
    conditions = build_conditions(features)
    print(f"Device: {device}  L{cfg['layer']} features: {features}  "
          f"{len(conditions)} passes/clip")

    model_cfg = MODEL_REGISTRY[cfg["model_flag"]]
    checkpoint = CHECKPOINT_REGISTRY[(cfg["model_flag"], cfg["dataset"])]
    processor = model_cfg["processor_class"].from_pretrained(checkpoint)
    model = model_cfg["model_class"].from_pretrained(checkpoint).to(device).eval()
    model.requires_grad_(False)

    sae, dim_mean = load_sae(cfg["model_flag"], cfg["layer"], cfg["sae_k"], device)
    dictionary = sae.dictionary.get_dictionary().detach()

    state, capture = {"mode": "raw", "indices": [], "dirs": None}, {}
    rng = np.random.default_rng(cfg["seed"])
    model_cfg["layer_getter"](model, cfg["layer"]).register_forward_hook(
        make_l5_hook(sae, dim_mean, dictionary, model_cfg["cls_offset"], state, capture))

    act_dir = cfg["out_dir"] / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    clips = sample_clips(cfg, args.n_clips)

    rows, t0 = [], time.time()
    for i, (clip_id, class_id, clip_path) in enumerate(clips):
        pixel_values = _preprocess_clip(clip_path, model_cfg["num_frames"], processor, device)
        clip_rows, snapshot = run_clip(model, pixel_values, class_id, conditions, state,
                                       capture, dictionary, features, rng)
        if not args.no_activations:
            torch.save({**snapshot, "clip_id": clip_id, "class_id": class_id},
                       act_dir / f"{clip_id}.pt")
        for r in clip_rows:
            rows.append({"clip_id": clip_id, "class_id": class_id, **r})
        print(f"  [{i+1}/{len(clips)}] {clip_id}  {(time.time()-t0)/(i+1):.1f}s/clip")

    df = pd.DataFrame(rows)
    out_path = cfg["out_dir"] / f"raw_stream_ablation_l{cfg['layer']}_n{len(clips)}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n  {len(df):,} rows -> {out_path}")
    print(f"  activations -> {act_dir}")


if __name__ == "__main__":
    main()
