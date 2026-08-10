"""
L5 ablation impact on L7's own feature space.

Zeros VM's confirmed L5 position-locked scaffold (all7) and measures how
much L7's own SAE feature space shifts as a result — cosine similarity and
L2 distance between L7's per-clip feature vector under baseline (L5 SAE-
reconstructed, not ablated) vs L5-ablated, same clip. Neither the existing
L5 nor L7 ablation run captured this — both only recorded final-logit
impact, never L7's intermediate activations.

Baseline uses ablate_indices=[] through the SAME L5 SAE-reconstruction
hook (not a raw unperturbed forward pass) — isolates the effect of zeroing
the scaffold specifically, holding SAE reconstruction noise constant,
matching how baseline_logit is defined throughout every other ablation
script this session.

Population: same clip set the L5 ablation run used (R-correct,
ablation_results_long_l5_job7ep_k64.parquet, target=all7).

Outputs (outputs/analysis/scaffold_ablation/):
    l5_ablation_l7_feature_impact.parquet — clip_id, class_id, cosine_similarity,
    l2_distance, relative_l2, l7_z_baseline, l7_z_ablated (both (6144,) — full
    per-feature L7 vectors, not just the collapsed scalars, so any later
    question can be answered from this one extraction without rerunning it)

Usage:
    uv run python src/stage3_analysis/l5_ablation_l7_feature_impact.py
    uv run python src/stage3_analysis/l5_ablation_l7_feature_impact.py --dry-run
"""

import argparse
import sys
import time
from pathlib import Path

import av
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from sae import BatchTopKSAE
from stage3_analysis.ablation_targets import get_targets
from stage3_analysis.dfa_engine import _preprocess_clip
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, resolve_sae_checkpoint, sample_frames_ssv2

CFG = {
    "model_flag": "videomae",
    "l5_layer": 5,
    "l7_layer": 7,
    "sae_k": 64,
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "video_dir": ROOT / "data/ssv2/20bn-something-something-v2",
    "l5_ablation_parquet": ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_l5_job7ep_k64.parquet",
    "out_dir": ROOT / "outputs/analysis/scaffold_ablation",
}
L5_SCAFFOLD = get_targets("ssv2", 5)["all7"]


def load_clips(cfg: dict) -> list[tuple[str, int, Path]]:
    """Same clip set the L5 ablation run used — R-correct, all7 target."""
    df = pd.read_parquet(cfg["l5_ablation_parquet"], columns=["clip_id", "class_id",
                                                              "perturbation_condition", "ablation_target"])
    df = df[(df["perturbation_condition"] == "R") & (df["ablation_target"] == "all7")]
    df = df.drop_duplicates("clip_id")
    video_dir = Path(cfg["video_dir"])
    result = []
    for _, row in df.iterrows():
        path = video_dir / f"{row['clip_id']}.webm"
        if path.exists():
            result.append((str(row["clip_id"]), int(row["class_id"]), path))
    print(f"  {len(result):,} clips")
    return result


def load_sae(model_flag: str, layer: int, sae_k: int, device: str) -> tuple:
    resolved = resolve_sae_checkpoint(model_flag, layer, dataset_name="ssv2", sae_k=sae_k)
    model_cfg = MODEL_REGISTRY[model_flag]
    ckpt = torch.load(resolved["sae_path"], weights_only=True, map_location=device)
    state_dict = ckpt["sae_state_dict"] if "sae_state_dict" in ckpt else ckpt
    nb_concepts = state_dict["dictionary._weights"].shape[0]
    top_k = resolved["sae_k"] * model_cfg["num_patch_tokens"]
    sae = BatchTopKSAE(input_shape=model_cfg["hidden_dim"], nb_concepts=nb_concepts,
                       top_k=top_k, device=device)
    sae.load_state_dict(state_dict)
    dim_mean = torch.load(resolved["dim_mean_path"], weights_only=True, map_location=device)
    sae.train()
    dummy = torch.zeros(model_cfg["num_patch_tokens"], model_cfg["hidden_dim"], device=device)
    with torch.no_grad():
        sae.encode((dummy - dim_mean).float())
    sae.eval().requires_grad_(False)
    return sae, dim_mean


def make_l5_hook(l5_sae, l5_dim_mean, cls_offset: int, state: dict):
    """Same splice-and-reconstruct logic as DFAEngine._splice_hook, but
    ablate_indices is read from `state` each call so one hook serves both
    the baseline ([]) and ablated (L5_SCAFFOLD) passes."""
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        cls, patches = hidden[:, :cls_offset], hidden[:, cls_offset:]
        B, T, D = patches.shape
        tokens = (patches.reshape(B * T, D) - l5_dim_mean).float()
        with torch.no_grad():
            _, z = l5_sae.encode(tokens)
        if state["ablate_indices"]:
            z[:, state["ablate_indices"]] = 0.0
        recon = l5_sae.decode(z)
        recon = (recon + l5_dim_mean).to(hidden.dtype).reshape(B, T, D)
        out = torch.cat([cls, recon], dim=1) if cls_offset else recon
        return (out,) + output[1:] if isinstance(output, tuple) else out
    return hook_fn


def make_l7_capture_hook(capture: dict):
    """Pure observation — captures L7's hidden state, changes nothing."""
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        capture["hidden"] = hidden.detach()
    return hook_fn


def run_clip(pixel_values: torch.Tensor, model, l7_sae, l7_dim_mean, l7_cls_offset: int,
            state: dict, capture: dict) -> dict:
    vectors = {}
    for name, indices in [("baseline", []), ("ablated", L5_SCAFFOLD)]:
        state["ablate_indices"] = indices
        with torch.no_grad():
            model(pixel_values=pixel_values)
        l7_hidden = capture["hidden"]
        l7_patches = (l7_hidden[0, l7_cls_offset:, :] - l7_dim_mean).float()
        with torch.no_grad():
            _, z = l7_sae.encode(l7_patches)
        vectors[name] = z.abs().sum(dim=0).cpu()  # (dict_size,) per-clip L7 feature vector

    a, b = vectors["baseline"], vectors["ablated"]
    cosine = float(torch.dot(a, b) / (a.norm() * b.norm()))
    l2 = float((a - b).norm())
    relative_l2 = l2 / float(a.norm())
    return {
        "cosine_similarity": cosine, "l2_distance": l2, "relative_l2": relative_l2,
        # full per-feature vectors, not just the collapsed scalars — the forward
        # passes are the expensive part; persisting these lets any downstream
        # question (which features moved, does it hit L7's own scaffold, per-
        # class breakdown) be answered later without rerunning the extraction.
        "l7_z_baseline": a.numpy(), "l7_z_ablated": b.numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = CFG
    device = cfg["device"]
    print(f"Device: {device}  L5 scaffold: {L5_SCAFFOLD}")

    model_cfg = MODEL_REGISTRY[cfg["model_flag"]]
    checkpoint = CHECKPOINT_REGISTRY[(cfg["model_flag"], "ssv2")]
    processor = model_cfg["processor_class"].from_pretrained(checkpoint)
    model = model_cfg["model_class"].from_pretrained(checkpoint).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    l5_sae, l5_dim_mean = load_sae(cfg["model_flag"], cfg["l5_layer"], cfg["sae_k"], device)
    l7_sae, l7_dim_mean = load_sae(cfg["model_flag"], cfg["l7_layer"], cfg["sae_k"], device)

    l5_state, l7_capture = {"ablate_indices": []}, {}
    model_cfg["layer_getter"](model, cfg["l5_layer"]).register_forward_hook(
        make_l5_hook(l5_sae, l5_dim_mean, model_cfg["cls_offset"], l5_state))
    model_cfg["layer_getter"](model, cfg["l7_layer"]).register_forward_hook(
        make_l7_capture_hook(l7_capture))

    clips = load_clips(cfg)
    if args.dry_run:
        clips = clips[:10]
        print("DRY RUN — 10 clips only")

    rows, t_start = [], time.time()
    for i, (clip_id, class_id, clip_path) in enumerate(clips):
        pixel_values = _preprocess_clip(clip_path, model_cfg["num_frames"], processor, device)
        result = run_clip(pixel_values, model, l7_sae, l7_dim_mean, model_cfg["cls_offset"], l5_state, l7_capture)
        result.update({"clip_id": clip_id, "class_id": class_id})
        rows.append(result)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(clips)}]  {elapsed/(i+1):.2f}s/clip")

    df = pd.DataFrame(rows)
    name = "l5_ablation_l7_feature_impact_dry_run.parquet" if args.dry_run else "l5_ablation_l7_feature_impact.parquet"
    out_path = cfg["out_dir"] / name
    df.to_parquet(out_path, index=False)
    print(f"  {len(df):,} clips -> {out_path}")
    print(df[["cosine_similarity", "l2_distance", "relative_l2"]].describe())


if __name__ == "__main__":
    main()
