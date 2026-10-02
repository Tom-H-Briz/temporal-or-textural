"""
Raw-residual pairing of pos_lock_synthetic_probe.py: do the scaffold's decoder
DIRECTIONS carry a position-locked signal in the UNMODIFIED model on
content-free video — no SAE splice, no SAE encoder anywhere?

Same inputs as the SAE probe, imported from it so they cannot drift: white /
black / 16 seeds of iid noise, 16 identical-content frames, so the positional
encoding is the only thing that differs across the 8 tubelets. Same configs and
the same strict scaffold members.

Readout, per member direction d_i (unit-normalised SAE decoder row), per tubelet:
    proj  = sum over the 196 patches of <h - dim_mean, d_i>   (plain projection)
    coef  = the same but least-squares coefficients on all members' directions
            jointly, removing crosstalk between the non-orthogonal directions
The SAE probe sums z over patches; this sums the raw readout over patches.

Chance baseline: on constant content every direction has SOME tubelet profile,
driven by the positional encoding alone. So for each config, 200 random
non-member decoder directions are drawn once and read out the same way on every
video; chance_at_locked is the fraction of them whose argmax lands on the
member's locked tubelet.

Outputs (outputs/analysis/pos_lock_synthetic/):
  pos_lock_synthetic_probe_raw.csv     long, one row per (config, member, video)
  pos_lock_synthetic_probe_paired.csv  joined with the SAE probe's rows
Usage: uv run python notebooks/pos_lock_synthetic_probe_raw.py   (CPU, minutes)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import pos_lock_synthetic_probe as sae_probe
from ToT_utils import (
    CHECKPOINT_REGISTRY, MODEL_REGISTRY, gather_by_position, get_processor,
    resolve_sae_checkpoint,
)

CFG = {
    **{k: sae_probe.CFG[k] for k in ("model_name", "sae_k", "job_label", "configs")},
    "n_random_dirs": 200,
    "seed": 0,
    "sae_csv": sae_probe.CFG["output_dir"] / "pos_lock_synthetic_probe.csv",
    "output_dir": sae_probe.CFG["output_dir"],
}


def load_directions(dataset_name: str, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(dict_size, 768) unit-normalised decoder rows and the SAE's dim_mean."""
    r = resolve_sae_checkpoint(CFG["model_name"], layer, dataset_name, CFG["sae_k"], CFG["job_label"])
    ckpt = torch.load(r["sae_path"], map_location="cpu", weights_only=True)
    w = (ckpt["sae_state_dict"] if "sae_state_dict" in ckpt else ckpt)["dictionary._weights"].float()
    mean = torch.load(r["dim_mean_path"], map_location="cpu", weights_only=True).float()
    return w / w.norm(dim=1, keepdim=True), mean


def capture_layers(model, layers: list[int]) -> tuple[dict, list]:
    """Pure-observation hooks on every requested layer — the model is not modified."""
    store, handles = {}, []
    getter = MODEL_REGISTRY[CFG["model_name"]]["layer_getter"]
    for L in layers:
        def hook(module, inp, out, L=L):
            store[L] = (out[0] if isinstance(out, tuple) else out)[0].detach().float().cpu()
        handles.append(getter(model, L).register_forward_hook(hook))
    return store, handles


def per_tubelet(h: torch.Tensor, mean: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """(8, k): readout of each direction summed over the 196 patches per tubelet."""
    return gather_by_position((h - mean) @ dirs.T, CFG["model_name"]).sum(dim=1)


def member_rows(proj, coef, rand_argmax, members, meta) -> list[dict]:
    rows = []
    for j, (feat, locked) in enumerate(members):
        p, c = proj[:, j], coef[:, j]
        others = torch.cat([p[:locked], p[locked + 1:]])
        pos = p.clamp(min=0)
        rows.append({
            **meta, "feature_idx": feat, "locked_position": locked,
            "proj_at_locked": round(float(p[locked]), 4),
            "share_at_locked_pos": round(float(pos[locked] / pos.sum()), 4) if pos.sum() > 0 else 0.0,
            "margin_z": round(float((p[locked] - others.mean()) / others.std()), 3),
            "argmax_position": int(p.argmax()),
            "fired_at_locked": int(p.argmax()) == locked,
            "coef_argmax_position": int(c.argmax()),
            "coef_fired_at_locked": int(c.argmax()) == locked,
            "chance_at_locked": round(float((rand_argmax == locked).float().mean()), 4),
        })
    return rows


def run_dataset(dataset_name: str, layers: list[int], rng: np.random.Generator) -> list[dict]:
    model_cfg = MODEL_REGISTRY[CFG["model_name"]]
    ckpt = CHECKPOINT_REGISTRY[(CFG["model_name"], dataset_name)]
    processor = get_processor(model_cfg, ckpt)
    model = model_cfg["model_class"].from_pretrained(ckpt).eval()
    store, handles = capture_layers(model, layers)
    setup = {}
    for L in layers:
        members = sae_probe.scaffold_members(dataset_name, L)
        dirs, mean = load_directions(dataset_name, L)
        feats = [f for f, _ in members]
        pool = np.setdiff1d(np.arange(dirs.shape[0]), feats)
        rand = dirs[rng.choice(pool, CFG["n_random_dirs"], replace=False).tolist()]
        setup[L] = (members, dirs[feats], mean, rand)
    rows = []
    for content, seed in sae_probe.synthetic_videos():
        pv = processor(sae_probe.make_frames(content, seed), return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            model(pixel_values=pv)
        for L in layers:
            members, Dk, mean, rand = setup[L]
            proj = per_tubelet(store[L], mean, Dk)
            coef = proj @ torch.linalg.inv(Dk @ Dk.T)          # joint least-squares coefficients
            rand_argmax = per_tubelet(store[L], mean, rand).argmax(dim=0)
            meta = {"dataset": dataset_name, "layer": L, "content": content, "seed": seed}
            rows += member_rows(proj, coef, rand_argmax, members, meta)
    for hd in handles:
        hd.remove()
    return rows


def main() -> None:
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    for dataset_name in dict.fromkeys(d for d, _ in CFG["configs"]):
        layers = [L for d, L in CFG["configs"] if d == dataset_name and sae_probe.scaffold_members(d, L)]
        rows += run_dataset(dataset_name, layers, rng)
        print(f"{dataset_name}: layers {layers} done")
    raw = pd.DataFrame(rows)
    raw.to_csv(CFG["output_dir"] / "pos_lock_synthetic_probe_raw.csv", index=False)

    keys = ["dataset", "layer", "feature_idx", "content", "seed"]
    sae = pd.read_csv(CFG["sae_csv"])[keys + ["fired_at_locked", "share_at_locked"]]
    paired = raw.merge(sae.rename(columns={"fired_at_locked": "sae_fired_at_locked",
                                           "share_at_locked": "sae_share_at_locked"}), on=keys)
    paired.to_csv(CFG["output_dir"] / "pos_lock_synthetic_probe_paired.csv", index=False)
    summary = (paired.groupby(["dataset", "layer", "content"])
               .agg(n=("feature_idx", "size"),
                    sae_fire=("sae_fired_at_locked", "mean"),
                    raw_fire=("fired_at_locked", "mean"),
                    raw_fire_joint=("coef_fired_at_locked", "mean"),
                    chance=("chance_at_locked", "mean"),
                    raw_share_pos=("share_at_locked_pos", "mean"),
                    margin_z=("margin_z", "median"))
               .round(3).reset_index())
    print(f"\nrows: raw {len(raw)}, paired {len(paired)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
