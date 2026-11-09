"""VideoMAE port of umt_object_contrast (09/11/26): UMT's clip sets (umt_{task}_contrast/sets.csv),
summaries, selection and controls; VM's own conditions R / C1 / Rev (C1, not C: C tears tubelets).
Pairs: attribution = grad x z of the classifier's logit(home) - logit(partner), R. Objects: VM has no
noun readout, so selection is AUROC only (need_attr=False). Ablation: umt_object_vtc.py MODEL=videomae.
Modes: extract (GPU) | select (CPU) | controls (CPU) | all
Usage: LAYER=7 TASK=pair MODE=all MAX_CLIPS=2 uv run python src/stage3_analysis/vm_object_contrast.py
"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
import umt_object_contrast as oc
L, TASK = oc.L, oc.CFG["task"]
CFG = dict(oc.CFG, model="videomae", conds=("R", "C1", "Rev"), need_attr=TASK == "pair",
           out_dir=ROOT / f"outputs/analysis/videomae_{TASK}_contrast_L{L}")


def clip_loader(clips, cfg, cond, processor):
    """Batch-1 (pv, row-index) loader: VM's own dataset class, sampler and processor (R / C1);
    Rev = R flipped in time by the caller. Row indices as labels, as umt_mlm_feature_contrast."""
    from ToT_utils import MODEL_REGISTRY
    from perturb_accuracy_vm_ssv2 import PerturbedSSv2Dataset
    paths = [Path(cfg["video_dir"]) / f"{i}.webm" for i in clips.clip_id]
    ds = PerturbedSSv2Dataset(paths, clips.clip_id.tolist(), list(range(len(clips))), processor,
                              MODEL_REGISTRY["videomae"]["num_frames"], "C1" if cond == "C1" else "R")
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["num_workers"])


def pair_attr(st, pv, x, z0, t, p):
    """grad x z of logit(t) - logit(p), x' = x + decode(z - z0) spliced at layer L (UMT noun_attr's
    path, classifier in place of the MLM head); summed over tokens -> (n_lat,)."""
    from vision_pair_phase3 import forward_T
    z = z0.clone().requires_grad_(True)
    T = forward_T(st, pv, (x + st["sae"].decode(z - z0)).unsqueeze(0), t, p)[0]
    return (torch.autograd.grad(T, z)[0] * z0).sum(0).cpu().numpy()


def extract_cond(cfg, st, clips, homes, cond):
    """One condition over the selection clips -> summary arrays (oc.summaries), and under R each pair
    clip's attribution toward its home class. homes: {clip_id: (pair key, home, partner)}."""
    from umt_vision_pair_contrast import apply_condition
    S = {k: np.full((len(clips), cfg["n_lat"]), np.nan, np.float16) for k in cfg["summaries"]}
    attr = {}
    for pv, i in clip_loader(clips, cfg, cond, st["processor"]):
        i = int(i); c = clips.clip_id.iloc[i]
        pv = apply_condition(pv, cond).to(cfg["device"])
        with torch.no_grad():
            st["model"](pixel_values=pv)              # hook fills st["x"] (1, N, D)
            x = st["x"][0].float()
            z0 = st["sae"].encode(x - st["dm"])[1]
        for k, v in oc.summaries(z0, cfg).items():
            S[k][i] = v.cpu().numpy()
        if cond == "R" and c in homes:
            key, t, p = homes[c]
            attr[(c, key)] = pair_attr(st, pv, x, z0, t, p)
    return S, attr


def run_extract(cfg, st, sets):
    """summ_{cond}.npz per condition and attr.npz, oc's layout. Pairs: home attribution (R).
    Objects: zeros per object row (VM has no noun readout; need_attr=False skips the check)."""
    sel = sets[sets.role != "test"]
    clips = pd.DataFrame({"clip_id": sorted(sel.clip_id.unique())})
    partner = lambda key, h: [int(x) for x in key.split("_") if int(x) != h][0]
    homes = {} if TASK == "object" else {r.clip_id: (r.object, int(r.true_class), partner(r.object, int(r.true_class)))
                                         for r in sel.itertuples()}
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    for cond in cfg["conds"]:
        t0 = time.time()
        S, attr = extract_cond(cfg, st, clips, homes, cond)
        np.savez(cfg["out_dir"] / f"summ_{cond}.npz", clip_id=clips.clip_id.to_numpy(dtype="U"), **S)
        if cond == "R":
            attr = attr or {(r.clip_id, r.object): np.zeros(cfg["n_lat"], np.float32) for r in sel[sel.role == "pos"].itertuples()}
            k = list(attr)
            np.savez(cfg["out_dir"] / "attr.npz", clip_id=np.array([c for c, _ in k], dtype="U"),
                     object=np.array([o for _, o in k], dtype="U"), attr=np.stack([attr[x] for x in k]))
        print(f"  {cond}: {int(np.isfinite(S['mean'][:, 0]).sum())}/{len(clips)} clips, {time.time() - t0:.0f}s", flush=True)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 paths, as the references
    torch.backends.cudnn.allow_tf32 = False
    mode, cfg = os.environ.get("MODE", "all"), dict(CFG)
    if mode not in ("extract", "select", "controls", "all"):
        sys.exit(f"MODE={mode}: use extract | select | controls | all")
    sets = pd.read_csv(cfg["sets_dir"] / "sets.csv", dtype={"clip_id": str})   # UMT's sets: same clips
    if cfg["max_clips"]:                              # local checks never overwrite the run
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_smoke")
        sets, cfg["ctrl_band"] = oc.smoke_sets(sets, cfg["max_clips"]), 0.5
    print(f"mode={mode} task={TASK} layer={L} out={cfg['out_dir']}", flush=True)
    if mode in ("extract", "all"):
        import vision_pair_phase3
        run_extract(cfg, vision_pair_phase3.setup(cfg), sets)
    if mode in ("select", "all"):
        oc.run_select(cfg, sets)
    if mode in ("controls", "all"):
        oc.run_controls(cfg, sets)


if __name__ == "__main__":
    main()
