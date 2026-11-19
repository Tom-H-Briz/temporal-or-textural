"""VM synthetic readout (19/11/26): are VM's top-20 L9 'down' candidates (pair 45/43, home 43) all down-motion
detectors, or something else? Directional suite (ball / pyramid x 8 directions x 2 speeds + static), resampled from
the 24 fps webms to VM's 16 frames (same uniform sampler as SSv2). Per feature: clip-summed L9 activation + profile.
Usage: uv run python src/stage3_analysis/vm_synthetic_down_readout.py"""
import json, os, sys
from pathlib import Path
import av, numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
A = ROOT / "outputs/analysis"
CFG = dict(model="videomae", layer=9, pair="45_43", home=43, n_top=20,
           cands=A / "videomae_pair_contrast_L9/candidates_45_43.csv",
           suite=ROOT / "outputs/synthetic_motion/directional_clips",
           real=A / "vm_clock_downstream_L5_raw",          # baseline L9 profiles on real down / up clips
           device="cuda" if torch.cuda.is_available() else "cpu",
           out_dir=Path(os.environ.get("OUT_DIR", str(A / "vm_synthetic_down_readout"))))


def top_features(cfg):
    """Top n down-side candidates in file order (AUROC, best first); passers and non-passers alike.
    FEATS=a,b,... overrides with any latents (no candidate columns: auroc / attr / passed NaN)."""
    if os.environ.get("FEATS"):
        f = [int(x) for x in os.environ["FEATS"].split(",")]
        return pd.DataFrame(dict(latent=f, auroc=np.nan, attr_mean=np.nan, passed=np.nan))
    c = pd.read_csv(cfg["cands"])
    return c[c.home == cfg["home"]].head(cfg["n_top"]).reset_index(drop=True)


def load_clip(path, processor, n_frames):
    """24 fps webm -> n_frames picked uniformly (ToT_utils.sample_frames_ssv2) -> processor pixel_values."""
    from ToT_utils import sample_frames_ssv2
    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    idx = sample_frames_ssv2(len(frames), n_frames)
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"]


def run_suite(cfg, feats):
    """Every suite clip through VM -> L9 SAE codes of feats, per tubelet: dict name -> (8, n_feats)."""
    import vision_pair_phase3
    from ToT_utils import MODEL_REGISTRY
    st = vision_pair_phase3.setup(cfg)
    n_frames, n_pos = MODEL_REGISTRY["videomae"]["num_frames"], 8
    meta, out = json.load(open(cfg["suite"] / "metadata.json")), {}
    for m in meta:
        pv = load_clip(cfg["suite"] / "full_rate" / f"{m['name']}.webm", st["processor"], n_frames)
        with torch.no_grad():
            st["model"](pixel_values=pv.to(cfg["device"]))
            z = st["sae"].encode(st["x"][0].float() - st["dm"])[1][:, feats]       # (1568, n_feats)
        out[m["name"]] = z.reshape(n_pos, -1, len(feats)).sum(1).cpu().numpy()
    return meta, out


def real_levels(cfg, feats):
    """Mean clip-summed baseline L9 activation on real down (43) / up (45) clips (11/11 raw run, arm 0)."""
    c = pd.read_csv(cfg["real"] / "clips.csv")
    p = np.load(cfg["real"] / "prof_L9_R.npy", mmap_mode="r")             # (clips, arms, 8, F)
    lv = {}
    for name, cls in (("real_down", 43), ("real_up", 45)):
        rows = np.where(c.class_id.values == cls)[0]
        lv[name] = np.asarray(p[rows, 0][:, :, feats], dtype=np.float32).sum(1).mean(0)
    return lv


def summarise(top, meta, prof, lv):
    """Per feature: mean clip sum per direction (over shape x speed), static, preferred direction,
    down index (090 - 270) / (090 + 270), direction tuning max / mean, best synthetic vs real down."""
    long = pd.DataFrame([dict(clip=m["name"], shape=m["shape"], angle=m["angle_deg"], speed=m["speed_label"],
                              latent=int(f), act=float(prof[m["name"]][:, j].sum()))
                         for m in meta for j, f in enumerate(top.latent)])
    mv = long[long.angle.notna()].groupby(["latent", "angle"]).act.mean().unstack()
    s = top[["latent", "auroc", "attr_mean", "passed"]].set_index("latent")
    s["real_down"], s["real_up"] = lv["real_down"], lv["real_up"]
    s["static"] = long[long.angle.isna()].groupby("latent").act.mean()
    s["pref_dir"] = mv.idxmax(1).astype(int)
    s["down_idx"] = (mv[90] - mv[270]) / (mv[90] + mv[270] + 1e-6)
    s["tuning"] = mv.max(1) / (mv.mean(1) + 1e-6)
    s["best_synth_vs_real"] = long.groupby("latent").act.max() / (s.real_down + 1e-6)
    return long, mv, s


def main(cfg):
    top = top_features(cfg)
    feats = top.latent.astype(int).tolist()
    meta, prof = run_suite(cfg, feats)
    long, mv, s = summarise(top, meta, prof, real_levels(cfg, feats))
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    np.savez(cfg["out_dir"] / "profiles.npz", latents=np.array(feats), **prof)    # clip -> (8, n_feats)
    long.to_csv(cfg["out_dir"] / "per_clip.csv", index=False)
    mv.to_csv(cfg["out_dir"] / "by_direction.csv")
    s.to_csv(cfg["out_dir"] / "summary.csv")
    print(mv.round(0).to_string(), "\n", s.round(2).to_string(), flush=True)


if __name__ == "__main__":
    main(CFG)
