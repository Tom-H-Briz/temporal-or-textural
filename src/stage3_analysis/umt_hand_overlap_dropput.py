"""Hand overlap on dropping/putting clips. Per clip (R, selection), per latent:
Pearson r of its (12 x 196) activation map with hand tracker #2782's map; mean over clips
where both are active, overall and by verb (dropping / putting). All 6144 latents.
Hand = where tracker #2782 fires (template from the up/down hand alignment, 15/10).
Output: outputs/analysis/umt_pair_table/hand_overlap_dropput.csv
Usage: uv run python src/stage3_analysis/umt_hand_overlap_dropput.py   (CPU, local sparse/R)"""
import numpy as np, pandas as pd
from pathlib import Path

A = Path("outputs/analysis")
HAND, NL, NT = 2782, 6144, 12 * 196
PAIRS = {(8, 104): "_tn_a", (10, 106): "_tn_b", (12, 112): "_tn_b"}
DROP, PUT = {8, 10, 12}, {104, 106, 112}
OUT = A / "umt_pair_table/hand_overlap_dropput.csv"


def clip_list():
    rows = []
    for (a, b), suf in PAIRS.items():
        d = A / f"umt_mlm_feature_contrast{suf}"
        p = pd.read_parquet(d / "per_clip.parquet")
        p = p[(p.condition == "R") & (p.split == "selection") & (p.a == a) & (p.b == b)]
        rows += [(d, c, t) for c, t in zip(p.clip_id, p.true_class)]
    df = pd.DataFrame(rows, columns=["dir", "clip_id", "cls"])
    return df.drop_duplicates("clip_id")          # a 106/112 clip sits in two pairs


def clip_r(d, cid):
    """(NL,) Pearson r of each latent's map with the hand map; nan where either is flat."""
    s = np.load(d / "sparse" / "R" / f"{cid}.npz")
    M = np.zeros((NL, NT), np.float32)
    np.add.at(M, (s["lat"].astype(int), s["tok"].astype(int)), s["z"].astype(np.float32))
    Mc = M - M.mean(1, keepdims=True)
    sd = np.sqrt((Mc ** 2).sum(1))
    h, hs = Mc[HAND], sd[HAND]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Mc @ h) / (sd * hs)


if __name__ == "__main__":
    clips = clip_list()
    R = np.full((len(clips), NL), np.nan, np.float32)
    for i, r in enumerate(clips.itertuples()):
        R[i] = clip_r(r.dir, r.clip_id)
        if i % 200 == 0:
            print(i, len(clips), flush=True)
    drop = clips.cls.isin(DROP).to_numpy()
    out = pd.DataFrame(dict(feature=np.arange(NL), overlap=np.nanmean(R, 0),
                            overlap_drop=np.nanmean(R[drop], 0), overlap_put=np.nanmean(R[~drop], 0),
                            frac_clips=np.isfinite(R).mean(0)))
    out.to_csv(OUT, index=False)
    print("hand #2782 active on", np.isfinite(R[:, HAND]).mean(), "of", len(clips), "clips")
