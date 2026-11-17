"""Per-frame L3 activation (z) and T_ab contribution (dfa) of selected L3 MLM-contrast picks on the 45/43 clips,
R / C / Rev, from umt_mlm_feature_contrast_L3/sparse. Out: outputs/analysis/umt_l3_pe_lineage/l3_contrast_profiles_45_43.parquet
(lat, frame, z, c, cond, cls, clip). Usage: uv run python src/stage3_analysis/umt_l3_contrast_profiles.py (repo root, CPU)."""
import numpy as np, pandas as pd
D = "outputs/analysis/umt_mlm_feature_contrast_L3/"
F = [4038, 693, 5478, 6017, 6105, 5927, 1792, 652]
pc = pd.read_parquet(D + "per_clip.parquet").query("a == 45")
cls = pc.drop_duplicates("clip_id").set_index("clip_id").true_class
rows = []
for cond in ("R", "C", "Rev"):
    for cid, c in cls.items():
        s = np.load(f"{D}sparse/{cond}/{cid}.npz")
        keep = np.isin(s["lat"], F)
        df = pd.DataFrame(dict(lat=s["lat"][keep], frame=s["tok"][keep] // 196,
                               z=s["z"][keep].astype(float), c=s["dfa"][keep]))
        g = df.groupby(["lat", "frame"]).agg(z=("z", "sum"), c=("c", "sum")).reset_index()
        rows.append(g.assign(cond=cond, cls=c, clip=cid))
out = pd.concat(rows)
out.to_parquet("outputs/analysis/umt_l3_pe_lineage/l3_contrast_profiles_45_43.parquet")
print(out.groupby(["lat", "cls", "cond"]).z.sum().unstack().div(cls.value_counts(), level="cls").round(1))
