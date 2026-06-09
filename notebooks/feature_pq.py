import numpy as np
import pandas as pd
import os
import sys
df = pd.read_parquet("/Users/gq25877/Temp_Or_Text/temporal-or-textural/outputs/analysis/chiral_extraction.parquet")

feature_id = 1623
threshold = 17

r2l = df[df["class_id"] == 94]

r_tfc = np.stack([np.array(v) for v in r2l["r_token_fire_counts"]])
a_tfc = np.stack([np.array(v) for v in r2l["a_token_fire_counts"]])
b_tfc = np.stack([np.array(v) for v in r2l["b_token_fire_counts"]])

r_active  = r_tfc[:, feature_id] >= threshold
a_absent  = a_tfc[:, feature_id] < threshold
b_absent  = b_tfc[:, feature_id] < threshold
survives  = r_active & a_absent & b_absent

print(f"Feature {feature_id} in R→L clips:")
print(f"  n_clips_active_R:       {r_active.sum()}")
print(f"  n_clips_surviving_filter: {survives.sum()}")
print(f"  per-clip surviving: {survives.astype(int).tolist()}")