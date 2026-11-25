"""Which SSv2 classes need the frame label? (19/11/26). Per class: accuracy under frozen PE (no frame label: a bag of
frames, umt_frozen_pe_groups.py, 4 clips, averaged over the L5 / L7 / L9 spliced runs) vs accuracy under frame shuffle
(label kept, content order scrambled; all-174 thrfix DFA parquets, <= 40 R-correct clips, averaged over layers).
Frozen-only loss = needs the frame label to bind sightings ("one thing moved" vs "two things"), not a specific order.
Out: outputs/analysis/umt_frozen_pe_groups/classes.csv. CPU, local.
Usage: uv run python src/stage3_analysis/umt_frozen_pe_classes.py"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
CFG = {"layers": (5, 7, 9), "frozen": ROOT / "outputs/analysis/umt_frozen_pe_groups",
       "dfa": str(ROOT / "outputs/analysis/dfa_mass_delta_umt/dfa_mass_delta_umt_ssv2_all174_thrfix_l{}_job7ep_k64.parquet"),
       "names": ROOT / "data/ssv2/labels/labels.json", "out": ROOT / "outputs/analysis/umt_frozen_pe_groups/classes.csv"}


def frozen_by_class(cfg):
    """Per class: full and frozen-PE accuracy and correct-logit change, mean over clips and the three spliced runs."""
    y = pd.read_csv(cfg["frozen"] / "clips.csv").class_id.to_numpy()
    out = []
    for L in cfg["layers"]:
        z = np.load(cfg["frozen"] / f"L{L}.npz")
        yy, c, F, Z = y[z["row"]], np.arange(len(z["row"])), z["full_logits"], z["frozen_logits"]
        out.append(pd.DataFrame({"class_id": yy, "full_acc": F.argmax(1) == yy, "frozen_acc": Z.argmax(1) == yy,
                                 "dlogit": Z[c, yy] - F[c, yy]}))
    return pd.concat(out).groupby("class_id").mean()


def shuffle_by_class(cfg):
    """Per class: shuffle accuracy on R-correct clips (mean over the three layer files) + label, clip count."""
    d = pd.concat([pd.read_parquet(cfg["dfa"].format(L), columns=["class_id", "sl_label", "correct_C"])
                   for L in cfg["layers"]])
    return d.groupby("class_id").agg(label=("sl_label", "first"), shuffle_acc=("correct_C", "mean"),
                                     n_shuffle=("correct_C", "size")).assign(n_shuffle=lambda t: t.n_shuffle // 3)


def main(cfg):
    t = shuffle_by_class(cfg).join(frozen_by_class(cfg), how="inner")
    t["name"] = t.index.map({int(v): k for k, v in json.loads(cfg["names"].read_text()).items()})
    t["shuffle_loss"] = 1 - t.shuffle_acc                        # clips are R-correct, so baseline is 1
    t["frozen_loss"] = t.full_acc - t.frozen_acc
    t["frozen_only"] = t.frozen_loss - t.shuffle_loss          # + = needs the frame label beyond order
    t.round(3).to_csv(cfg["out"])
    print("classes", len(t), "| spearman shuffle_loss vs frozen_loss %.2f (p %.1g)" % spearmanr(t.shuffle_loss, t.frozen_loss))
    print(t[["shuffle_loss", "frozen_loss", "dlogit"]].describe().loc[["mean", "50%"]].round(2))
    cols = ["name", "label", "shuffle_loss", "frozen_loss", "frozen_only"]
    for name, s in (("frozen only (survive shuffle, die frozen)", t.frozen_only.nlargest(15)),
                    ("survive both", (t.frozen_loss + t.shuffle_loss).nsmallest(15))):
        print("\n" + name); print(t.loc[s.index, cols].round(2).to_string())


if __name__ == "__main__":
    main(CFG)
