"""Verb vs noun L9 targets in pure PE / mixed / content terms (20/11/26). CPU, local. Same buckets as
umt_4250_breakdown.py (dictionary rule, cut blank >= 0.8 x real, L5). Per target: (1) linear attribution from every L5
latent summed by bucket; (2) top-20 single ablations (feature sheets) re-bucketed. Verb - noun gap with exact
permutation p over the 35 label splits. Out: outputs/analysis/umt_verb_noun_breakdown/.
Usage: uv run python src/stage3_analysis/umt_verb_noun_breakdown.py"""
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "stage3_analysis")]
A = ROOT / "outputs/analysis"
TARGETS = [("down #4250", "verb", "4250"), ("push #5083", "verb", "push_5083"), ("behind #491", "verb", "behind_491"),
           ("front #768", "verb", "front_768"), ("ball #713", "noun", "ball_713"), ("paper #4228", "noun", "paper_4228"),
           ("cup #6011", "noun", "cup_6011")]
CFG = {"out": A / "umt_verb_noun_breakdown", "buckets": ("pure PE", "mixed", "content")}


def full_attr(b):
    """Per target: % of the target's activation from each bucket (for, against, net) over every L5 latent."""
    import umt_4250_feature_sheet as sheet
    rows = []
    for name, kind, key in TARGETS:
        a = np.asarray(sheet.attribution(dict(sheet.PRESETS[key], attr_run=ROOT / sheet.PRESETS[key]["attr_run"])))
        for k in CFG["buckets"]:
            m = b == k
            rows.append(dict(target=name, type=kind, bucket=k, for_pct=a[m & (a > 0)].sum(), against_pct=a[m & (a < 0)].sum(), net_pct=a[m].sum()))
    return pd.DataFrame(rows)


def top20(b):
    """Per target: the 20 single ablations from its feature sheet (R, spliced, % of target removed), dictionary bucket."""
    rows = []
    for name, kind, key in TARGETS:
        s = pd.read_csv(A / f"feature_sheets/UMT_{key}_feature_analysis.csv")
        col = [c for c in s.columns if c.startswith("% of") and "ablation" in c][0]
        for f, v in zip(s["Feature Idx"].astype(int), s[col]):
            rows.append(dict(target=name, type=kind, feature=f, bucket=b[f], pct_removed=v))
    return pd.DataFrame(rows)


def shares(full, t20):
    """Per target: PE-using (pure PE + mixed) share of the 'for' side, from all latents and from the top-20 ablations."""
    pos = t20[t20.pct_removed > 0]
    out = []
    for name, kind, _ in TARGETS:
        f, p = full[full.target == name], pos[pos.target == name]
        out.append(dict(target=name, type=kind,
                        pe_using_share_all_latents=100 * f[f.bucket != "content"].for_pct.sum() / f.for_pct.sum(),
                        pe_using_share_top20_ablation=100 * p[p.bucket != "content"].pct_removed.sum() / p.pct_removed.sum(),
                        pure_pe_share_top20=100 * p[p.bucket == "pure PE"].pct_removed.sum() / p.pct_removed.sum(),
                        mixed_share_top20=100 * p[p.bucket == "mixed"].pct_removed.sum() / p.pct_removed.sum()))
    return pd.DataFrame(out)


def gaps(sh):
    """Verb mean - noun mean per measure; exact permutation p (share of the 35 3-of-7 'noun' labellings with gap >= observed)."""
    rows, n = [], len(sh)
    for col in [c for c in sh.columns if "share" in c]:
        v = sh[col].to_numpy(); is_noun = (sh.type == "noun").to_numpy()
        g = lambda nn: v[~nn].mean() - v[nn].mean()
        perms = [g(np.isin(np.arange(n), c)) for c in itertools.combinations(range(n), int(is_noun.sum()))]
        rows.append(dict(measure=col, verb_mean=v[~is_noun].mean(), noun_mean=v[is_noun].mean(), gap_pt=g(is_noun),
                         perm_p=np.mean(np.array(perms) >= g(is_noun) - 1e-9), n_splits=len(perms)))
    return pd.DataFrame(rows)


COL = {"pure PE": "#2a78d6", "mixed": "#eb6834", "content": "#1baf7a"}


def plot(full, t20):
    """Left: top-20 single-ablation share by bucket per target. Right: net linear attribution by bucket (all L5 latents)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = [t[0] for t in TARGETS]; pos = t20[t20.pct_removed > 0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8)); w = 0.26
    for i, n in enumerate(names):
        p, bot = pos[pos.target == n], 0
        for k in CFG["buckets"]:
            v = 100 * p[p.bucket == k].pct_removed.sum() / p.pct_removed.sum()
            a1.bar(i, v, bottom=bot, color=COL[k], edgecolor="white", label=k if i == 0 else None)
            a1.text(i, bot + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white"); bot += v
        for j, k in enumerate(CFG["buckets"]):
            a2.bar(i + (j - 1) * w, full[(full.target == n) & (full.bucket == k)].net_pct.iloc[0], w * 0.9, color=COL[k])
    for ax in (a1, a2):
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right", fontsize=8); ax.axvline(3.5, color="k", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    a1.set_ylabel("% of positive single-ablation mass")
    fig.legend(*a1.get_legend_handles_labels(), frameon=False, fontsize=9, ncol=3, loc="upper center")   # clear of the bars
    a1.set_title("Top-20 L5 inputs: share by bucket (verbs | nouns)", fontsize=10)
    a2.axhline(0, color="k", lw=0.6); a2.set_ylabel("net % of target (linear attribution, every L5 latent)")
    a2.set_title("Net contribution by bucket (verbs | nouns)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(CFG["out"] / "verb_noun_by_bucket.png", dpi=150); plt.close(fig)


def main():
    CFG["out"].mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(A / "umt_layer_pe_characterise/features_L5.csv").set_index("latent").cls.map(
        {"pe": "pure PE", "mixed": "mixed", "content": "content", "dead": "dead"}).to_numpy()
    full, t20 = full_attr(b), top20(b)
    sh = shares(full, t20)
    for n, x in (("attr_by_bucket", full), ("top20_ablation", t20), ("shares", sh), ("gaps", gaps(sh))):
        x.round(4).to_csv(CFG["out"] / f"{n}.csv", index=False)
    plot(full, t20)


if __name__ == "__main__":
    main()
