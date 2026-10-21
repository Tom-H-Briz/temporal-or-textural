"""Per-word impact map of ablating one L9 SAE feature, MLM readout (CPU, local, existing outputs).

Each caption word was masked alone (pair word = the contrast run's mask; others with the pair word
visible) and scored before / after the single-feature ablation (umt_mlm_group_ablation, dose 1).
Left: mean change in log p(true word) by word role x class x condition. Right: every caption, each
word coloured by its change (blue = the model loses that word, red = gains).
Only features ablated as singles in that run are available (the 15 group members per pair).
Usage: FEATURE=4250 PAIR=45,43 uv run python src/stage3_analysis/visualisation/umt_mlm_feature_wordmap.py
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/stage3_analysis"))
import umt_mlm_feature_profile as prof_mod

CFG = {
    "feature":  int(os.environ.get("FEATURE", "4250")),
    "pair":     tuple(int(x) for x in os.environ.get("PAIR", "45,43").split(",")),
    "roles":    ["pair word", "verb", "object", "other"],
    "vlim":     1.0,                                                   # colour limit, |Δ log p|
}
# runs holding this pair: ablation (per_clip_edits / _other_words), contrast (mask_pos, word ids)
CFG["ablation"] = prof_mod.find_run(CFG["pair"], prof_mod.CFG["ablation_dirs"], "per_clip_edits.parquet")
CFG["contrast"] = prof_mod.find_run(CFG["pair"], prof_mod.CFG["in_dirs"], "per_clip.parquet")
CFG["out_dir"] = prof_mod.profile_dir(CFG["feature"])


def pair_word_rows(cfg):
    """One row per clip x condition for the masked pair word: position, token, Δ log p(true)."""
    e = pd.read_parquet(cfg["ablation"] / "per_clip_edits.parquet")
    e = e[(e.a == cfg["pair"][0]) & (e.b == cfg["pair"][1]) & (e.edit == f"single_{cfg['feature']}")]
    pc = pd.read_parquet(cfg["contrast"] / "per_clip.parquet").query("condition == 'R'")
    e = e.merge(pc[["clip_id", "mask_pos", "word_a_id", "word_b_id"]], on="clip_id")
    tok = np.where(e.true_class == cfg["pair"][0], e.word_a_id, e.word_b_id)
    return pd.DataFrame(dict(clip_id=e.clip_id, true_class=e.true_class, condition=e.condition,
                             position=e.mask_pos, token=tok, is_obj=False, d=e.d_logp_pair, pair=True))


def word_table(cfg):
    """Every masked word of every clip x condition, with role and Δ log p(true word)."""
    o = pd.read_parquet(cfg["ablation"] / "per_clip_other_words.parquet")
    o = o[(o.a == cfg["pair"][0]) & (o.b == cfg["pair"][1]) & (o.edit == f"single_{cfg['feature']}")]
    o = o.assign(d=o.logp_edit - o.logp_base, pair=False)
    cols = ["clip_id", "true_class", "condition", "position", "token", "is_obj", "d", "pair"]
    w = pd.concat([pair_word_rows(cfg)[cols], o[cols]], ignore_index=True)
    w["clip_id"] = w.clip_id.astype(int)
    # role: the masked pair word; the caption's first word (template verb); object word-pieces; rest
    w["role"] = np.select([w.pair, w.position == 1, w.is_obj], cfg["roles"][:3], cfg["roles"][3])
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    w["word"] = tok.convert_ids_to_tokens(w.token.astype(int).tolist())
    return w.sort_values(["condition", "true_class", "clip_id", "position"])


def panel_roles(ax, w, names, cfg):
    """Heatmap: mean Δ log p(true word), rows = role, cols = class x condition; n in each cell."""
    cols = [(cls, cond) for cond in ("R", "Rev") for cls in cfg["pair"]]
    roles = [r for r in cfg["roles"] if r in set(w.role)]
    g = w.groupby(["role", "true_class", "condition"]).d
    M = np.array([[g.mean().get((r, c, k), np.nan) for c, k in cols] for r in roles])
    N = np.array([[g.size().get((r, c, k), 0) for c, k in cols] for r in roles])
    ax.imshow(M, cmap="RdBu_r", vmin=-cfg["vlim"], vmax=cfg["vlim"], aspect="auto")
    for i in range(len(roles)):
        for j in range(len(cols)):
            ink = "white" if abs(M[i, j]) > 0.6 * cfg["vlim"] else "#0b0b0b"
            ax.text(j, i, f"{M[i, j]:+.2f}\nn={N[i, j]}", ha="center", va="center", fontsize=8, color=ink)
    ax.set_xticks(range(len(cols)), [f"{names[c]} clips\n{k}" for c, k in cols], fontsize=8)
    ax.set_yticks(range(len(roles)), roles, fontsize=9)
    ax.set_title("Mean Δ log p(true word) when ablated", loc="left", fontsize=10)


def panel_captions(ax, w, title, cfg):
    """One caption per line, each word boxed and coloured by its Δ log p; the pair word in bold."""
    cmap, norm = plt.cm.RdBu_r, matplotlib.colors.Normalize(-cfg["vlim"], cfg["vlim"])
    clips = list(dict.fromkeys(w.clip_id))
    for y, cid in enumerate(clips):
        x = 0.0
        ax.text(-0.3, y, str(cid), ha="right", va="center", fontsize=6.5, color="#8a8984")
        for r in w[w.clip_id == cid].itertuples():
            ink = "white" if abs(r.d) > 0.6 * cfg["vlim"] else "#0b0b0b"
            ax.text(x, y, r.word, ha="left", va="center", fontsize=7.5, color=ink,
                    fontweight="bold" if r.pair else "normal",
                    bbox=dict(boxstyle="round,pad=0.25", fc=cmap(norm(r.d)), ec="none"))
            x += 0.24 * len(r.word) + 0.55
    ax.set_xlim(-0.4, 12); ax.set_ylim(len(clips) - 0.5, -0.8)
    ax.axis("off"); ax.set_title(title, loc="left", fontsize=10)


def main():
    """Heatmap + two caption panels: the class the feature acts on (largest |Δ pair word| on R),
    on R; and the other class on Rev, where a direction feature usually switches on."""
    cfg = CFG
    a, b = cfg["pair"]
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    w = word_table(cfg)
    names = {a: w[w.pair & (w.true_class == a)].word.iloc[0], b: w[w.pair & (w.true_class == b)].word.iloc[0]}
    on = w[w.pair & (w.condition == "R")].groupby("true_class").d.mean().abs().idxmax()
    off = b if on == a else a
    n = max(w[w.true_class == c].clip_id.nunique() for c in cfg["pair"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.28 * n + 1.5), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    panel_roles(axes[0], w, names, cfg)
    for ax, (cls, cond) in zip(axes[1:], [(on, "R"), (off, "Rev")]):
        panel_captions(ax, w[(w.true_class == cls) & (w.condition == cond)], f"'{names[cls]}' clips, {cond}", cfg)
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=matplotlib.colors.Normalize(-cfg["vlim"], cfg["vlim"])),
                 ax=axes, shrink=0.4, label="Δ log p(true word)  (blue = lost, red = gained)")
    fig.suptitle(f"UMT L9 #{cfg['feature']} ablated (dose 1): impact on every caption word, MLM, pair {a}/{b}, held-out clips",
                 x=0.01, ha="left", fontsize=11)
    out = cfg["out_dir"] / f"wordmap_{a}_{b}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"saved {out}")
    w.drop(columns="pair").to_csv(cfg["out_dir"] / f"wordmap_{a}_{b}.csv", index=False)


if __name__ == "__main__":
    main()
