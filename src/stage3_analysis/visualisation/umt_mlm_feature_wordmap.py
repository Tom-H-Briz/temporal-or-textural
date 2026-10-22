"""Per-word impact map of ablating one L9 SAE feature, MLM readout (CPU, local, existing outputs).

Each caption word was masked alone (pair word = the contrast run's mask; others with the pair word
visible) and scored before / after the single-feature ablation (umt_mlm_group_ablation, dose 1).
Figure: every caption of one class, each word coloured by its damage = -Δ log p(true word) when that
word is masked (white = none, yellow, red = most; red = the 95th pct in the figure; gains are white).
Class: CLASS (default = the class whose pair word the ablation hurts most); condition: COND (default R).
Only features ablated as singles in that run are available (the 15 group members per pair).
Per-word values for both classes and conditions: wordmap_<a>_<b>.csv.
MODE=template: one averaged row per class (real clips, COND=R; COND=R+Rev adds reversed rows), the SSv2 template with "something" = object words.
Usage: FEATURE=4250 PAIR=45,43 [CLASS=43] [COND=R] [MODE=template] uv run python src/stage3_analysis/visualisation/umt_mlm_feature_wordmap.py
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
    "cls":      os.environ.get("CLASS", ""),                           # "" = auto
    "cond":     os.environ.get("COND", "R"),
    "mode":     os.environ.get("MODE", "captions"),                    # captions | template | compare
    "edit":     os.environ.get("EDIT", ""),                            # stored edit; "" = single_<FEATURE>
    "rows":     os.environ.get("ROWS", ""),                            # compare: edit list; "" = default set
    "vmax":     float(os.environ.get("VMAX", 0)),                      # template/compare: fixed red level (0 = own max)
    "labels":   ROOT / "data/ssv2/labels/labels.json",
    "roles":    ["pair word", "verb", "object", "other"],              # kept in the CSV
    "colours":  ["#ffffff", "#fdd835", "#d7301f"],                     # damage: none -> mid -> most
    "red_pct":  95,                                                    # damage at this pct = full red
}
# runs holding this pair: ablation (per_clip_edits / _other_words), contrast (mask_pos, word ids)
CFG["ablation"] = prof_mod.find_run(CFG["pair"], prof_mod.CFG["ablation_dirs"], "per_clip_edits.parquet")
CFG["contrast"] = prof_mod.find_run(CFG["pair"], prof_mod.CFG["in_dirs"], "per_clip.parquet")
CFG["out_dir"] = prof_mod.profile_dir(CFG["feature"])
ABL = "raw-stream" if os.environ.get("ABLATION") == "raw" else "SAE-spliced"   # for titles


def pair_word_rows(cfg):
    """One row per clip x condition for the masked pair word: position, token, Δ log p(true)."""
    e = pd.read_parquet(cfg["ablation"] / "per_clip_edits.parquet")
    e = e[(e.a == cfg["pair"][0]) & (e.b == cfg["pair"][1]) & (e.edit == (cfg["edit"] or f"single_{cfg['feature']}"))]
    pc = pd.read_parquet(cfg["contrast"] / "per_clip.parquet").query("condition == 'R'")
    e = e.merge(pc[["clip_id", "mask_pos", "word_a_id", "word_b_id"]], on="clip_id")
    tok = np.where(e.true_class == cfg["pair"][0], e.word_a_id, e.word_b_id)
    return pd.DataFrame(dict(clip_id=e.clip_id, true_class=e.true_class, condition=e.condition,
                             position=e.mask_pos, token=tok, is_obj=False, d=e.d_logp_pair, pair=True))


def word_table(cfg):
    """Every masked word of every clip x condition, with role and Δ log p(true word)."""
    o = pd.read_parquet(cfg["ablation"] / "per_clip_other_words.parquet")
    o = o[(o.a == cfg["pair"][0]) & (o.b == cfg["pair"][1]) & (o.edit == (cfg["edit"] or f"single_{cfg['feature']}"))]
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


def clip_slots(c):
    """One clip's words -> {template key: damage}. Object runs -> ('slot', k) by order, mean over their
    word-pieces; every other word keyed by its text (template words are fixed across captions)."""
    c = c.sort_values("position")
    run = (c.is_obj & ~c.is_obj.shift(fill_value=False)).cumsum()        # object run index, 1..k
    out = {("slot", k): -g.d.mean() for k, g in c[c.is_obj].groupby(run[c.is_obj])}
    out.update({("word", r.word): -r.d for r in c[~c.is_obj].itertuples()})
    return out


def margin_note(cfg, cls, cond):
    """Pair word vs its partner: logit difference = log p(true) - log p(partner) (normaliser cancels).
    -> 'margin -X%, k/n flip': pooled fraction removed, clips correct at baseline that flip."""
    e = pd.read_parquet(cfg["ablation"] / "per_clip_edits.parquet")
    e = e[(e.a == cfg["pair"][0]) & (e.b == cfg["pair"][1]) & (e.true_class == cls) & (e.condition == cond)
          & (e.edit == (cfg["edit"] or f"single_{cfg['feature']}"))]
    sgn = 1 if cls == cfg["pair"][0] else -1                    # true minus partner
    base, edit = sgn * (e.logp_a_base - e.logp_b_base), sgn * (e.logp_a - e.logp_b)
    if base.mean() <= 0:                                        # baseline wrong on average (e.g. Rev): % is meaningless
        return f"Δmargin {edit.mean() - base.mean():+.2f}, {int(((base <= 0) & (edit > 0)).sum())}/{len(e)} fixed"
    frac = (base.mean() - edit.mean()) / base.mean()
    return f"margin {-frac:+.0%}, {int(((base > 0) & (edit <= 0)).sum())}/{len(e)} flip"


def template_row(s, template, tok, note=None):
    """Template tokens with the mean (over clips) damage of each: [(text, mean damage, n, note)];
    `note` (e.g. margin_note) is attached to the pair word."""
    per_clip = pd.DataFrame([clip_slots(c) for _, c in s.groupby("clip_id")])
    pair_word = s[s.pair].word.iloc[0] if s.pair.any() else None
    row, k = [], 0
    for t in tok.tokenize(template.lower()):
        k += t == "something"
        key = ("slot", k) if t == "something" else ("word", t)
        vals = per_clip[key].dropna() if key in per_clip else pd.Series(dtype=float)
        row.append((t, vals.mean() if len(vals) else np.nan, len(vals), note if t == pair_word else None))
    return row


def panel_template(fig, ax, rows, cmap, norm):
    """rows: [(label, [(text, damage, n, note)])]; one template per line, boxes coloured by mean damage
    (gains white), mean value printed under each word. Monospace layout as panel_captions."""
    fs, gap = 11, 1.0
    char = 0.6 * fs / 72 / (ax.get_position().width * fig.get_figwidth())
    for y, (label, row) in enumerate(rows):
        ax.text(-0.01, y, label, ha="right", va="center", fontsize=9, color="#52514e")
        x = 0.0
        for text, dmg, n, note in row:
            if text == "|":                                         # gap between two templates on one row
                x += 3 * gap * char
                continue
            wid = max(len(text), 5) * char
            fc = "#eeeeee" if np.isnan(dmg) else cmap(norm(max(dmg, 0)))
            ax.add_patch(plt.Rectangle((x, y - 0.3), wid, 0.42, fc=fc, ec="none"))
            ax.text(x + wid / 2, y - 0.09, text, ha="center", va="center", fontsize=fs, family="monospace")
            ax.text(x + wid / 2, y + 0.27, "—" if np.isnan(dmg) else f"ΔCE {dmg:+.2f}", ha="center",
                    va="center", fontsize=8, color="#52514e")
            if note:                                                # pair word: margin to partner, flips
                ax.text(x + wid / 2, y + 0.52, note, ha="center", va="center", fontsize=8, color="#0b0b0b")
            x += wid + gap * char
    ax.axis("off")


def panel_captions(fig, ax, w, cmap, norm):
    """One caption per line; each word boxed and coloured by damage; boxes placed from the rendered
    width of the word (monospace) so they never overlap. Word-pieces (##) join their word."""
    # monospace: each character is 0.6 x fontsize wide, so word boxes are placed exactly (no measuring)
    fs, gap = 9, 0.6                                               # gap in characters
    char = 0.6 * fs / 72 / (ax.get_position().width * fig.get_figwidth())   # one char, axes fraction
    for y, cid in enumerate(dict.fromkeys(w.clip_id)):
        ax.text(-0.01, y, str(cid), ha="right", va="center", fontsize=7, color="#8a8984")
        x = 0.0
        for r in w[w.clip_id == cid].itertuples():
            piece = r.word.startswith("##")
            word = r.word[2:] if piece else r.word
            x -= gap * char if piece else 0                         # ## pieces butt onto their word
            ax.add_patch(plt.Rectangle((x, y - 0.38), len(word) * char, 0.76,
                                       fc=cmap(norm(max(-r.d, 0))), ec="none"))
            ax.text(x, y, word, ha="left", va="center", fontsize=fs, family="monospace",
                    color="#0b0b0b", fontweight="bold" if r.pair else "normal")
            x += (len(word) + gap) * char
    ax.axis("off")


def compare_rows(cfg):
    """Default compare set: every single in the pair's flip / no-change groups, then the three
    groups of 5 and their mass-matched control sets. -> [(edit, label)]"""
    if cfg["rows"]:
        return [(e, e) for e in cfg["rows"].split(",")]
    g = pd.read_csv(cfg["ablation"] / "groups.csv")
    g = g[(g.a == cfg["pair"][0]) & (g.b == cfg["pair"][1])]
    rows = [(f"single_{f}", f"#{f} ({grp})") for grp in ("flip", "nochange")
            for f in g[g.group == grp].feature]
    rows += [(e, f"{e} group of 5") for e in ("flip", "nochange", "gated")]
    return rows + [(f"ctrl_{e}", f"control for {e}") for e in ("flip", "nochange", "gated")]


def main_compare(cfg):
    """One row per edit: template of class A | template of class B (R), shared colour scale."""
    import json
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    names = {int(v): k for k, v in json.load(open(cfg["labels"])).items()}
    rows = []
    for edit, label in compare_rows(cfg):
        w = word_table(dict(cfg, edit=edit))
        w = w[w.condition == "R"]
        e_cfg = dict(cfg, edit=edit)
        a, b = (template_row(w[w.true_class == c], names[c], tok, margin_note(e_cfg, c, "R")) for c in cfg["pair"])
        rows.append((label, a + [("|", np.nan, 0, None)] + b))
    return rows


def draw_rows(cfg, rows, width):
    """Figure with one template line per row, shared white-yellow-red scale (max = largest mean damage)."""
    top = cfg["vmax"] or np.nanmax([t[1] for _, r in rows for t in r] + [1e-6])   # VMAX: shared scale
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("damage", cfg["colours"])
    norm = matplotlib.colors.Normalize(0, top)
    fig, ax = plt.subplots(figsize=(width, 1.0 * len(rows) + 1.2), dpi=130)
    fig.subplots_adjust(top=1 - 0.5 / (1.0 * len(rows) + 1.2), bottom=0.02)   # no dead band above row 1
    ax.set_xlim(0, 1); ax.set_ylim(len(rows) - 0.2, -0.6)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.5, pad=0.01,
                 label="ΔCE (loss increase)")
    panel_template(fig, ax, rows, cmap, norm)
    return fig


def main_template(cfg, w):
    """Averaged template map: one row per class x condition (R, Rev), shared colour scale."""
    import json
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    names = {int(v): k for k, v in json.load(open(cfg["labels"])).items()}
    rows = []
    for cond in cfg["cond"].split("+"):                  # R only by default; COND=R+Rev adds reversed
        for cls in cfg["pair"]:
            s = w[(w.true_class == cls) & (w.condition == cond)]
            rows.append((f"{cond}, n={s.clip_id.nunique()}", template_row(s, names[cls], tok, margin_note(cfg, cls, cond))))
    top = cfg["vmax"] or np.nanmax([t[1] for _, r in rows for t in r] + [1e-6])   # VMAX: shared scale
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("damage", cfg["colours"])
    norm = matplotlib.colors.Normalize(0, top)
    fig, ax = plt.subplots(figsize=(5.5, 1.0 * len(rows) + 1.2), dpi=130)
    ax.set_xlim(0, 1); ax.set_ylim(len(rows) - 0.2, -0.6)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.7, pad=0.01,
                 label="ΔCE (loss increase)")
    panel_template(fig, ax, rows, cmap, norm)
    return fig


def main():
    """One panel: captions of CLASS (default: the class whose pair word the ablation hurts most) under COND."""
    cfg = CFG
    a, b = cfg["pair"]
    if cfg["mode"] == "compare":                    # many edits of one pair -> outputs/.../compare/
        out = prof_mod.profile_dir("compare").parent / "compare" / f"wordmap_compare_{a}_{b}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = draw_rows(cfg, main_compare(cfg), width=9)
        fig.suptitle(f"Ablation damage ({ABL}) per template word, pair {a}/{b}, R, held-out clips "
                     f"('something' = object words, averaged per clip)", x=0.02, ha="left", fontsize=10)
        fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"saved {out}")
        return
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    w = word_table(cfg)
    w.drop(columns="pair").to_csv(cfg["out_dir"] / f"wordmap_{a}_{b}.csv", index=False)
    if cfg["mode"] == "template":
        fig = main_template(cfg, w)
        fig.suptitle(f"#{cfg['feature']} ablated ({ABL}): mean damage per template word, held-out {a}/{b} clips "
                     f"('something' = the object words, averaged per clip)", x=0.02, ha="left", fontsize=10)
        out = cfg["out_dir"] / f"wordmap_template_{a}_{b}_{cfg['cond'].replace('+', '_')}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"saved {out}")
        return
    cls = int(cfg["cls"] or w[w.pair & (w.condition == "R")].groupby("true_class").d.mean().idxmin())
    s = w[(w.true_class == cls) & (w.condition == cfg["cond"])]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("damage", cfg["colours"])
    norm = matplotlib.colors.Normalize(0, max(np.percentile((-s.d).clip(lower=0), cfg["red_pct"]), 1e-6))
    n = s.clip_id.nunique()
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * n + 1.2), dpi=130)        # measure text at save dpi
    ax.set_xlim(0, 1); ax.set_ylim(n - 0.5, -0.6)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.5, pad=0.01,
                 label="damage = −Δ log p(true word) when ablated")
    panel_captions(fig, ax, s, cmap, norm)          # after the colorbar: it resizes ax, which sets char width
    word = s[s.pair].word.iloc[0]
    fig.suptitle(f"#{cfg['feature']} ablated ({ABL}): damage to each masked word, '{word}' clips ({cls}), {cfg['cond']}, "
                 f"held-out, n={n}", x=0.02, ha="left", fontsize=10)
    out = cfg["out_dir"] / f"wordmap_{a}_{b}_{word}_{cfg['cond']}.png"     # class word + condition: no overwrites
    fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"saved {out}")


if __name__ == "__main__":
    main()
