"""UMT object word maps (03/11/26): which caption words does ablating a layer's object features
(umt_object_contrast_L{LAYER} picks, singles + groups + mass-matched controls) damage? Every
word-piece of each held-out test caption is masked alone (umt_mlm_class_wordmap convention),
scored before / after each spliced edit, real video -> per_word.parquet.
Roles are relative to the ablated object (plot time): own noun, other object noun, other object
words, verb, template words.
Modes: run (GPU) | plot (CPU). Usage: LAYER=7 MODE=run uv run python src/stage3_analysis/umt_object_wordmap.py
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
import umt_object_ablation as oa
L = int(os.environ.get("LAYER", 9))
OUT = ROOT / f"outputs/analysis/umt_object_wordmap_L{L}"


def run(max_clips):
    """R pass per test clip -> every word masked alone x every edit -> per_word.parquet."""
    from umt_bridge_phase3 import tower_to_layer
    from umt_mlm_class_wordmap import all_word_variants, clip_rows
    from umt_mlm_feature_contrast import clip_loader, setup
    from umt_mlm_group_ablation import load_validation
    cfg = oa.contrast_cfg(dict(oa.CFG, splice=True))            # real picks; max_clips caps clips only
    assert cfg["layer"] == L
    s = dict(setup(cfg), splice=True)
    edits, val, D = oa.edit_list(cfg), load_validation(cfg), s["sae"].get_dictionary()
    clips = pd.DataFrame({"clip_id": sorted({c for c, _ in cfg["test_rows"]})})
    clips = clips.head(max_clips) if max_clips else clips
    out, seen, t0 = [], set(), time.time()
    for pv, y in clip_loader(clips, cfg, "R"):
        c = clips.clip_id.iloc[int(y)]
        if c in seen:                                 # decode-retry duplicate
            continue
        seen.add(c)
        xL = tower_to_layer(s["model"], pv.to(cfg["device"]), layer=L)
        with torch.no_grad():
            z0 = s["sae"].encode((xL[0] - s["dim_mean"]).float())[1]
        out += clip_rows(cfg, s, c, all_word_variants(cfg, s, *val[c]), xL, z0, D, edits)
    print(f"L{L}: {len(seen)}/{len(clips)} clips, {len(edits)} edits, {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.DataFrame(out)


NOUNS = {"box": {"box", "boxes"}, "pen": {"pen", "pens"}, "cup": {"cup", "cups"}}


def with_roles(w):
    """Edit's object (box_*, ctrl_box_* -> box) and each word's role relative to it."""
    w = w[w.edit != "baseline"].copy()
    w["edit_obj"] = w.edit.str.replace("ctrl_", "").str.split("_").str[0]
    w["is_ctrl"] = w.edit.str.startswith("ctrl_")
    own = [wd in NOUNS[o] for wd, o in zip(w.word, w.edit_obj)]
    other = [any(wd in v for k, v in NOUNS.items() if k != o) for wd, o in zip(w.word, w.edit_obj)]
    w["wrole"] = np.select([own, other, w.role == "object", w.role == "verb"],
                           ["own noun", "other noun", "other object words", "verb"], "template")
    return w
ROLES = ["own noun", "other noun", "other object words", "verb", "template"]


def main_edit(w, o):
    """An object's main edit: its group if it has one, else its single feature."""
    names = set(w.edit[~w.is_ctrl & (w.edit_obj == o)])
    return f"{o}_group" if f"{o}_group" in names else sorted(names)[0]


def role_table(w):
    """Damage (mean -d log p(true word)) by role for each object's main edit minus its control."""
    rows = []
    for o in NOUNS:
        e = main_edit(w, o)
        g = w[w.edit.isin([e, f"ctrl_{e}"])]
        m = g.pivot_table(index="wrole", columns="is_ctrl", values="dce", aggfunc="mean")
        n = g[~g.is_ctrl].groupby("wrole").size()
        for r in ROLES:
            if r in m.index:
                rows.append(dict(object=o, edit=e, role=r, damage=m.loc[r, False], ctrl=m.loc[r, True],
                                 minus_ctrl=m.loc[r, False] - m.loc[r, True], n_words=int(n.get(r, 0))))
    return pd.DataFrame(rows)


def plot_roles(tabs, out):
    """One panel per layer: object (main edit) x word role, damage minus control, shared scale."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from umt_mlm_feature_wordmap import CFG as WM
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("damage", WM["colours"])
    vmax = max(t.minus_ctrl.max() for t in tabs.values())
    fig, axes = plt.subplots(1, len(tabs), figsize=(4.2 * len(tabs), 2.6), dpi=130, squeeze=False)
    for ax, (layer, t) in zip(axes[0], tabs.items()):
        M = t.pivot(index="object", columns="role", values="minus_ctrl").reindex(index=list(NOUNS), columns=ROLES)
        ax.imshow(M.clip(lower=0).values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(M.values):
            ax.text(j, i, "" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(ROLES)), [r.replace(" ", "\n") for r in ROLES], fontsize=7)
        ax.set_yticks(range(len(NOUNS)), [f"{o}\n{t[t.object == o].edit.iloc[0]}" for o in NOUNS], fontsize=7)
        ax.set_title(f"L{layer}", fontsize=10)
    fig.suptitle("Object-feature ablation (spliced, R): damage to each masked word by role, minus control",
                 x=0.02, ha="left", fontsize=10)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); print(f"saved {out}")


def plot_captions(w, o, layer, out, n=30, seed=0):
    """n test captions naming o, each word coloured by the main edit's damage (own noun in bold)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from umt_mlm_feature_wordmap import CFG as WM, panel_captions
    e = main_edit(w, o)
    has = w.groupby("clip_id").word.apply(lambda x: bool(set(x) & NOUNS[o]))
    if not has.any():                                 # no caption names o (small smoke sets)
        return print(f"L{layer} {o}: no captions, skipped")
    ids = pd.Series(has[has].index).sample(min(n, int(has.sum())), random_state=seed)
    g = w[(w.edit == e) & w.clip_id.isin(ids)].sort_values(["clip_id", "position"])
    g = g.assign(d=-g.dce, pair=g.word.isin(NOUNS[o]))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("damage", WM["colours"])
    norm = matplotlib.colors.Normalize(0, max(np.percentile(g.dce.clip(lower=0), 95), 1e-6))
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(ids) + 1.2), dpi=130)
    ax.set_xlim(0, 1); ax.set_ylim(len(ids) - 0.5, -0.6)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.5, pad=0.01,
                 label="damage = −Δ log p(true word) when ablated")
    panel_captions(fig, ax, g, cmap, norm)
    fig.suptitle(f"L{layer} {e} ablated (SAE-spliced): damage to each masked word, {len(ids)} test captions naming '{o}', R",
                 x=0.02, ha="left", fontsize=9)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); print(f"saved {out}")


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 paths, as the references
    torch.backends.cudnn.allow_tf32 = False
    mode, mc = os.environ.get("MODE", "run"), int(os.environ.get("MAX_CLIPS", 0))
    sfx = "_smoke" if mc else ""
    if mode == "run":
        out = OUT.with_name(OUT.name + sfx); out.mkdir(parents=True, exist_ok=True)
        run(mc).to_parquet(out / "per_word.parquet", index=False)
        return
    if mode != "plot":
        sys.exit(f"MODE={mode}: use run | plot")
    sys.path.insert(0, str(ROOT / "src/stage3_analysis/visualisation"))
    tabs, fig_dir = {}, ROOT / f"outputs/analysis/umt_object_wordmap{sfx}"
    for layer in [int(x) for x in os.environ.get("LAYERS", "5,7,9").split(",")]:
        d = ROOT / f"outputs/analysis/umt_object_wordmap_L{layer}{sfx}"
        w = with_roles(pd.read_parquet(d / "per_word.parquet"))
        tabs[layer] = role_table(w)
        tabs[layer].to_csv(d / "role_table.csv", index=False)
        for o in NOUNS:
            plot_captions(w, o, layer, d / f"wordmap_{o}.png")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_roles(tabs, fig_dir / "roles_by_layer.png")


if __name__ == "__main__":
    main()
