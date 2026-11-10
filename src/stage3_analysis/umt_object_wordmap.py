"""UMT object word maps (03/11/26): which caption words does ablating a layer's object features
(umt_object_contrast_L{LAYER} picks, singles + groups; controls dropped 10/11) damage? Every
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
from umt_object_contrast import CFG as OC
L, TASK = int(os.environ.get("LAYER", 9)), os.environ.get("TASK", "object")
OUT = ROOT / f"outputs/analysis/umt_{TASK}_wordmap_L{L}"


def run(max_clips):
    """R pass per test clip -> every word masked alone x every edit -> per_word.parquet."""
    from umt_bridge_phase3 import tower_to_layer
    from umt_mlm_class_wordmap import all_word_variants, clip_rows
    from umt_mlm_feature_contrast import clip_loader, setup
    from umt_mlm_group_ablation import load_validation
    cfg = oa.contrast_cfg(dict(oa.CFG, splice=True, max_clips=0))   # real picks; MAX_CLIPS caps clips only
    assert cfg["layer"] == L
    s = dict(setup(cfg), splice=True)
    edits, val, D = oa.edit_list(cfg), load_validation(cfg), s["sae"].get_dictionary()
    if TASK == "pair":                                # pairs: groups only, ~10 units per layer
        grouped = {n[:-len("_group")] for n, _ in edits if n.endswith("_group") and not n.startswith("ctrl_")}
        keep = lambda n: n.endswith("_group") or unit_of(n) not in grouped
        edits = [(n, f) for n, f in edits if n == "baseline" or keep(n)]
    edits = [(n, f) for n, f in edits if not n.startswith("ctrl_")]   # 10/11: controls dropped (raw edit - baseline)
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


if TASK == "pair":                                    # unit = pair_home (45_43_43); words = the pair's two
    NOUNS = {f"{p}_{h}": set(ws) for p, ws in OC["pairs"].items() for h in p.split("_")}
    ROLES = ["own pair word", "other pair word", "object words", "other verb", "template"]
else:
    NOUNS = {o: {o, o + "s", o + "es"} for o in OC["objects"]}
    ROLES = ["own noun", "other noun", "other object words", "verb", "template"]
unit_of = lambda n: "_".join(n.replace("ctrl_", "").split("_")[:3 if TASK == "pair" else 1])
ROLE_NAMES = {"own pair word": "its own key word (e.g. down)", "other pair word": "the partner's key word (e.g. up)",
              "other verb": "'moving' / 'turning'", "own noun": "the object's name", "other noun": "other objects' names",
              "template": "other template words"}   # figure labels only; the tables keep ROLES
TPL = {int(v): k for k, v in json.load(open(ROOT / "data/ssv2/labels/labels.json")).items()}


def name(o):
    """Plain name for figures: 45_43_43 -> 'Moving something down' / 'vs Moving something up'; objects as is."""
    if TASK != "pair":
        return f"'{o}' features", ""
    a, b, h = (int(x) for x in o.split("_"))
    return TPL[h], f"vs {TPL[b if h == a else a]}"


def edit_name(e):
    """'box_group' -> 'feature group'; '93_86_93_3844' -> 'one feature (#3844)'; None -> 'no features passed'."""
    return "no features passed" if e is None else "feature group" if e.endswith("_group") else f"one feature (#{e.split('_')[-1]})"


def with_roles(w):
    """Edit's unit (box_*, ctrl_box_* -> box; 45_43_43_* -> 45_43_43) and each word's role relative to it."""
    w = w[w.edit != "baseline"].copy()
    w["edit_obj"], w["is_ctrl"] = w.edit.map(unit_of), w.edit.str.startswith("ctrl_")
    own = [wd in NOUNS[o] for wd, o in zip(w.word, w.edit_obj)]
    other = [any(wd in v for k, v in NOUNS.items() if v != NOUNS[o]) for wd, o in zip(w.word, w.edit_obj)]
    w["wrole"] = np.select([own, other, w.role == "object", w.role == "verb"], ROLES[:4], ROLES[4])
    return w


def main_edit(w, o):
    """An object's main edit: its group if it has one, else its single feature."""
    names = set(w.edit[~w.is_ctrl & (w.edit_obj == o)])
    if not names:                                     # no features passed for this unit at this layer
        return None
    return f"{o}_group" if f"{o}_group" in names else sorted(names)[0]


def role_table(w):
    """Damage (mean -d log p(true word)) by role for each object's main edit (raw, no control)."""
    rows = []
    for o in NOUNS:
        e = main_edit(w, o)
        if e is None:
            continue
        m = w[w.edit == e].groupby("wrole").dce.agg(["mean", "size"])
        rows += [dict(object=o, edit=e, role=r, damage=m.loc[r, "mean"], n_words=int(m.loc[r, "size"]))
                 for r in ROLES if r in m.index]
    return pd.DataFrame(rows)


def plot_roles(tabs, out):
    """One panel per layer: object (main edit) x word role, raw damage, shared scale."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from umt_mlm_feature_wordmap import CFG as WM
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("damage", WM["colours"])
    vmax = max(t.damage.max() for t in tabs.values())
    fig, axes = plt.subplots(1, len(tabs), figsize=(4.2 * len(tabs), 1.2 + 0.45 * len(NOUNS)), dpi=130, squeeze=False)
    for ax, (layer, t) in zip(axes[0], tabs.items()):
        M = t.pivot(index="object", columns="role", values="damage").reindex(index=list(NOUNS), columns=ROLES)
        ax.imshow(M.clip(lower=0).values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(M.values):
            ax.text(j, i, "" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(ROLES)), [ROLE_NAMES.get(r, r).replace(" ", "\n") for r in ROLES], fontsize=7)
        first = ax is axes[0][0]                      # class names once; blank row = no features passed at that layer
        ax.set_yticks(range(len(NOUNS)), ["\n".join(filter(None, name(o))) if first else "" for o in NOUNS], fontsize=7)
        ax.set_title(f"Layer {layer}", fontsize=10)
    fig.suptitle(f"Removing each {'verb' if TASK == 'pair' else 'object'}'s features: how much harder each kind of caption word "
                 "becomes to predict (masked, real video; raw, no control; 0 = no effect)", x=0.02, y=1.06, ha="left", fontsize=10)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); print(f"saved {out}")


def plot_captions(w, o, layer, out, n=30, seed=0):
    """n test captions naming o, each word coloured by the main edit's damage (own noun in bold)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from umt_mlm_feature_wordmap import CFG as WM, panel_captions
    e = main_edit(w, o)
    if e is None:
        return print(f"L{layer} {o}: no features, skipped")
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
                 label="damage: drop in log-probability of the word")
    panel_captions(fig, ax, g, cmap, norm)
    what = f"'{name(o)[0]}' features ({name(o)[1]})" if TASK == "pair" else name(o)[0]
    fig.suptitle(f"Layer {layer}: {what} removed, {edit_name(e)}.\n{len(ids)} test captions; darker = that word became harder "
                 "to predict when masked (key word in bold)", x=0.02, ha="left", fontsize=9)
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
    tabs, fig_dir = {}, ROOT / f"outputs/analysis/umt_{TASK}_wordmap{sfx}"
    for layer in [int(x) for x in os.environ.get("LAYERS", "5,7,9").split(",")]:
        d = ROOT / f"outputs/analysis/umt_{TASK}_wordmap_L{layer}{sfx}"
        w = with_roles(pd.read_parquet(d / "per_word.parquet"))
        tabs[layer] = role_table(w)
        tabs[layer].to_csv(d / "role_table.csv", index=False)
        for o in NOUNS:
            plot_captions(w, o, layer, d / f"wordmap_{o}.png")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_roles(tabs, fig_dir / "roles_by_layer.png")


if __name__ == "__main__":
    main()
