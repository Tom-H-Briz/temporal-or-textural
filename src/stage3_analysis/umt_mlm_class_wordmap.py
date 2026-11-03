"""Which caption words does ablating a class's top-DFA L9 features hit? (UMT MLM, refit head, R).

Every caption word-piece is masked alone; ΔCE = -Δ log p(true word) per edit: the top-N group,
each member alone, and one mass-matched control set. Raw-stream ablation x' = x9 - Σ z_f d_f.
Usage: CLASS=126 TOP_N=8 uv run python src/stage3_analysis/umt_mlm_class_wordmap.py
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
from umt_mlm_group_ablation import CFG as GA_CFG, edit_logits, load_validation
from umt_mlm_feature_contrast import clip_loader, setup as contrast_setup
from umt_mlm_refit import encode, mlm_text
from umt_bridge_phase3 import mass_controls, tower_to_layer

CFG = dict(GA_CFG,                                       # layer 9, k64, 7ep, paths, device
           cls=int(os.environ.get("CLASS", 126)),
           top_n=int(os.environ.get("TOP_N", 8)),
           dfa=ROOT / "outputs/analysis/dfa_mass_delta_umt/dfa_mass_delta_umt_ssv2_all174_thrfix_l9_job7ep_k64.parquet",
           num_workers=0, splice=False)
CFG["out_dir"] = ROOT / f"outputs/analysis/umt_mlm_class_wordmap/c{CFG['cls']}"


def class_clips(cfg):
    """The class's clips in the all-174 DFA run, and its top-N features by |mean signed DFA| on R."""
    d = pd.read_parquet(cfg["dfa"])
    d = d[d.class_id == cfg["cls"]].reset_index(drop=True)
    m = np.stack(d.signed_vec_R.values).mean(0)
    top = np.argsort(-np.abs(m))[: cfg["top_n"]]
    rows = pd.DataFrame(dict(clip_id=d.clip_id.astype(str)))   # loader: row index = label
    return rows, [int(f) for f in top], m


def all_word_variants(cfg, s, label, ph):
    """One row per word-piece masked alone (position 1.. ; 0 is [CLS]) -> dict for edit_logits,
    plus the true token, word text and role (verb = first word, object, template)."""
    enc, _ = encode(s["tok"], s["pre_text"](label), 25)
    obj = set(s["tok"](s["pre_text"](" ".join(ph)), add_special_tokens=False)["input_ids"])
    pos = list(range(1, len(enc)))
    mask = s["tok"].mask_token_id
    ids = torch.tensor([[*enc[:p], mask, *enc[p + 1:]] for p in pos], device=cfg["device"])
    role = ["verb" if p == 1 else "object" if enc[p] in obj else "template" for p in pos]
    return dict(ids=ids, attn=torch.ones_like(ids), pos=pos, true=[enc[p] for p in pos],
                word=s["tok"].convert_ids_to_tokens([enc[p] for p in pos]), role=role)


def encode_clips(cfg, s, rows):
    """R pass per clip -> {row index: (x9, z0)} (fp32, CPU-sized: ~30 clips)."""
    out = {}
    for pv, y in clip_loader(rows, cfg, "R"):
        i = int(y)
        if i in out:                                     # decode-retry skipped to a later clip
            continue
        x9 = tower_to_layer(s["model"], pv.to(cfg["device"]), layer=cfg["layer"])
        with torch.no_grad():
            _, z0 = s["sae"].encode((x9[0] - s["dim_mean"]).float())
        out[i] = (x9, z0)
    return out


def build_edits(top, enc):
    """[(name, feats)]: top-N group, mass-matched control set (Phase 3's mass_controls; pool =
    every active feature outside the top N), and each member alone."""
    mass = np.mean([z0.sum(0).cpu().numpy() for _, z0 in enc.values()], axis=0)
    pool = set(np.where(mass > 0)[0].tolist()) - set(top)
    ctrl, gap = mass_controls(mass, np.array(top), None, None, pool=pool)
    print(f"control {ctrl.tolist()}, mass gaps max {gap.max():.2f}")
    return ([("top", top), ("control", [int(c) for c in ctrl])]
            + [(f"single_{f}", [f]) for f in top])


def clip_rows(cfg, s, cid, cap, x9, z0, D, edits):
    """Per word x edit: log p(true word) baseline vs edited -> list of dicts."""
    h8, add = mlm_text(cap["ids"], cap["attn"], s["w"])
    tr = torch.tensor(cap["true"], device=cfg["device"])[:, None]
    def lp(feats):
        return edit_logits(s, h8, add, cap, x9, z0, D, feats, 1.0).log_softmax(-1).gather(1, tr)[:, 0]
    base, out = lp([]), []
    for name, feats in edits:
        d = (lp(feats) - base).cpu().numpy()
        out += [dict(clip_id=cid, edit=name, position=p, word=w, role=r,
                     lp_base=float(b), dce=float(-x))
                for p, w, r, b, x in zip(cap["pos"], cap["word"], cap["role"], base.cpu().numpy(), d)]
    return out


def main():
    cfg, t0 = CFG, time.time()
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    rows, top, _ = class_clips(cfg)
    print(f"class {cfg['cls']}: {len(rows)} clips, top {cfg['top_n']} {top}, device {cfg['device']}")
    s = dict(contrast_setup(cfg), splice=False)
    enc, val, D = encode_clips(cfg, s, rows), load_validation(cfg), s["sae"].get_dictionary()
    edits, out = build_edits(top, enc), []
    for i, (x9, z0) in sorted(enc.items()):
        cid = rows.clip_id[i]
        out += clip_rows(cfg, s, cid, all_word_variants(cfg, s, *val[cid]), x9, z0, D, edits)
        print(f"  {cid} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(out)
    df.to_parquet(cfg["out_dir"] / "per_word.parquet", index=False)
    print(df.groupby(["edit", "role"]).dce.mean().unstack().round(3).to_string())
    tpl = df[df.role != "object"].groupby(["edit", "word"]).dce.mean().unstack().round(3)
    print(tpl.to_string())
    print(f"done {(time.time() - t0) / 60:.1f} min -> {cfg['out_dir']}")


def plot(cfg, edit):
    """Caption word map for one stored edit (MODE=plot EDIT=single_5340), drawn by the pair-run
    word map's panel_captions; red = the 95th pct of damage in this figure."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "src/stage3_analysis/visualisation"))
    from umt_mlm_feature_wordmap import CFG as WM, panel_captions
    w = pd.read_parquet(cfg["out_dir"] / "per_word.parquet").query("edit == @edit")
    w = w.assign(d=-w.dce, pair=w.role == "verb").sort_values(["clip_id", "position"])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("damage", WM["colours"])
    norm = matplotlib.colors.Normalize(0, max(np.percentile(w.dce.clip(lower=0), 95), 1e-6))
    n = w.clip_id.nunique()
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * n + 1.2), dpi=130)
    ax.set_xlim(0, 1); ax.set_ylim(n - 0.5, -0.6)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.5, pad=0.01,
                 label="damage = −Δ log p(true word) when ablated")
    panel_captions(fig, ax, w, cmap, norm)
    fig.suptitle(f"{edit} ablated (raw-stream): damage to each masked word, class {cfg['cls']}, R, n={n}",
                 x=0.02, ha="left", fontsize=10)
    out = cfg["out_dir"] / f"wordmap_{edit}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"saved {out}")


if __name__ == "__main__":
    if os.environ.get("MODE") == "plot":
        plot(CFG, os.environ["EDIT"])
    else:
        main()
