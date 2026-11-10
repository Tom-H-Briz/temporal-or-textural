"""Noun / verb scale-up workbook (10/11/26, CPU, local): 9 nouns, 16 opposite pairs, L5/7/9, UMT + VideoMAE.
Copy of object_verb_meeting_tables.py's layout (its cell / table / figure helpers are imported, that file is
untouched). Every number is RAW edit - baseline: the controls are broken in this run (pending a fix).
Needs notebooks/umt_peak_certainty.py run first (Layers sheet). Findings: findings/noun_verb_scaleup_findings_091126.md.
Usage: uv run python notebooks/noun_verb_scaleup_tables.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "notebooks"), str(ROOT / "src" / "stage3_analysis")]
from object_verb_meeting_tables import A, cell, figure_sheet, main_edit, pick, widths, write_flat, write_wide, FB, FI, FT
from umt_object_contrast import CFG as OC
TPL = {int(v): k for k, v in json.load(open(ROOT / "data/ssv2/labels/labels.json")).items()}
CFG = dict(layers=(5, 7, 9), objects=OC["objects"], pairs=list(OC["pairs"]), wordmap_noun_L=7, wordmap_verb_L=9,
           min_word_n=15, n_similar=4, certainty=A / "noun_verb_scaleup/peak_certainty.csv",
           out=A / "object_verb_meeting/noun_verb_scaleup_101126.xlsx")

NOUN_MEASURES = ["Features chosen (n)", "Best AUROC (any latent)", "MLM: % of vision's log p removed",
                 "MLM: noun top-1 change (pp)", "VTC: 174-way acc change (pp)"]


def noun_rows(cfg):
    """UMT, per noun x layer: selection, masked-noun effect (MLM, test rows, R), class accuracy (VTC); raw."""
    rows = []
    for L in cfg["layers"]:
        f = pd.read_csv(A / f"umt_object_contrast_L{L}/features.csv")
        lp, ac = (pd.read_csv(A / f"umt_object_ablation_L{L}_splice/summary_{k}.csv") for k in ("logp", "acc"))
        vt = pd.read_csv(A / f"umt_object_vtc_L{L}_splice/summary_acc.csv")
        for o in cfg["objects"]:
            e, best = main_edit(lp.edit, o), pd.read_csv(A / f"umt_object_contrast_L{L}/candidates_{o}.csv").auroc.max()
            r, a = (pick(d, edit=e, object=o, cond="R", source="test") if e else None for d in (lp, ac))
            v = pick(vt, edit=e, clips="names object") if e else None
            vals = [int((f.object == o).sum()), best, np.nan if r is None else r.pct_headroom,
                    np.nan if a is None else 100 * a.d_acc, np.nan if v is None else 100 * v.d_acc]
            rows += [dict(object=o, layer=L, edit=e, measure=m, value=x) for m, x in zip(NOUN_MEASURES, vals)]
    return pd.DataFrame(rows)


units = lambda cfg: [(p, int(h), [int(c) for c in p.split("_") if int(c) != int(h)][0]) for p in cfg["pairs"] for h in p.split("_")]
VERB_MEASURES = ["Features chosen (n)", "MLM: within-pair acc change (pp)", "VTC: 174-way acc change (pp)",
                 "VTC: within-pair acc change (pp)", "MLM: Rev rescue of partner (pp)"]


def verb_rows(cfg, model="umt"):
    """Per verb unit (pair, home class) x layer, raw pp. UMT: all five measures; VideoMAE: VTC only (no MLM)."""
    rows, pp = [], lambda r, k: np.nan if r is None else 100 * r[f"d_{k}"]
    for L in cfg["layers"]:
        f = pd.read_csv(A / f"{model}_pair_contrast_L{L}/features.csv")
        vt = pd.read_csv(A / f"{model}_pair_vtc_L{L}_splice/summary_acc.csv")
        ab = pd.read_csv(A / f"umt_pair_ablation_L{L}_splice/summary_acc.csv") if model == "umt" else None
        for p, h, other in units(cfg):
            e = main_edit(vt.edit, f"{p}_{h}")
            mlm, rev = ((pick(ab, edit=e, object=f"{p}:{x}", cond=c, source="test") if e and ab is not None else None)
                        for x, c in ((h, "R"), (other, "Rev")))
            v = pick(vt, edit=e, clips="home class") if e else None
            vals = [int(((f.object == p) & (f.home == h)).sum()), pp(mlm, "pair_acc"), pp(v, "acc"), pp(v, "pair_acc"), pp(rev, "pair_acc")]
            rows += [dict(unit=TPL[h], against=TPL[other], layer=L, edit=e, measure=m, value=x) for m, x in zip(VERB_MEASURES, vals)]
    return pd.DataFrame(rows)


VM_LOGIT, VM_ACC = "VTC: true-class logit change", "VTC: 174-way acc change (pp)"


def vm_noun_rows(cfg):
    """VideoMAE, per noun x layer (clips naming the noun): features chosen, logit change, accuracy change; raw.
    Object L5 has no features.csv (nothing passed for any noun)."""
    rows = []
    for L in cfg["layers"]:
        fp = A / f"videomae_object_contrast_L{L}/features.csv"
        f = pd.read_csv(fp) if fp.exists() else pd.DataFrame(columns=["object"])
        lg, ac = (pd.read_csv(A / f"videomae_object_vtc_L{L}_splice/summary_{k}.csv") if f.size else None for k in ("logit", "acc"))
        for o in cfg["objects"]:
            e = main_edit(lg.edit, o) if lg is not None else None
            g, a = (pick(d, edit=e, clips="names object") if e else None for d in (lg, ac))
            vals = [int((f.object == o).sum()), np.nan if g is None else g.d_logit, np.nan if a is None else 100 * a.d_acc]
            rows += [dict(object=o, layer=L, edit=e, measure=m, value=x)
                     for m, x in zip(["Features chosen (n)", VM_LOGIT, VM_ACC], vals)]
    return pd.DataFrame(rows)


def vm_verb_logit_rows(cfg):
    """VideoMAE, per verb unit x layer (home-class clips): true-class logit change, raw."""
    rows = []
    for L in cfg["layers"]:
        lg = pd.read_csv(A / f"videomae_pair_vtc_L{L}_splice/summary_logit.csv")
        for p, h, other in units(cfg):
            e = main_edit(lg.edit, f"{p}_{h}")
            g = pick(lg, edit=e, clips="home class") if e else None
            rows.append(dict(unit=TPL[h], against=TPL[other], layer=L, edit=e, measure=VM_LOGIT,
                             value=np.nan if g is None else g.d_logit))
    return pd.DataFrame(rows)


KINDS = {"noun_mlm": "Noun: MLM % of log p removed", "verb_mlm": "Verb: MLM within-pair acc lost (pp)",
         "verb_vtc": "Verb: VTC within-pair acc lost (pp)"}


OVERVIEW = [   # from findings/noun_verb_scaleup_findings_091126.md (Status, §3.4-3.5, §4.5)
    ("UMT: nouns L7, verbs L9 holds", ["Nouns: 5 of 8 peak confidently at L7 (2,329 test rows); box and toy lean L7; phone peaks at L9; "
        "water has no features. Masked-noun damage stays on the noun's own name (and its category: cup -> mug, bowl).",
        "Verbs: 6 units peak confidently at L9 on the MLM (571 clips), 7 on the VTC (693 clips); 2 confident L7 units (into, un). "
        "Camera motion peaks early (L5). Verb features hit their own key word, not the partner's.",
        "Confident = paired bootstrap over clips, P(peak) >= 0.95, peak effect >= 5. See 'Layers'."]),
    ("VideoMAE", ["Nouns: almost nothing passes (none at L5); water L7 is the only effect (-0.36 logit, -6.2 pp).",
        "Verbs: mostly largest at L9, but 3 features per side barely move the decision while 30-50+ pass the contrast.",
        "Two-clip follow-up (10 Nov): camera motion sits in 1-2 shared raw directions; object motion in no shared direction "
        "(findings §3.5)."]),
    ("Method", ["SAE-spliced ablation of each unit's top features (up to 3, as a group), held-out test clips, real video (R).",
        "MLM = refit masked-word head (noun or verb word masked). VTC = UMT's / VideoMAE's 174-way class prediction."]),
    ("Caveats", ["RAW edit - baseline everywhere: the mass-matched controls broke in this run (pending a fix).",
        "MLM verb test: caption leakage pins 23 of 32 units at ceiling; the VTC carries those.",
        "Verb N is small (28-209 home clips per unit). CIs cover clip resampling only, not feature selection.",
        "Noun and verb peaks are on different measures: compare layers within a unit, not sizes across units."]),
]
NOTES = ["Each caption word is hidden in turn and UMT's masked-word head guesses it from the video and the rest of the caption; "
         "the features are then removed (SAE splice) and we measure how much harder the true word is to guess.",
         "Damage = drop in log-probability of the true word (raw, no control). 0 = no effect; ~0.7 = probability roughly halved.",
         "Summary heatmap (first figure): mean damage by kind of word. Blank row = no features passed at that layer.",
         "Caption maps: 30 random test captions, each word coloured by damage; bold = the key word(s)."]


def overview_sheet(ws, index):
    """Key points, method, caveats, and a linked index of the other sheets."""
    cell(ws, 1, 1, "Noun / verb scale-up: 9 nouns, 16 opposite pairs, L5 / L7 / L9, UMT and VideoMAE (raw, no controls)", FT)
    r = 3
    for head, points in OVERVIEW:
        cell(ws, r, 1, head, FB); r += 1
        for p in points:
            cell(ws, r, 1, "•"); cell(ws, r, 2, p, wrap=True); ws.row_dimensions[r].height = 28; r += 1
        r += 1
    write_flat(ws, r, "Sheets", pd.DataFrame({"sheet": [s for s, _ in index], "contents": [c for _, c in index]}),
               "Source: findings/noun_verb_scaleup_findings_091126.md", links={"sheet": [s for s, _ in index]})
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 22, 120


def nouns_sheet(ws, cfg, nouns):
    """UMT nouns by layer (wide), then the other words each noun's features hit (word map, raw)."""
    fm = {m: "0" if m.startswith("Features") else "0.00" if "AUROC" in m else "0.0" for m in NOUN_MEASURES}
    r = write_wide(ws, 1, "UMT nouns: selection, masked-noun effect (MLM), class effect (VTC)", nouns, ["object"], NOUN_MEASURES, fm,
                   "Raw edit - baseline (controls pending). Main edit = the noun's features as a group (one feature where only one passed). "
                   "% of log p removed: share of vision's contribution (baseline - blank-video log p). '–' = no features passed.")
    L, sw = cfg["wordmap_noun_L"], similar_words(cfg, cfg["wordmap_noun_L"])
    words = pd.DataFrame({"object": list(sw), f"other words hit (L{L}, raw damage)": list(sw.values()), "word map": "Word maps nouns"})
    write_flat(ws, r, "Nouns are categories: other caption words hit by each noun's features", words,
               f"Every caption word masked alone; mean -Δ log p, words seen ≥ {cfg['min_word_n']} times; own noun excluded.",
               links={"word map": words["word map"].tolist()})
    widths(ws); ws.column_dimensions["B"].width = 45   # the words list shares column B with the L5 'features' column


def verbs_sheet(ws, verbs):
    """UMT verb units by layer (wide), link to the verb word maps."""
    fm = {m: "0" if m.startswith("Features") else "0.0" for m in VERB_MEASURES}
    r = write_wide(ws, 1, "UMT verbs (opposite pairs): key-word effect (MLM) and class effect (VTC), home-class clips", verbs,
                   ["unit", "against"], VERB_MEASURES, fm, "Raw edit - baseline, pp (controls pending). Rev rescue: the partner class's "
                   "reversed clips, which look like this class. '–' = no features passed at that layer. MLM: 23 of 32 units at ceiling.")
    cell(ws, r, 1, "Word maps: verbs →", FB).hyperlink = "#'Word maps verbs'!A1"
    widths(ws); ws.column_dimensions["A"].width = ws.column_dimensions["B"].width = 40


def layers_sheet(ws, cfg):
    """Peak layer per unit with certainty, one table per readout."""
    r = 1
    for kind, title in KINDS.items():
        d = layer_rows(cfg, kind)
        r = write_flat(ws, r, title, d, "Effect: bigger = stronger. 95% CI and P(peak): paired bootstrap over the unit's test clips "
                       "(2,000 resamples, same resample at every layer). n = clips (nouns: masked rows).",
                       fmts={f"P(L{L} peak)": "0.00" for L in cfg["layers"]})
    widths(ws, first=40, rest=18); ws.column_dimensions["B"].width = 8


TWO_CLIPS = {"camera ↓": "101394", "object ↓": "180656"}   # findings §3.5: Turning the camera down / Moving something down
TWO_CLIP_ROWS = ["Clip", "Contrast passers, home side (n)", "Pair margin, unablated (spliced)",
                 "  … 3 passers zeroed", "  … 10 passers zeroed", "  … 20 passers zeroed (capped at pool)", "  … all passers zeroed",
                 "Passers' decoder |cos|, median (random ≈ 0.025)", "Top-10 leftover share of attribution after all zeroed (%)",
                 "Pair margin, raw stream (no SAE)", "  … 1 readout direction removed", "  … 2 directions removed",
                 "  … 3 directions removed", "Passers' coverage of readout direction 1 (%, sum cos²)",
                 "Of the 10 atoms best aligned with it: passers (n)"]


def two_clip_rows():
    """Findings §3.5 as one table: dose sweep, decoder geometry, leftover attribution, raw readout directions;
    columns VM / UMT x camera / object down. Reads the vtc_{dose_sweep,leftover_mass,readout_direction}_L9 outputs."""
    ds, cs = pd.read_csv(A / "vtc_dose_sweep_L9/dose_sweep.csv", dtype={"clip_id": str}), pd.read_csv(A / "vtc_dose_sweep_L9/cos_summary.csv")
    ms = pd.read_csv(A / "vtc_leftover_mass_L9/mass.csv", dtype={"clip_id": str})
    rd = pd.read_csv(A / "vtc_readout_direction_L9/margins.csv", dtype={"clip_id": str})
    ac = pd.read_csv(A / "vtc_readout_direction_L9/atom_cos.csv", dtype={"clip_id": str})
    cols = {}
    for m, short in (("videomae", "VM"), ("umt", "UMT")):
        for name, c in TWO_CLIPS.items():
            d, n = ds[(ds.model == m) & (ds.clip_id == c)].set_index("arm"), int(ds[(ds.model == m) & (ds.clip_id == c)].n_pool.iloc[0])
            one = lambda t: t[(t.model == m) & (t.clip_id == c)].iloc[0]
            cols[f"{short} {name}"] = [c, n, d.pair_margin["batch_base"]] + [d.pair_margin[f"dose_{min(k, n)}"] for k in (3, 10, 20, n)] + \
                [cs[(cs.model == m) & (cs.unit == d.unit.iloc[0])].iloc[0]["median"], 100 * one(ms).top10_share_full] + \
                [rd[(rd.model == m) & (rd.clip_id == c) & (rd.k == k)].margin.iloc[0] for k in (0, 1, 2, 3)] + \
                [100 * one(ac).pass_sumsq, int(one(ac).top10_in_passers)]
    return pd.DataFrame(cols, index=TWO_CLIP_ROWS).reset_index(names="measure")


def vm_sheet(ws, cfg):
    """VideoMAE (VTC only, no MLM): the two-downs redundancy table, then nouns and verbs, logit and accuracy separate."""
    n = vm_noun_rows(cfg)
    v = pd.concat([vm_verb_logit_rows(cfg), verb_rows(cfg, "videomae")])
    note = "Raw edit - baseline (controls pending). VideoMAE's own classifier; 3 features per side at most. '–' = none passed."
    t = two_clip_rows()
    r = write_flat(ws, 1, "Redundancy, two downs (L9, one clip each): camera motion vs object motion, VideoMAE vs UMT", t,
                   "Single clips, no controls: leads, not results (findings §3.5). Pair margin = home - partner logit. Dose = home-side passers "
                   "zeroed (spliced), ranked by mean of AUROC and attribution rank. Readout direction = gradient of the margin wrt the L9 "
                   "residual, removed around dim_mean, recomputed each step. VM camera: 1-2 shared directions, passers catch part of it "
                   "(low redundancy). VM object: no shared direction, passers align at chance (high redundancy).",
                   fmts={c: "0.00" for c in t.columns[1:]})
    r = write_wide(ws, r, "VideoMAE nouns: true-class logit change (clips naming the noun)", n, ["object"],
                   ["Features chosen (n)", VM_LOGIT], {"Features chosen (n)": "0", VM_LOGIT: "0.00"}, note)
    r = write_wide(ws, r, "VideoMAE nouns: class accuracy change (pp)", n, ["object"], [VM_ACC], {VM_ACC: "0.0"}, note)
    r = write_wide(ws, r, "VideoMAE verbs: true-class logit change (home-class clips)", v, ["unit", "against"],
                   ["Features chosen (n)", VM_LOGIT], {"Features chosen (n)": "0", VM_LOGIT: "0.00"}, note)
    acc = ["VTC: 174-way acc change (pp)", "VTC: within-pair acc change (pp)"]
    write_wide(ws, r, "VideoMAE verbs: class accuracy change (pp)", v, ["unit", "against"], acc, {m: "0.0" for m in acc},
               note + " 30-50+ features pass the contrast per side at L9: see findings §3.3-3.5 on redundancy.")
    widths(ws); ws.column_dimensions["A"].width = ws.column_dimensions["B"].width = 40


F, S = "findings/noun_verb_scaleup_findings_091126.md ", "src/stage3_analysis/"
REFERENCES = [   # sheet, findings, data (outputs/analysis/), script
    ("Nouns", F + "§4.1-4.2", "umt_object_{contrast_L*,ablation_L*_splice,vtc_L*_splice}/; umt_object_wordmap_L7/per_word.parquet",
     S + "umt_object_{contrast,ablation,vtc,wordmap}.py"),
    ("Verbs", F + "§4.3-4.4", "umt_pair_{contrast_L*,ablation_L*_splice,vtc_L*_splice}/", S + "same scripts, TASK=pair"),
    ("Layers", F + "§4.1, §4.3-4.5", "noun_verb_scaleup/peak_certainty.csv", "notebooks/umt_peak_certainty.py"),
    ("VideoMAE", F + "§3", "videomae_{object,pair}_{contrast_L*,vtc_L*_splice}/", S + "vm_object_contrast.py; umt_object_vtc.py MODEL=videomae"),
    ("Word maps nouns / verbs", F + "Next steps", "umt_{object,pair}_wordmap[_L7, _L9]/*.png", S + "umt_object_wordmap.py (MODE=plot)"),
]


def references_sheet(ws):
    """Where each sheet's numbers come from."""
    d = pd.DataFrame(REFERENCES, columns=["sheet", "findings", "data (outputs/analysis/)", "script"])
    write_flat(ws, 1, "References: sources for each sheet", d, "Workbook made by notebooks/noun_verb_scaleup_tables.py. "
               "outputs/ is gitignored: data paths are local / Isambard only.")
    for j, w in zip("ABCD", (22, 45, 90, 55)):
        ws.column_dimensions[j].width = w


def similar_words(cfg, L):
    """Per noun: the other caption words its main edit damages most at layer L (raw mean -d log p): 'mug 0.80, ...'."""
    w = pd.read_parquet(A / f"umt_object_wordmap_L{L}/per_word.parquet")
    out = {}
    for o in cfg["objects"]:
        e = main_edit(w.edit, o)
        if e is None:
            out[o] = "no features passed"
            continue
        d = w[w.edit == e].groupby("word").dce.agg(["mean", "size"])
        d = d[(d["size"] >= cfg["min_word_n"]) & ~d.index.isin([o, o + "s", o + "es"])].sort_values("mean", ascending=False)
        out[o] = ", ".join(f"{wd} {m:.2f}" for wd, m in d["mean"].head(cfg["n_similar"]).items())
    return out


def sure(g, best, p, min_effect=5.0):
    """Peak label: no effect if the peak's CI touches 0; small if the peak is < min_effect (findings §4.4 cut);
    one layer only if features passed at a single layer; else confident (P >= 0.95) / probable (0.80-0.95) / unsure."""
    if g.lo[best] <= 0:
        return "no clear effect"
    if g.value[best] < min_effect:
        return "small (< 5)"
    if len(g) == 1:
        return "one layer only"
    return "confident" if p >= 0.95 else "probable" if p >= 0.8 else "unsure"


def layer_rows(cfg, kind):
    """One kind from umt_peak_certainty: per unit, effect [95% CI] and P(peak) at each layer, peak and how sure."""
    c = pd.read_csv(cfg["certainty"])
    c = c[c.kind == kind]
    name = lambda u: u if kind == "noun_mlm" else TPL[int(u.split("_")[-1])]
    rows = []
    for u, g in c.groupby("unit", sort=False):
        g = g.set_index("L")
        best = int(g.p_peak.idxmax())
        p = g.p_peak.max()
        rows.append(dict(unit=name(u), n=int(g.n.iloc[0]),
                         **{f"L{L} effect [95% CI]": f"{g.value[L]:.1f} [{g.lo[L]:.1f}, {g.hi[L]:.1f}]" if L in g.index else "–" for L in cfg["layers"]},
                         **{f"P(L{L} peak)": g.p_peak.get(L, np.nan) for L in cfg["layers"]},
                         peak=f"L{best}", certainty=sure(g, best, p)))
    return pd.DataFrame(rows)


def main(cfg):
    wb = Workbook()
    index = [("Nouns", "UMT nouns by layer; other words hit"), ("Verbs", "UMT verb units by layer"),
             ("Layers", "peak layer per unit, 95% CI, P(peak), how sure"), ("VideoMAE", "nouns and verbs, logit and accuracy"),
             ("Word maps nouns", "figures"), ("Word maps verbs", "figures"), ("References", "findings, data files, scripts")]
    overview_sheet(wb.active, index); wb.active.title = "Overview"
    nouns_sheet(wb.create_sheet("Nouns"), cfg, noun_rows(cfg))
    verbs_sheet(wb.create_sheet("Verbs"), verb_rows(cfg))
    layers_sheet(wb.create_sheet("Layers"), cfg)
    vm_sheet(wb.create_sheet("VideoMAE"), cfg)
    Ln, Lv = cfg["wordmap_noun_L"], cfg["wordmap_verb_L"]
    figure_sheet(wb, "Word maps nouns", "Word maps: noun features (raw damage to each masked word)",
                 [("Summary: damage by kind of word, every noun and layer", A / "umt_object_wordmap/roles_by_layer.png")] +
                 [(f"'{o}' features removed (layer {Ln})", A / f"umt_object_wordmap_L{Ln}/wordmap_{o}.png") for o in cfg["objects"]
                  if (A / f"umt_object_wordmap_L{Ln}/wordmap_{o}.png").exists()], NOTES)
    figure_sheet(wb, "Word maps verbs", "Word maps: verb features (raw damage to each masked word)",
                 [("Summary: damage by kind of word, every verb unit and layer", A / "umt_pair_wordmap/roles_by_layer.png")] +
                 [(f"'{TPL[h]}' features removed (layer {Lv}), against '{TPL[o]}'", A / f"umt_pair_wordmap_L{Lv}/wordmap_{p}_{h}.png")
                  for p, h, o in units(cfg) if (A / f"umt_pair_wordmap_L{Lv}/wordmap_{p}_{h}.png").exists()], NOTES)
    references_sheet(wb.create_sheet("References"))
    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    wb.save(cfg["out"]); print(f"saved {cfg['out']}")


if __name__ == "__main__":
    main(CFG)
