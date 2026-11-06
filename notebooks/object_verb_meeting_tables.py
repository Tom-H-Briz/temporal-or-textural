"""Object (noun) vs verb features across UMT layers: meeting workbook (05/11/26, CPU, local).
Reads the outputs of umt_object_contrast / umt_object_ablation / umt_object_vtc / umt_object_wordmap
(TASK=object and pair) and writes one Excel file: overview, nouns, verbs, layers, cup by class, and
the word-map figures embedded (so the file can be shared as is). Findings:
findings/umt_object_verb_layers_findings_031126.md.
Usage: uv run python notebooks/object_verb_meeting_tables.py
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs" / "analysis"
CFG = dict(layers=(5, 7, 9), objects=("box", "pen", "cup"), container_layer=7, min_word_n=15, n_similar=4,
           pairs={"45_43": ("up", "down"), "168_165": ("camera up", "camera down"), "46_5": ("opening", "closing"),
                  "93_86": ("push, left to right", "pull, left to right"),
                  "94_87": ("push, right to left", "pull, right to left")},
           labels=ROOT / "data/ssv2/labels/labels.json", train=ROOT / "data/ssv2/labels/train.json",
           pair_table=A / "umt_pair_table" / "pair_summary.csv",
           out=A / "object_verb_meeting" / "object_verb_summary_051126.xlsx")


def main_edit(edits, unit):
    """A unit's main edit: its group if it has one, else its single feature, else None (nothing passed)."""
    names = sorted(e for e in set(edits) if e.startswith(unit + "_") and e[len(unit) + 1:].replace("group", "0").isdigit())
    return f"{unit}_group" if f"{unit}_group" in names else (names[0] if names else None)


def pick(df, **eq):
    """The one row of df matching every column == value (None if absent)."""
    m = np.logical_and.reduce([df[k] == v for k, v in eq.items()])
    return df[m].iloc[0] if m.any() else None


def units(cfg):
    """Verb units: (pair, home class, label), e.g. ('45_43', 43, 'down')."""
    return [(p, int(h), names[i]) for p, names in cfg["pairs"].items() for i, h in enumerate(p.split("_"))]


NOUN_MEASURES = ["Selection: best AUROC (within class)", "Noun: % of vision's log p removed",
                 "Noun: top-1 change (pp)", "Class: 174-way acc change (pp)"]


def noun_rows(cfg):
    """Per object x layer: selection strength, noun log p and top-1 (MLM), class accuracy (VTC); all edit - control."""
    rows = []
    for o in cfg["objects"]:
        for L in cfg["layers"]:
            best = pd.read_csv(A / f"umt_object_contrast_L{L}/candidates_{o}.csv").auroc.max()   # any latent, as findings §4
            lp, ac = (pd.read_csv(A / f"umt_object_ablation_L{L}_splice/summary_{k}.csv") for k in ("logp", "acc"))
            vt = pd.read_csv(A / f"umt_object_vtc_L{L}_splice/summary_acc.csv")
            e = main_edit(lp.edit, o)
            r, a, v = pick(lp, edit=e, object=o, cond="R", source="test"), pick(ac, edit=e, object=o, cond="R", source="test"), \
                pick(vt, edit=e, clips="names object")
            vals = [best, r.pct_headroom, 100 * (a.d_acc - a.ctrl_d_acc), 100 * (v.d_acc - v.ctrl_d_acc)]
            rows += [dict(object=o, layer=L, edit=e, measure=m, value=x) for m, x in zip(NOUN_MEASURES, vals)]
    return pd.DataFrame(rows)


def similar_words(cfg, o, L):
    """Words (not the object's own noun) most damaged by o's main edit at layer L, minus control: 'mug 0.80, ...'."""
    w = pd.read_parquet(A / f"umt_object_wordmap_L{L}/per_word.parquet")
    e = main_edit(w.edit, o)
    g = w[w.edit.isin([e, f"ctrl_{e}"])].pivot_table(index=["clip_id", "position", "word"], columns="edit", values="dce")
    d = (g[e] - g[f"ctrl_{e}"]).groupby("word").agg(["mean", "size"])
    d = d[(d["size"] >= cfg["min_word_n"]) & ~d.index.isin([o, o + "s", o + "es"])].sort_values("mean", ascending=False)
    return ", ".join(f"{wd} {m:.2f}" for wd, m in d["mean"].head(cfg["n_similar"]).items())


CONTAINER = re.compile(r"(?<!hole )\binto something|\bpour|\bspill|in it\b|\bout of\b|overflow|\binside\b|\bempty\b")
is_container = lambda tpl: bool(CONTAINER.search(tpl.lower())) and not tpl.lower().startswith("plugging")


def vtc_clip_effects(cfg, o, L):
    """Per clip naming o: class, template, container flag, true-class logit and correct change (edit - control)."""
    d = pd.read_parquet(A / f"umt_object_vtc_L{L}_splice/per_clip.parquet")
    d = d[d.objects.str.split(",").apply(lambda x: o in x)]
    e = main_edit(d.arm, o)
    lt, co = (d.pivot(index="clip_id", columns="arm", values=v).astype(float) for v in ("logit_true", "correct"))
    tpl = {int(v): k.replace("[", "").replace("]", "") for k, v in json.load(open(cfg["labels"])).items()}
    cls = d.drop_duplicates("clip_id").set_index("clip_id").true_class.loc[lt.index]
    return pd.DataFrame({"class": cls, "template": cls.map(tpl), "container": cls.map(tpl).map(is_container),
                         "d_logit": lt[e] - lt[f"ctrl_{e}"], "d_acc_pp": 100 * (co[e] - co[f"ctrl_{e}"])})


def container_rows(cfg, L):
    """Per object: clips naming it in container vs other classes, class logit and accuracy change (edit - control)."""
    rows = []
    for o in cfg["objects"]:
        for cont, g in vtc_clip_effects(cfg, o, L).groupby("container"):
            rows.append(dict(object=o, classes="container" if cont else "other", clips=len(g),
                             **{"class logit change": g.d_logit.mean(), "class acc change (pp)": g.d_acc_pp.mean()}))
    return pd.DataFrame(rows)


VERB_MEASURES = ["Features chosen (n)", "MLM: within-pair acc change (pp)", "Class: 174-way acc change (pp)",
                 "Class: within-pair acc change (pp)", "Rev rescue of partner (pp)"]


def verb_rows(cfg):
    """Per verb unit x layer: MLM key-word pair accuracy, VTC 174-way and within-pair accuracy (home class clips),
    and the reverse rescue (partner class's reversed clips); all edit - control, pp."""
    rows, pp = [], lambda r, k: np.nan if r is None else 100 * (r[f"d_{k}"] - r[f"ctrl_d_{k}"])
    for L in cfg["layers"]:
        f = pd.read_csv(A / f"umt_pair_contrast_L{L}/features.csv")
        ab, vt = pd.read_csv(A / f"umt_pair_ablation_L{L}_splice/summary_acc.csv"), pd.read_csv(A / f"umt_pair_vtc_L{L}_splice/summary_acc.csv")
        for p, h, label in units(cfg):
            other, e = [int(c) for c in p.split("_") if int(c) != h][0], main_edit(ab.edit, f"{p}_{h}")
            mlm = pick(ab, edit=e, object=f"{p}:{h}", cond="R", source="test") if e else None
            rev = pick(ab, edit=e, object=f"{p}:{other}", cond="Rev", source="test") if e else None
            v = pick(vt, edit=e, clips="home class") if e else None
            vals = [len(f[(f.object == p) & (f.home == h)]), pp(mlm, "pair_acc"), pp(v, "acc"), pp(v, "pair_acc"), pp(rev, "pair_acc")]
            rows += [dict(pair=p, against=[w for w in cfg["pairs"][p] if w != label][0], home=h, unit=label, layer=L, edit=e, measure=m, value=x) for m, x in zip(VERB_MEASURES, vals)]
    return pd.DataFrame(rows)


def layer_rows(nouns, verbs, min_effect=5.0):
    """One row per unit: its MLM effect at each layer and the peak layer (none if every |effect| < min_effect).
    Nouns: % of vision's log p removed; verbs: within-pair accuracy lost (pp, sign flipped so bigger = stronger)."""
    n = nouns[nouns.measure == NOUN_MEASURES[1]].assign(unit=lambda d: d.object, kind="noun")
    v = verbs[verbs.measure == VERB_MEASURES[1]].assign(value=lambda d: -d.value,
                                                        kind=lambda d: np.where(d.pair == "168_165", "camera motion", "verb"))
    rows = []
    for (kind, unit), g in pd.concat([n, v]).groupby(["kind", "unit"], sort=False):
        s = g.set_index("layer").value
        peak = "none" if s.abs().max() < min_effect or s.isna().all() else f"L{int(s.idxmax())}"
        rows.append(dict(kind=kind, unit=unit, **{f"L{L}": s.get(L) for L in s.index}, peak=peak))
    return pd.DataFrame(rows)

F, FB, FI, FT = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True), \
    Font(name="Arial", size=9, italic=True, color="555555"), Font(name="Arial", size=13, bold=True)
HEAD = PatternFill("solid", fgColor="DDE4EE")


def cell(ws, r, c, v, font=F, fill=None, fmt=None, wrap=False):
    """Write one cell (NaN -> '–'), Arial, optional fill / number format."""
    x = ws.cell(r, c, "–" if isinstance(v, float) and np.isnan(v) else v)
    x.font, x.alignment = font, Alignment(wrap_text=wrap, vertical="top")
    if fill: x.fill = fill
    if fmt and isinstance(v, (int, float)): x.number_format = fmt
    return x


def write_wide(ws, r, title, long, index, measures, fmts, note, layers=(5, 7, 9)):
    """Long (index.., measure, layer, value) -> a table with one merged header per measure over its L5/L7/L9 columns."""
    cell(ws, r, 1, title, FB)
    keys = long[index].drop_duplicates().values.tolist()
    for j, name in enumerate(index):
        cell(ws, r + 1, j + 1, name, FB, HEAD); cell(ws, r + 2, j + 1, "", FB, HEAD)
    c = len(index) + 1
    for m in measures:
        ws.merge_cells(start_row=r + 1, start_column=c, end_row=r + 1, end_column=c + len(layers) - 1)
        cell(ws, r + 1, c, m, FB, HEAD, wrap=True)
        for k, L in enumerate(layers):
            cell(ws, r + 2, c + k, f"L{L}", FB, HEAD)
            for i, key in enumerate(keys):
                v = long[(long.measure == m) & (long.layer == L) & np.logical_and.reduce([long[n] == x for n, x in zip(index, key)])].value
                cell(ws, r + 3 + i, c + k, float(v.iloc[0]) if len(v) else np.nan, fmt=fmts[m])
        c += len(layers)
    for i, key in enumerate(keys):
        for j, x in enumerate(key):
            cell(ws, r + 3 + i, j + 1, x)
    cell(ws, r + 3 + len(keys), 1, note, FI)
    return r + 5 + len(keys)


def write_flat(ws, r, title, df, note, fmts=None, links=None):
    """df as a plain table. links: {column: sheet name per row} -> that cell links to the sheet inside the workbook."""
    cell(ws, r, 1, title, FB)
    for j, col in enumerate(df.columns):
        cell(ws, r + 1, j + 1, col, FB, HEAD, wrap=True)
    for i, row in enumerate(df.itertuples(index=False)):
        for j, (col, v) in enumerate(zip(df.columns, row)):
            x = cell(ws, r + 2 + i, j + 1, v, fmt=(fmts or {}).get(col), wrap=isinstance(v, str) and len(v) > 30)
            if links and col in links:
                x.hyperlink, x.font = f"#'{links[col][i]}'!A1", Font(name="Arial", size=10, color="0563C1", underline="single")
    cell(ws, r + 2 + len(df), 1, note, FI)
    return r + 4 + len(df)


def widths(ws, first=14, rest=11):
    """Column widths: wider label columns, narrow number columns."""
    for j in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(j)].width = first if j <= 3 else rest


def figure_sheet(wb, name, title, figs, notes=(), max_w=900):
    """A sheet of embedded figures under a 'how to read' block, stacked, each under its caption (scaled to max_w px)."""
    ws, r = wb.create_sheet(name), 4
    cell(ws, 1, 1, title, FT)
    cell(ws, 2, 1, "← Overview", Font(name="Arial", size=10, color="0563C1", underline="single")).hyperlink = "#'Overview'!A1"
    if notes:
        cell(ws, r, 1, "How to read these figures", FB); r += 1
        for n in notes:
            cell(ws, r, 1, f"•  {n}"); r += 1
        r += 1
    for caption, path in figs:
        cell(ws, r, 1, caption, FB)
        img = XLImage(str(path))
        s = min(1.0, max_w / img.width)
        img.width, img.height = int(img.width * s), int(img.height * s)
        ws.add_image(img, f"A{r + 1}")
        r += int(img.height / 20) + 3                 # default row ~20 px
    return ws


def specific_rows(cfg):
    """2 Nov pair table (L9 MLM contrast): class-specific features, AP shares, VTC within-pair accuracy, train clips."""
    tpl = {int(v): k for k, v in json.load(open(cfg["labels"])).items()}
    train = pd.Series([tpl_of(t["template"]) for t in json.load(open(cfg["train"]))]).value_counts()
    p = pd.read_csv(cfg["pair_table"])
    kind = {"opposite": "opposite", "same_frame": "same frame", "excluded_flip_aug": "excluded (flip aug)"}
    return pd.DataFrame({"kind": p.kind.map(kind), "pair": p.pair, "class A": p.a.map(tpl), "class B": p.b.map(tpl),
        "class-specific features": p.n_specific, "AP share: specific (%)": 100 * p.share_specific,
        "AP share: timing machinery (%)": 100 * p.share_machinery,
        "VTC within-pair acc A": p.vis_acc_a, "VTC within-pair acc B": p.vis_acc_b, "weaker side": p.vis_acc_min,
        "train clips A": p.a.map(tpl).map(train), "train clips B": p.b.map(tpl).map(train),
        "val clips A": p.n_val_a, "val clips B": p.n_val_b})


tpl_of = lambda t: t.replace("[", "").replace("]", "")   # train.json template -> labels.json key


OVERVIEW = [
    ("Nouns (box, pen, cup)", ["Easy to find: within-class contrast (object vs no object, same classes), 500 + 500 clips per object.",
        "Features are categories: pen also carries pencil, marker; cup carries mug, glass, tea.",
        "Little impact on the class prediction (≤ 1.7 pp). Cup is the exception: −6.7 pp on container / liquid classes at L7."]),
    ("Verbs (opposite pairs)", ["The contrast has to use the classes (class A vs class B of a pair). The pairs are vision driven: "
        "captions differ only in the class word, and text alone (blank video) is at chance (2 Nov).",
        "Much more impact on the class prediction: 5–61 pp on the features' own class.",
        "Some classes have critical features: down (−48 pp at L9). Without down, up is assumed.",
        "Finer same-frame pairs (putting / stuffing, pouring / spilling, pulling / taking out) have no class-specific "
        "features and perform worse (0.63–0.75 on the weaker class; 2 Nov). See 'Pairs without features'."]),
    ("Layers", ["Different layers focus on different features: camera motion early (L5), nouns mid (L7), verbs late (L9).",
        "Consistent with the view that verb features are harder to build."]),
    ("Method", ["SAE-spliced ablation of each unit's top features (up to 3, as a group) vs mass-matched control features; "
        "every number is edit − control, held-out test clips, real video unless stated.",
        "MLM = refit masked-word head (noun or verb masked). Class = UMT's 174-way video-text (VTC) prediction."]),
    ("Caveats", ["By design: object features are chosen within class, and SSv2 templates are object-agnostic ('something'), "
        "so a small class effect for nouns is expected; verb features are chosen between classes.",
        "Word-level word-map values rest on 15–60 occurrences; camera down has 31 test clips."]),
]


WORDMAP_NOTES = [
    "Test: each word of a caption is hidden in turn, and UMT's masked-word head guesses it from the video and the rest of "
    "the caption. We then remove the features (SAE splice) and measure how much harder the true word is to guess.",
    "Damage = drop in log-probability of the true word. 0 = no effect; ~0.7 = the word's probability roughly halved.",
    "Summary heatmap (first figure): mean damage by kind of word, minus the same with control features (unrelated features "
    "of the same size). Blank row = no features passed at that layer.",
    "Caption maps: 30 random test captions, each word coloured by its damage (control not subtracted). Grey number = SSv2 "
    "clip id; bold = the key word(s).",
]
NOUN_NOTES = WORDMAP_NOTES + ["Read: the object's features should hit the object's name and nothing else. "
    "Caption maps are layer 7, where the object effects are largest."]
VERB_NOTES = WORDMAP_NOTES + ["Each row is one class's features, e.g. 'Moving something down' features, contrasted with "
    "its partner class (Moving something up). Captions come from both classes; both key words are bold.",
    "Read: they should hit their own key word (down) and not the partner's (up) or the object words. "
    "Caption maps are layer 9. Push / pull at layer 5 are one feature each."]


def overview_sheet(ws, index):
    """Key points, method, caveats, and a linked index of the other sheets."""
    cell(ws, 1, 1, "UMT vision features: objects (nouns) vs verbs across layers L5 / L7 / L9", FT)
    r = 3
    for head, points in OVERVIEW:
        cell(ws, r, 1, head, FB); r += 1
        for p in points:
            cell(ws, r, 1, "•"); cell(ws, r, 2, p, wrap=True); ws.row_dimensions[r].height = 28; r += 1
        r += 1
    r = write_flat(ws, r, "Sheets", pd.DataFrame({"sheet": [s for s, _ in index], "contents": [c for _, c in index]}),
                   "Source: findings/umt_object_verb_layers_findings_031126.md", links={"sheet": [s for s, _ in index]})
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 22, 120


def nouns_sheet(ws, cfg, nouns):
    """Nouns by layer (wide), then L7 words also hit + word-map links, then container vs other classes."""
    fm = {NOUN_MEASURES[0]: "0.00", NOUN_MEASURES[1]: "0.0", NOUN_MEASURES[2]: "0.0", NOUN_MEASURES[3]: "0.0"}
    r = write_wide(ws, 1, "Nouns: selection, effect on the masked noun (MLM), effect on the class (VTC)", nouns, ["object"],
                   NOUN_MEASURES, fm, "Edit − control throughout. Main edit = the object's top features as a group (L5: one feature). "
                   "Headroom = vision's contribution to the noun (baseline − blank-video log p).")
    L = cfg["container_layer"]
    words = pd.DataFrame({"object": cfg["objects"], "edit (L7)": "feature group",
                          "words also hit (damage − control)": [similar_words(cfg, o, L) for o in cfg["objects"]],
                          "word map": ["Word maps nouns"] * len(cfg["objects"])})
    r = write_flat(ws, r, f"Nouns are categories: other caption words hit by each object's L{L} features", words,
                   f"Every caption word masked alone; mean −Δ log p, edit − control, words seen ≥ {cfg['min_word_n']} times.",
                   links={"word map": words["word map"].tolist()})
    c = container_rows(cfg, L)
    write_flat(ws, r, f"Cup is the exception: class effect split by container / liquid classes (L{L})", c,
               "Container classes by template wording (into something, pour, spill, in it, out of, overflow, inside, empty); "
               "21 classes. See sheet 'Cup by class'.", fmts={"class logit change": "0.000", "class acc change (pp)": "0.0"})
    widths(ws)
    ws.column_dimensions["C"].width = 40


def verbs_sheet(ws, cfg, verbs):
    """Verb units by layer (wide), with a link to the verb word maps."""
    fm = {m: ("0" if m.startswith("Features") else "0.0") for m in VERB_MEASURES}
    r = write_wide(ws, 1, "Verbs (vision-driven opposite pairs): effect on the key word (MLM) and on the class (VTC)", verbs,
                   ["unit", "against"], VERB_MEASURES, fm, "Edit − control, pp, on the features' own class (Rev rescue: the partner "
                   "class's reversed clips, which look like the own class). '–' = no features passed the 0.7 bar at that layer.")
    cell(ws, r, 1, "Word maps: verbs →", FB).hyperlink = "#'Word maps verbs'!A1"
    widths(ws)


def layers_sheet(ws, layers):
    """Peak layer per unit (MLM effect)."""
    write_flat(ws, 1, "Different layers, different features: MLM effect by layer and peak layer", layers,
               "Nouns: % of vision's log p removed. Verbs and camera: within-pair accuracy lost (pp). "
               "Peak = largest effect; 'none' if every layer is below 5.", fmts={f"L{L}": "0.0" for L in (5, 7, 9)})
    widths(ws)


def cup_sheet(ws, cfg, min_n=5):
    """Every class with ≥ min_n cup test clips: the L7 cup features' effect on the class (most hurt first)."""
    d = vtc_clip_effects(cfg, "cup", cfg["container_layer"])
    g = d.groupby(["class", "template", "container"]).agg(**{"clips": ("d_logit", "size"), "class logit change": ("d_logit", "mean"),
                                                             "class acc change (pp)": ("d_acc_pp", "mean")}).reset_index()
    g = g[g.clips >= min_n].sort_values("class logit change").assign(container=lambda t: t.container.map({True: "yes", False: "no"}))
    write_flat(ws, 1, f"Cup features (L{cfg['container_layer']}) by class: container / liquid classes are hit hardest", g,
               f"Clips naming a cup, classes with ≥ {min_n}. Edit − control. 'Showing that something is empty' gains: "
               "4 of 29 clips flip to correct (control: 0).", fmts={"class logit change": "0.000", "class acc change (pp)": "0.0"})
    widths(ws); ws.column_dimensions["B"].width = 60


def specific_sheet(ws, cfg):
    """2 Nov: pairs with vs without class-specific features (no-feature pairs shaded), then group means."""
    d = specific_rows(cfg)
    fm = {c: ("0.0" if "%" in c else "0.00" if "acc" in c or "weaker" in c else "0") for c in d.columns[4:]}
    r = write_flat(ws, 1, "Pairs with no class-specific features perform worse (L9, vision-driven pairs, 2 Nov)", d,
                   "AP = attribution patching (gradient × activation on the MLM pair word), share of the pair's total importance. "
                   "Accuracy: 174-way VTC, logit A > logit B, all val clips. Train clips: SSv2 train set. Camera left / right "
                   "excluded (UMT trains with random horizontal flips).", fmts=fm)
    for i in np.flatnonzero((d["class-specific features"] == 0) & (d.kind != "excluded (flip aug)")):
        for j in range(1, len(d.columns) + 1):
            ws.cell(3 + i, j).fill = PatternFill("solid", fgColor="FFF2CC")
    d["group"] = np.where(d.kind.str.startswith("excluded"), None, np.where(d["class-specific features"] > 0,
                          d.kind + ", with features", "same frame, none"))
    g = d.dropna(subset=["group"]).assign(**{"min train clips": d[["train clips A", "train clips B"]].min(axis=1)})
    g = g.groupby("group", sort=False).agg(pairs=("pair", ", ".join), **{"weaker side (mean)": ("weaker side", "mean"),
        "min train clips (mean)": ("min train clips", "mean"), "AP timing share % (mean)": ("AP share: timing machinery (%)", "mean")})
    write_flat(ws, r, "By group", g.reset_index(), "Shaded rows above: no class-specific features.",
               fmts={"weaker side (mean)": "0.00", "min train clips (mean)": "0", "AP timing share % (mean)": "0.0"})
    widths(ws); ws.column_dimensions["C"].width = ws.column_dimensions["D"].width = 34


F, S = "findings/", "src/stage3_analysis/"
REFERENCES = [   # sheet, findings, data (outputs/analysis/), script
    ("Nouns", F + "umt_object_verb_layers_findings_031126.md §3–5, §8", "umt_object_contrast_L{5,7,9}/candidates_*.csv; "
     "umt_object_ablation_L{5,7,9}_splice/summary_{logp,acc}.csv; umt_object_vtc_L{5,7,9}_splice/summary_acc.csv; "
     "umt_object_wordmap_L7/per_word.parquet", S + "umt_object_{contrast,ablation,vtc,wordmap}.py"),
    ("Verbs", F + "umt_object_verb_layers_findings_031126.md §6, §8", "umt_pair_contrast_L{5,7,9}/features.csv; "
     "umt_pair_ablation_L{5,7,9}_splice/summary_acc.csv; umt_pair_vtc_L{5,7,9}_splice/summary_acc.csv",
     S + "umt_object_{contrast,ablation,vtc}.py (TASK=pair)"),
    ("Pairs without features", F + "umt_pair_specific_features_findings_021126.md §3", "umt_pair_table/pair_summary.csv; "
     "data/ssv2/labels/train.json (train clip counts)", S + "umt_pair_specific_features.py"),
    ("Layers", F + "umt_object_verb_layers_findings_031126.md §4, §6", "built from the Nouns and Verbs tables", "this notebook"),
    ("Cup by class", "not in a findings doc (cup / container test, 5 Nov)", "umt_object_vtc_L7_splice/per_clip.parquet",
     S + "umt_object_vtc.py"),
    ("Word maps nouns / verbs", F + "umt_object_verb_layers_findings_031126.md §5 (nouns), §8 (verbs)",
     "umt_{object,pair}_wordmap[_L7, _L9]/*.png", S + "umt_object_wordmap.py (TASK=object / pair)"),
]


def references_sheet(ws):
    """Where each sheet's numbers come from: findings doc, data files, script."""
    d = pd.DataFrame(REFERENCES, columns=["sheet", "findings", "data (outputs/analysis/ unless stated)", "script"])
    write_flat(ws, 1, "References: sources for each sheet", d, "Workbook made by notebooks/object_verb_meeting_tables.py. "
               "outputs/ is gitignored: data paths are local / Isambard only.")
    for j, w in zip("ABCD", (22, 55, 90, 50)):
        ws.column_dimensions[j].width = w


def main(cfg):
    nouns, verbs = noun_rows(cfg), verb_rows(cfg)
    wb = Workbook()
    index = [("Nouns", "objects by layer; words also hit; container split"), ("Verbs", "verb pairs by layer"),
             ("Pairs without features", "2 Nov: pairs with no class-specific features, accuracy, train clips"),
             ("Layers", "peak layer per unit"), ("Cup by class", "the L7 cup exception, class by class"),
             ("Word maps nouns", "figures"), ("Word maps verbs", "figures"),
             ("References", "findings docs, data files and scripts behind each sheet")]
    overview_sheet(wb.active, index); wb.active.title = "Overview"
    nouns_sheet(wb.create_sheet("Nouns"), cfg, nouns)
    verbs_sheet(wb.create_sheet("Verbs"), cfg, verbs)
    specific_sheet(wb.create_sheet("Pairs without features"), cfg)
    layers_sheet(wb.create_sheet("Layers"), layer_rows(nouns, verbs))
    cup_sheet(wb.create_sheet("Cup by class"), cfg)
    figure_sheet(wb, "Word maps nouns", "Word maps: object features (damage to each masked word, edit − control)",
                 [("Summary: damage by kind of word, every object and layer", A / "umt_object_wordmap/roles_by_layer.png")] +
                 [(f"'{o}' features removed (layer 7): captions naming a {o}", A / f"umt_object_wordmap_L7/wordmap_{o}.png")
                  for o in cfg["objects"]], NOUN_NOTES)
    other = lambda p, label: [x for x in cfg["pairs"][p] if x != label][0]
    figure_sheet(wb, "Word maps verbs", "Word maps: verb features (damage to each masked word)",
                 [("Summary: damage by kind of word, every verb and layer", A / "umt_pair_wordmap/roles_by_layer.png")] +
                 [(f"'{label}' features removed (layer 9), against '{other(p, label)}'", A / f"umt_pair_wordmap_L9/wordmap_{p}_{h}.png")
                  for p, h, label in units(cfg)], VERB_NOTES)
    references_sheet(wb.create_sheet("References"))
    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    wb.save(cfg["out"])
    print(f"saved {cfg['out']}")


if __name__ == "__main__":
    main(CFG)
