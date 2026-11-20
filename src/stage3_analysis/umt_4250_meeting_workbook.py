"""Builds outputs/meeting/umt_4250_breakdown_meeting_201126.xlsx: #4250 broken down in pure PE / mixed / content
terms, DFA through the layers, behaviour under perturbation, verb vs noun; figures embedded next to their tables.
Usage (20/11/26): uv run python src/stage3_analysis/umt_4250_meeting_workbook.py"""
import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment, Font, PatternFill

from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])   # repo root; run after umt_4250_breakdown.py and umt_verb_noun_breakdown.py
BD, IN = f"{ROOT}/outputs/analysis/umt_4250_breakdown", f"{ROOT}/outputs/analysis/umt_4250_inputs_by_layer"
OUT = f"{ROOT}/outputs/meeting/umt_4250_breakdown_meeting_201126.xlsx"
F, B, H = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True), PatternFill("solid", fgColor="DDE6F0")
wb = Workbook(); wb.remove(wb.active)


def tab(name, source, df, img=None, img_scale=0.6, widths=None):
    """Table from row 1 (source as a note on A1); optional figure placed to the right of the table."""
    ws = wb.create_sheet(name)
    for j, h in enumerate(df.columns, 1):
        c = ws.cell(1, j, h); c.font, c.fill = B, H; c.alignment = Alignment(wrap_text=True, vertical="top")
    ws["A1"].comment = Comment(f"Source: {source}", "breakdown")
    for i, row in enumerate(df.itertuples(index=False), 2):
        for j, v in enumerate(row, 1):
            ws.cell(i, j, None if pd.isna(v) else (round(v, 3) if isinstance(v, float) else v)).font = F
    for j in range(len(df.columns)):
        ws.column_dimensions[chr(65 + j)].width = (widths or {}).get(j, 14)
    if img:
        im = Image(img); im.width, im.height = im.width * img_scale, im.height * img_scale
        ws.add_image(im, f"{chr(65 + len(df.columns) + 1)}1")
    ws.freeze_panes = "A2"
    return ws


def blocks(name, parts, widths):
    """Several titled tables stacked down one sheet: parts = [(title, source, DataFrame)]."""
    ws, r = wb.create_sheet(name), 1
    for title, source, df in parts:
        ws.cell(r, 1, title).font = Font(name="Arial", size=11, bold=True)
        if source: ws.cell(r, 1).comment = Comment(f"Source: {source}", "breakdown")
        r += 1
        for j, h in enumerate(df.columns, 1):
            c = ws.cell(r, j, h); c.font, c.fill = B, H; c.alignment = Alignment(wrap_text=True, vertical="top")
        for i, row in enumerate(df.itertuples(index=False), r + 1):
            for j, v in enumerate(row, 1):
                c = ws.cell(i, j, v); c.font = F; c.alignment = Alignment(wrap_text=True, vertical="top")
        r += len(df) + 3
    for j, w in widths.items():
        ws.column_dimensions[chr(65 + j)].width = w


S = "src/stage3_analysis/umt_4250_breakdown.py -> outputs/analysis/umt_4250_breakdown/"
story = pd.DataFrame([
    [1, "At L9 the down decision's top latent is #4250, and #4250 is a content latent (silent on a blank clip)", "rank 1 of 6,144 in down DFA; 4.2% of all DFA for 'down'", "03_4250_in_decision", "26 down clips (R-correct)"],
    [2, "Through the layers, the down decision's DFA moves from pure PE to content; mixed holds steady", "share of DFA for down: pure PE 44 -> 34 -> 25%, content 25 -> 33 -> 40%, mixed 31-35% (L5 -> L7 -> L9)", "02_down_DFA_by_layer", "22-26 down clips per layer"],
    [3, "Shuffling frame order leaves that split about the same", "L9 shuffled: pure PE 22%, mixed 34%, content 44% of DFA for down", "02_down_DFA_by_layer", "DFA ~3x larger under shuffle (gradients grow when the model is confused)"],
    [4, "#4250 is built, net, from mixed latents; pure PE and content are large in both directions and cancel", "net % of #4250: mixed +35 (from L5), +57 (from L7); pure PE -3 / -3; content -4 / +1", "04_4250_built_by_bucket", "linear attribution over all latents, 311 down clips"],
    [5, "The mixed inputs that carry #4250 at L7 need the motion in the right order; the pure-PE inputs do not", "reversed / real on down clips: #344 0.21, #1029 0.23, #230 0.33, #3423 0.56; pure-PE inputs 0.9-1.2", "05_4250_from_L7, 08_perturbation", "L5 down-clip reversal not stored"],
    [6, "Across the dictionary, pure PE stays locked to the PE; mixed latents need the PE's time code", "order index pure PE +0.47 / +0.27 / +0.24, mixed ~0; frozen PE keeps 47-81% of pure PE on blank, 0-13% of mixed", "08_perturbation", "medians over active latents"],
    [7, "Verb features lean on mixed features; noun features lean on content and pure PE", "share of top-20 ablation mass in mixed: verbs 56%, nouns 18% (gap +38 pt); pure PE + mixed: 76% vs 51% (+25 pt)", "09_verb_noun_shares", "7 targets: exact permutation p = 0.029 is the floor"],
    [8, "Summed over every latent, mixed builds verb features and works against noun features", "net mixed: down +35%, behind +10%; paper -34%, ball -20%", "10_verb_noun_net", "linear attribution, L5"],
    [9, "Use moves more than activation: pure PE stays active but is used less with depth; mixed is used more while its activation falls", "down clips L5 -> L9: pure PE activation -6 pt vs use -20 pt; mixed activation -5 pt vs use +5 pt", "11_activation_vs_use", "L5 activation from 8 down clips; DFA from 22-26"],
    [10, "VideoMAE: the down decision rests on content from L5 and the split does not move with depth", "share of DFA for down, pure PE / content: VideoMAE 12 / 53 -> 11 / 61 (L5 -> L9) vs UMT 44 / 25 -> 25 / 40", "12_VM_down_DFA_by_layer, 14_mass_by_layer", "VideoMAE has no single carrier feature like #4250"],
    [11, "VideoMAE: activation and use both stay flat; content is used more than its activation share", "content: 56-62% of DFA from 45-51% of activation", "13_VM_activation_vs_use", "VideoMAE L5 activation from 4 down clips"]],
    columns=["step", "claim", "key number", "evidence sheet", "caveat"])
VT = "outputs/analysis/umt_pair_vtc_L9_splice/summary_{}.csv (row 45_43_43_4250; L9 spliced; 155 down clips; other pairs 2,710 clips)"
acc = pd.DataFrame([["down clips: 174-way top-1", 70.3, -41.9], ["down clips: down vs up pair accuracy", 96.1, -16.8],
                    ["up clips: top-1", 61.5, 0.0], ["clips of other pairs: top-1", 68.6, -0.04]], columns=["readout", "baseline %", "change (pt)"])
lg = pd.DataFrame([["down logit (down clips)", -1.92], ["down - up margin (down clips)", -4.64], ["down logit, other pairs' clips", 0.00]],
                  columns=["readout", "change when #4250 is ablated"])
dec = pd.DataFrame([["rank in down DFA at L9", "1 of 6,144"], ["share of all DFA for 'down' at L9", "4.2%"],
                    ["bucket", "content (silent on a blank clip)"], ["why break it down", "one latent carries the down decision: removing it costs 42 pt of down top-1 and leaves other classes unchanged"]],
                   columns=["measure", "value"])
defs = pd.DataFrame([
    ["pure PE", "Fires on a blank grey clip (position embedding only, no content) at >= 0.8 x its mean level on real clips, and a synthetic moving object lifts it < 2 x its blank level", "the PE alone drives it"],
    ["mixed (PE x content)", "Fires on a blank clip, but below 0.8 x its real level, or an object lifts it >= 2 x blank", "the PE sets when it can fire; content sets how much"],
    ["content", "Below 1% of its real level on a blank clip", "needs content to fire"],
    ["real level", "Mean clip-summed activation over 696 val clips (4 per class, 174 classes)", "per latent, per layer"],
    ["active", "Fires on >= 50% of the real clips", "used for counts and medians"],
    ["DFA", "Gradient x activation of the correct-class logit, summed over tokens (trained SAE threshold); for = positive, against = negative", "the decision-critical mass"],
    ["C (shuffle)", "Frames shuffled, same frames", "perturbation"], ["Rev", "Frames reversed", "perturbation"],
    ["frozen PE", "Frame 0's PE given to every frame (content unchanged)", "removes the time code"],
    ["order index", "Per clip: r(Rev profile, R profile) - r(Rev profile, R reversed), median over clips; + = profile stays put when the video is reversed", "locked to PE vs follows content"]],
    columns=["term", "definition", "meaning"])
blocks("01_Front", [("#4250 (UMT L9 'down'): what ablating it does - accuracy", VT.format("acc"), acc),
                    ("#4250: what ablating it does - logit", VT.format("logit") + "; margin from findings/vm_down_pool_findings_191126.md §1", lg),
                    ("#4250 in the down decision", S + "4250_in_decision.csv", dec),
                    ("Definitions", "src/stage3_analysis/umt_layer_pe_characterise_analyse.py", defs),
                    ("Story", "this workbook", story)], widths={0: 34, 1: 75, 2: 55, 3: 26, 4: 40})
fim = Image(f"{ROOT}/outputs/analysis/umt_verb_noun_breakdown/verb_noun_by_bucket.png")   # verb / noun chart on the front
fim.width, fim.height = fim.width * 0.45, fim.height * 0.45
wb["01_Front"].add_image(fim, "G2")
tab("02_down_DFA_by_layer", S + "down_dfa_by_layer.csv (all-174 trained-threshold DFA, class 43, R-correct clips; dfa = mean per clip)",
    pd.read_csv(f"{BD}/down_dfa_by_layer.csv"), f"{BD}/down_dfa_by_layer.png", 0.55)
tab("03_4250_in_decision", S + "4250_in_decision.csv (L9, R)", pd.read_csv(f"{BD}/4250_in_decision.csv"))
tab("04_4250_built_by_bucket", S + "4250_attr_by_bucket.csv (% of #4250's activation; linear attribution, every latent)",
    pd.read_csv(f"{BD}/4250_attr_by_bucket.csv"), f"{BD}/4250_attr_by_bucket.png", 0.6)
inp = pd.read_csv(f"{BD}/4250_inputs_behaviour.csv")
tab("05_4250_from_L7", S + "4250_inputs_behaviour.csv (top 20 by |linear attribution|)", inp[inp.layer == 7].drop(columns="layer"), f"{IN}/inputs_4250_L7.png", 0.5)
tab("06_4250_from_L5", S + "4250_inputs_behaviour.csv (top 20 by |linear attribution|; down-clip reversal not stored at L5)",
    inp[inp.layer == 5].drop(columns=["layer", "down_Rev_over_R"]), f"{IN}/inputs_4250_L5.png", 0.5)
tab("07_spectrum", "src/stage3_analysis/umt_4250_inputs_by_layer.py (inputs at log10 blank / real; dotted lines: 1% and 0.8 x real)",
    inp[["layer", "feature", "bucket", "blank_real", "pct_of_4250_linear"]], f"{IN}/spectrum_4250_layers.png", 0.55)
tab("08_perturbation", S + "behaviour_by_bucket.csv (medians, active latents; frozen_kept on blank, blank for content)",
    pd.read_csv(f"{BD}/behaviour_by_bucket.csv"), f"{BD}/behaviour_by_bucket.png", 0.55)
VN = f"{ROOT}/outputs/analysis/umt_verb_noun_breakdown"
SV = "src/stage3_analysis/umt_verb_noun_breakdown.py -> outputs/analysis/umt_verb_noun_breakdown/"
ws = tab("09_verb_noun_shares", SV + "shares.csv (L5 inputs of 7 L9 targets; top-20 = single ablations, spliced, R, val; buckets as 01_Front)",
         pd.read_csv(f"{VN}/shares.csv"), f"{VN}/verb_noun_by_bucket.png", 0.42, widths={0: 16})
g, r0 = pd.read_csv(f"{VN}/gaps.csv"), 12                                      # gap table under the 7 target rows
for j, h in enumerate(g.columns, 1):
    c = ws.cell(r0, j, h); c.font, c.fill = B, H
for i, row in enumerate(g.itertuples(index=False), r0 + 1):
    for j, v in enumerate(row, 1):
        ws.cell(i, j, round(v, 3) if isinstance(v, float) else v).font = F
ws.cell(r0 - 1, 1, "Verb - noun gap (exact permutation p over the 35 ways to label 3 of 7 targets as nouns; 0.029 = the floor)").font = B
t20 = pd.read_csv(f"{VN}/top20_ablation.csv")
net = pd.read_csv(f"{VN}/attr_by_bucket.csv")
tab("10_verb_noun_net", SV + "attr_by_bucket.csv (linear attribution from every L5 latent, % of target activation) and top20_ablation.csv",
    pd.concat([net, pd.DataFrame([{}]), t20.rename(columns={"pct_removed": "single_ablation_pct_removed"})], ignore_index=True))
tab("11_activation_vs_use", S + "activation_vs_use.csv (down clips; share of each layer's total activation vs |DFA| vs DFA for 'down', by bucket)",
    pd.read_csv(f"{BD}/activation_vs_use.csv"), f"{BD}/activation_vs_use.png", 0.6)
VB, SVB = f"{ROOT}/outputs/analysis/vm_down_breakdown", "src/stage3_analysis/vm_down_breakdown.py -> outputs/analysis/vm_down_breakdown/"
tab("12_VM_down_DFA_by_layer", SVB + "down_dfa_by_layer.csv (VideoMAE trained-threshold all-174 DFA, class 43; C1 = tubelet-pair shuffle)",
    pd.read_csv(f"{VB}/down_dfa_by_layer.csv"), f"{VB}/down_dfa_by_layer.png", 0.55)
tab("13_VM_activation_vs_use", SVB + "activation_vs_use.csv (down clips; L5 activation from 4 clips, L7 / L9 from 311)",
    pd.read_csv(f"{VB}/activation_vs_use.csv"), f"{VB}/activation_vs_use.png", 0.6)
PC = f"{ROOT}/outputs/analysis"
u, v = pd.read_csv(f"{PC}/umt_layer_pe_characterise/summary_active.csv"), pd.read_csv(f"{PC}/vm_layer_pe_characterise/summary_active.csv")
vd = pd.read_csv(f"{VB}/dict_dfa_mass.csv"); nm = {"pe": "pure PE", "mixed": "mixed", "content": "content"}
mass = pd.concat([pd.DataFrame({"model": "UMT", "layer": u.layer, "group": u.cls.map(nm), "pct_active_latents": u.pct_active,
                                "pct_activation_mass": u.pct_act_mass_active, "pct_DFA_mass": u.pct_dfa_mass_active}),
                  pd.DataFrame({"model": "VideoMAE", "layer": v.layer, "group": v.cls.map(nm), "pct_active_latents": v.pct_active,
                                "pct_activation_mass": v.pct_act_mass, "pct_DFA_mass": vd.pct_dfa_mass_active.values})])
tab("14_mass_by_layer", "outputs/analysis/{umt,vm}_layer_pe_characterise/summary_active.csv; vm_down_breakdown/dict_dfa_mass.csv (active latents, all classes)",
    mass, f"{ROOT}/outputs/meeting/umt_vm_spectrum_mass_by_layer_201126.png", 0.45)
MT = f"{PC}/meeting_tables_201126"; src = pd.read_csv(f"{MT}/sources.csv").set_index("table").source
for i, (n, img) in enumerate([("VM_pure_PE_shapes", f"{ROOT}/outputs/meeting/vm_pure_pe_profiles_L5_201126.png"),
                              ("VM_UMT_down_logit", None), ("VM_UMT_down_acc", None), ("VM_down_groups_logit", None), ("VM_down_groups_acc", None),
                              ("frozen_PE_logit", None), ("frozen_PE_acc", None), ("frozen_PE_classes", None), ("PE_behaviour", None),
                              ("specificity_layers", None), ("specificity_parents", None), ("verb_noun_gap_191126", None),
                              ("verb_noun_test_191126", None), ("L3_MLM", None)], 15):
    tab(f"{i}_{n}", src[n], pd.read_csv(f"{MT}/{n}.csv"), img, 0.4)   # 19/11-and-earlier tables (meeting_tables_201126)
wb.save(OUT); print("saved", OUT)
