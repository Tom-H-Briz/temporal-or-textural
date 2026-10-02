"""Part C: metrics from the saved similarity matrix S (174 x N).

C1 — reproduction: repo's own itm_eval (text→video R@k, multi-GT min-rank).
C2 — classification: video→template top-1/top-5, per-class, splits, confusion.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
UMT_MM = ROOT / "models/unmasked_teacher/multi_modality"
sys.path.insert(0, str(UMT_MM))

CONFIG = dict(
    raw_dir=ROOT / "outputs/umt_ssv2_tpl_cls/raw",
    out_dir=ROOT / "outputs/umt_ssv2_tpl_cls",
    labels=ROOT / "data/ssv2/labels/labels.json",
    sl_temporal=[1, 6, 10, 12, 26, 34, 39, 45, 47, 56, 60, 63, 67, 70, 77, 81, 95, 150],
    sl_static=[9, 15, 42, 50, 58, 61, 63, 71, 78, 90, 93, 106, 109, 114, 121, 130, 138, 165],
)

# Reversal word pairs: curated antonyms (brief examples + SSv2-named pairs).
# Pairs are matched on single-word differences between template strings.
ANTONYMS = {
    ("left", "right"), ("putting", "taking"), ("opening", "closing"),
    ("pulling", "pushing"), ("covering", "uncovering"),
    ("attaching", "detaching"), ("toward", "away"), ("into", "out"),
    ("raising", "lowering"), ("lifting", "dropping"), ("up", "down"),
    ("onto", "off"), ("fake", "real"), ("moving", "pretending"),
}


def load_raw():
    S = np.load(CONFIG["raw_dir"] / "S.npy")
    meta = pd.read_csv(CONFIG["raw_dir"] / "meta.csv")
    maps = json.load(open(CONFIG["raw_dir"] / "gt_maps.json"))
    labels = json.load(open(CONFIG["labels"]))
    id2name = {int(v): k for k, v in labels.items()}
    return S, meta, maps, id2name


def c1_reproduction(S, maps):
    """Repo itm_eval. Axis naming per itm_eval source: its img_r* (ir) block
    ranks videos per text with scores_t2i -> that is TEXT->VIDEO (the
    published T2V axis); its txt_r* (tr) block is VIDEO->TEXT."""
    from tasks.retrieval_utils import itm_eval

    txt2img = {int(k): v for k, v in maps["txt2img"].items()}
    img2txt = {int(k): v for k, v in maps["img2txt"].items()}
    res = itm_eval(S.T.copy(), S.copy(), txt2img, img2txt)
    return dict(
        t2v_r1=res["img_r1"], t2v_r5=res["img_r5"], t2v_r10=res["img_r10"],
        v2t_r1=res["txt_r1"], v2t_r5=res["txt_r5"], v2t_r10=res["txt_r10"],
    )


def word_pair_diff(a, b):
    """The differing single word pair between two templates, else None."""
    wa, wb = a.lower().split(), b.lower().split()
    if len(wa) != len(wb):
        return None
    diffs = [(x, y) for x, y in zip(wa, wb) if x != y]
    return diffs[0] if len(diffs) == 1 else None


def is_reversal(a, b):
    """Left/right swap or a curated antonym single-word difference."""
    pair = word_pair_diff(a, b)
    if pair and (pair in ANTONYMS or pair[::-1] in ANTONYMS):
        return True
    wa, wb = a.lower().split(), b.lower().split()
    swap = {"left": "right", "right": "left"}
    return len(wa) == len(wb) and [swap.get(w, w) for w in wa] == wb and wa != wb


def c2_classification(S, meta, maps, id2name):
    """Top-1/5 overall + per class + most-confused.

    Predictions index the caption axis (annotation order); meta gt uses
    labels.json ids. Map caption index -> labels id via gt_maps texts
    (same normalisation as embed_pass: pre_text + bracket strip).
    """
    sys.path.insert(0, str(ROOT / "notebooks/umt_ssv2_tpl_cls"))
    from embed_pass import stub_decord
    stub_decord()  # dataset package imports decord at module level
    from dataset.utils import pre_text
    strip = lambda s: pre_text(s).replace("[", "").replace("]", "")
    labels = json.load(open(CONFIG["labels"]))
    cap2id = {strip(k): int(v) for k, v in labels.items()}
    cap_idx_to_label = np.array([cap2id[strip(t)] for t in maps["texts"]])

    gt = meta["gt_template_id"].to_numpy()
    order = np.argsort(-S, axis=0)                     # (174, N) caption rank per video
    top1 = cap_idx_to_label[order[0]]                  # labels-id space
    top5_hit = (cap_idx_to_label[order[:5]] == gt[None, :]).any(axis=0)
    rows = []
    for cid, name in id2name.items():
        m = gt == cid
        confused = Counter(top1[m & (top1 != cid)])
        rows.append(dict(
            class_id=cid, template=name, n=int(m.sum()),
            top1_correct=int((top1[m] == cid).sum()),
            top5_correct=int(top5_hit[m].sum()),
            most_confused_id=(confused.most_common(1)[0][0] if confused else -1),
            most_confused_rate=(round(confused.most_common(1)[0][1] / m.sum(), 3) if confused else 0.0),
        ))
    per_class = pd.DataFrame(rows)
    overall = dict(
        n=len(gt), top1=float((top1 == gt).mean()), top5=float(top5_hit.mean()),
    )
    return per_class, overall, top1, gt


def split_stats(per_class, ids):
    """Top-1 with counts for a class-id subset."""
    sub = per_class[per_class["class_id"].isin(ids)]
    return dict(
        classes=len(sub), n=int(sub["n"].sum()),
        top1_correct=int(sub["top1_correct"].sum()),
        top1_rate=round(sub["top1_correct"].sum() / max(sub["n"].sum(), 1), 4),
    )


def confusion(top1, gt, id2name):
    """Top-20 confused (gt, pred) pairs; reversal flagged via is_reversal."""
    pairs = Counter((int(g), int(p)) for g, p in zip(gt, top1) if g != p)
    rows = []
    for (g, p), n in pairs.most_common(20):
        rows.append(dict(
            gt_id=g, gt_template=id2name[g], pred_id=p, pred_template=id2name[p],
            count=n, reversal_pair=bool(is_reversal(id2name[g], id2name[p])),
        ))
    reversal_errors = sum(
        n for (g, p), n in pairs.items() if is_reversal(id2name[g], id2name[p])
    )
    total_errors = sum(pairs.values())
    return pd.DataFrame(rows), reversal_errors, total_errors


def main():
    S, meta, maps, id2name = load_raw()
    out = CONFIG["out_dir"]
    c1 = c1_reproduction(S, maps)
    per_class, overall, top1, gt = c2_classification(S, meta, maps, id2name)
    # consistency: C2 top-1 is the same argmax as the video->text axis of C1
    assert abs(overall["top1"] * 100 - c1["v2t_r1"]) < 0.02, (overall["top1"], c1["v2t_r1"])
    chiral_ids = [c for c, n in id2name.items() if "left" in n.lower() or "right" in n.lower()]
    sl = {k: split_stats(per_class, v) for k, v in
          [("sl_temporal", CONFIG["sl_temporal"]), ("sl_static", CONFIG["sl_static"]),
           ("chiral", chiral_ids), ("rest_non_chiral", [c for c in id2name if c not in chiral_ids])]}
    conf, rev_err, tot_err = confusion(top1, gt, id2name)
    per_class.to_csv(out / "per_class.csv", index=False)
    conf.to_csv(out / "top20_confused_pairs.csv", index=False)
    json.dump(dict(c1=c1, overall=overall, splits=sl, chiral_ids=chiral_ids,
                   reversal_errors=rev_err, total_errors=tot_err),
              open(out / "metrics_raw.json", "w"), indent=2)
    print("C1:", c1)
    print("C2:", overall)
    print("splits:", sl)
    print(f"reversal errors {rev_err}/{tot_err}")
    return c1, overall, sl, conf


if __name__ == "__main__":
    main()
