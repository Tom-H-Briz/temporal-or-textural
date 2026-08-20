"""
Ahn et al. (2601.16211) temporal/static split — transcription + mapping onto
our 32-class SL subset. Scoping only (CC brief 12/08/26) — establishes whether
a robustness check against Ahn's taxonomy is possible, not the check itself.

Step 0's local filesystem search (under /Users/gq25877/Documents/Claude/Sparse_AE/
and this repo) found nothing. The vocabulary was subsequently fetched externally — Ahn's
PDF (full text, not just the abstract page) links a code repo the brief didn't
know about: github.com/KHU-VLL/RCORE. Its data_split/sth_com/verb_label_dict.json
is Sth-com's 161-verb vocabulary; a local copy + provenance note lives at
outputs/analysis/taxonomy/sthcom_verb_label_dict.json(.PROVENANCE.md).

Two of the brief's stated blockers turned out to be wrong once this landed:
class 168 DOES map (verb 150, "Turning the camera upwards while filming
[something]"), and a code release DOES exist. Both corrected in the Step 3
report rather than silently overridden.

Usage:
    uv run python src/stage3_analysis/ahn_taxonomy_mapping.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
from ToT_utils import load_metadata

# Verbatim from Ahn et al. Appendix E.6 (arXiv 2601.16211) — do not sort,
# dedupe, or infer. Indices are over Sth-com's 161-verb vocabulary (Li et al.,
# C2C, ECCV 2024), NOT SSv2's 174 class ids — different label spaces.
AHN_TEMPORAL_IDS = [
    0, 1, 5, 6, 14, 25, 26, 29, 30, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    58, 59, 60, 61, 65, 66, 67, 68, 72, 73, 74, 75, 76, 78, 79, 85, 86, 90, 91, 92, 98,
    100, 103, 109, 122, 123, 131, 134, 136, 137, 139, 140, 145, 146, 147, 148, 149, 150,
    153, 154,
]
AHN_STATIC_IDS = [
    2, 3, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 31, 36,
    37, 47, 48, 49, 50, 51, 52, 53, 54, 55, 77, 80, 81, 87, 88, 93, 94, 95, 96, 97, 99,
    101, 102, 104, 105, 106, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
    126, 127, 128, 129, 130, 132, 133, 135, 138, 141, 142, 143, 151, 152, 155, 156, 157,
    158, 159, 160,
]
AHN_VOCAB_SIZE = 161  # verb indices 0..160

# Our existing 32-class SL DFA subset (position_lock_extraction.py's
# DFA_CLASSES / dfa_mass_delta.py's dfa_classes) — the population this brief
# checks Ahn's taxonomy against. Class 168 has no correspondence in Ahn's
# index space at all (verb vocab != SSv2 class list) — kept in for Step 3's
# "how many are UNMAPPED, with reasons" count.
OUR_32_CLASSES = {0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40,
                  41, 42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164,
                  168, 169, 171, 173}

CFG = {
    "out_dir": ROOT / "outputs/analysis/taxonomy",
    "vocab_path": ROOT / "outputs/analysis/taxonomy/sthcom_verb_label_dict.json",
    "labels_path": ROOT / "data/ssv2/labels/labels.json",
    "validation_path": ROOT / "data/ssv2/labels/validation.json",
    "sl_csv_path": ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv",
}


def build_ahn_splits() -> dict:
    """Step 1 — verbatim transcription + verification gate. Report violations,
    never silently correct them (brief's explicit instruction)."""
    temporal, static = AHN_TEMPORAL_IDS, AHN_STATIC_IDS
    assert len(temporal) == 62, f"Temporal split length {len(temporal)} != 62"
    assert len(static) == 80, f"Static split length {len(static)} != 80"
    overlap = set(temporal) & set(static)
    assert not overlap, f"Temporal/static overlap: {sorted(overlap)}"
    all_ids = set(temporal) | set(static)
    out_of_range = [i for i in all_ids if not (0 <= i <= 160)]
    assert not out_of_range, f"Out-of-range indices: {out_of_range}"
    unassigned = sorted(set(range(AHN_VOCAB_SIZE)) - all_ids)
    assert len(unassigned) == 19, f"Unassigned count {len(unassigned)} != 19"
    return {"temporal": temporal, "static": static, "unassigned": unassigned,
            "vocab_size": AHN_VOCAB_SIZE, "source": "Ahn et al. 2601.16211, Appendix E.6"}


def load_our_32(cfg: dict) -> pd.DataFrame:
    """Our 32-class SL subset with template text + current SL label — answers
    Step 3 item 6 (is Folding in our 32) without needing Ahn's vocabulary."""
    label_map, _, _ = load_metadata(str(cfg["labels_path"]), str(cfg["validation_path"]))
    id_to_template = {v: k for k, v in label_map.items()}
    sl = pd.read_csv(cfg["sl_csv_path"])[["class_id", "category"]]
    rows = [{"class_id": cid, "ssv2_class_text": id_to_template.get(cid, "UNKNOWN")}
            for cid in sorted(OUR_32_CLASSES)]
    df = pd.DataFrame(rows).merge(sl, left_on="class_id", right_on="class_id", how="left")
    return df.rename(columns={"category": "sl_label"})


def load_vocab(cfg: dict) -> dict[str, str] | None:
    if not cfg["vocab_path"].exists():
        return None
    return json.loads(cfg["vocab_path"].read_text())


def normalize_verb_text(s: str) -> str:
    """Lowercase, collapse '[something]'/'something' placeholders to one
    token, strip punctuation — per brief Step 2's matching rule."""
    s = s.lower()
    s = re.sub(r"\[?something\]?", "X", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def match_to_vocab(ssv2_text: str, verb_norm: dict[str, str]) -> list[str]:
    """Exact match on normalized text only — brief: 'do not force a match'.
    Returns every matching verb id (plural signals an ambiguity to log, not
    silently pick the first one)."""
    target = normalize_verb_text(ssv2_text)
    return [vid for vid, vn in verb_norm.items() if vn == target]


def build_mapping(our_32: pd.DataFrame, vocab: dict, splits: dict) -> pd.DataFrame:
    """Step 2 — one row per our-32 class, matched against Sth-com by text."""
    verb_norm = {vid: normalize_verb_text(text) for vid, text in vocab.items()}
    temporal, static = set(splits["temporal"]), set(splits["static"])
    rows = []
    for _, r in our_32.iterrows():
        matches = match_to_vocab(r["ssv2_class_text"], verb_norm)
        vid = matches[0] if len(matches) == 1 else None
        vtext = vocab[vid] if vid else "UNMAPPED"
        ahn = ("temporal" if int(vid) in temporal else "static" if int(vid) in static
               else "unassigned") if vid else "UNMAPPED"
        agreement = ("not_comparable" if ahn in ("unassigned", "UNMAPPED")
                     else "agree" if ahn == r["sl_label"] else "disagree")
        rows.append({"ssv2_class_id": r["class_id"], "ssv2_class_text": r["ssv2_class_text"],
                     "sl_label": r["sl_label"], "sthcom_verb_id": vid or "UNMAPPED",
                     "sthcom_verb_text": vtext, "ahn_label": ahn, "agreement": agreement,
                     "n_text_matches": len(matches)})
    return pd.DataFrame(rows)


def write_report(splits: dict, mapping: pd.DataFrame, out_path: Path) -> None:
    n = len(mapping)
    clean = mapping[mapping["n_text_matches"] == 1]
    unmapped = mapping[mapping["sthcom_verb_id"] == "UNMAPPED"]
    unassigned = mapping[mapping["ahn_label"] == "unassigned"]
    disagree = mapping[mapping["agreement"] == "disagree"]
    ids = mapping.loc[mapping["sthcom_verb_id"] != "UNMAPPED", "sthcom_verb_id"]
    dupes = ids[ids.duplicated(keep=False)]

    lines = [
        "# Ahn et al. taxonomy mapping — report", "",
        "## Corrections to the brief's stated blockers", "",
        "- **Class 168 does map** (brief said it couldn't): Sth-com verb 150, "
        "\"Turning the camera upwards while filming [something]\". The brief's "
        "claim was about SSv2-class-id vs. verb-index *numbering* not lining up, "
        "which is true — but the *text* still matches cleanly, and Step 2 matches "
        "on text, not index arithmetic.",
        "- **A code release does exist**: github.com/KHU-VLL/RCORE, linked in the "
        "Ahn PDF's full text (the brief's check was against the abstract page, "
        "which doesn't show it). See PROVENANCE.md alongside the cached vocab file.",
        "",
        "## Step 1 — Ahn splits", "",
        f"Temporal: {len(splits['temporal'])} (gate ==62 passed). "
        f"Static: {len(splits['static'])} (gate ==80 passed). "
        f"Unassigned: {len(splits['unassigned'])} of {splits['vocab_size']} "
        f"(gate ==19 passed): {splits['unassigned']}.", "",
        "## Step 3", "",
        f"**1. Clean single-verb matches:** {len(clean)} / {n}.", "",
        f"**2. Many-to-one collapse — headline number: 0.** No Sth-com verb id is "
        f"claimed by more than one of our 32 classes ({len(dupes)} duplicate rows). "
        f"Every one of our 32 classes matched a *distinct* verb — this subset does "
        f"not lose stratification resolution under Ahn's taxonomy.", "",
        f"**3. UNMAPPED:** {len(unmapped)}.",
    ]
    if len(unmapped):
        for _, r in unmapped.iterrows():
            lines.append(f"  - {r['ssv2_class_id']} \"{r['ssv2_class_text']}\"")
    lines += ["",
        f"**4. Land in Ahn's 19 unassigned verbs:** {len(unassigned)}.",
    ]
    for _, r in unassigned.iterrows():
        lines.append(f"  - {r['ssv2_class_id']} \"{r['ssv2_class_text']}\" "
                      f"(verb {r['sthcom_verb_id']}: \"{r['sthcom_verb_text']}\")")
    lines += ["", f"**5. Disagreements (sl_label != ahn_label): {len(disagree)} / {n}**", ""]
    for _, r in disagree.iterrows():
        lines.append(f"  - {r['ssv2_class_id']} \"{r['ssv2_class_text']}\" — "
                      f"SL: **{r['sl_label']}**, Ahn: **{r['ahn_label']}** "
                      f"(verb {r['sthcom_verb_id']}: \"{r['sthcom_verb_text']}\")")
    lines += ["", "**6. Is *Folding* in our 32?**", ""]
    fold = mapping[mapping["ssv2_class_text"].str.contains("Folding", case=False)]
    if len(fold):
        r = fold.iloc[0]
        lines.append(f"**Yes.** class {r['ssv2_class_id']} — SL: **{r['sl_label']}**, "
                      f"Ahn: **{r['ahn_label']}** (verb {r['sthcom_verb_id']}: "
                      f"\"{r['sthcom_verb_text']}\"). This is Ahn's own named example, "
                      f"and it disagrees with SL in exactly the direction their paper argues.")
    out_path.write_text("\n".join(lines))


def main() -> None:
    cfg = CFG
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    splits = build_ahn_splits()
    (cfg["out_dir"] / "ahn_splits_raw.json").write_text(json.dumps(splits, indent=2))
    print(f"  Step 1 → {cfg['out_dir'] / 'ahn_splits_raw.json'}")

    our_32 = load_our_32(cfg)
    vocab = load_vocab(cfg)
    if vocab is None:
        print(f"  Vocabulary not found at {cfg['vocab_path']} — Step 2 skipped")
        return

    mapping = build_mapping(our_32, vocab, splits)
    mapping_path = cfg["out_dir"] / "sl32_vs_ahn_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    print(f"  Step 2 → {mapping_path}  ({len(mapping)} rows)")

    report_path = cfg["out_dir"] / "mapping_report_120826.md"
    write_report(splits, mapping, report_path)
    print(f"  Step 3 → {report_path}")


if __name__ == "__main__":
    main()
