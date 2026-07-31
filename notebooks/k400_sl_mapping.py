"""
Match Sevilla-Lara K400 temporal/static class names (arXiv:1907.08340v2 appendix)
against our K400 val.csv label space and the finetuned model's label2id.
Persists a hand-verifiable candidate 64-class mapping table (name-matching only).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from transformers import AutoConfig

ROOT = Path(__file__).parent.parent

# SL names verified directly against arXiv:1907.08340v2 Appendix (fetched + read 31/07/26).
# Reference indices are for hand-verification only — NEVER match on them (see 14/06 SSv2
# incident: SL index 63 mapped to two contradictory labels in their own appendix).
SL_TEMPORAL = [
    (30, "bouncing on trampoline"), (34, "breakdancing"), (41, "busking"),
    (45, "cartwheeling"), (63, "cleaning shoes"), (75, "country line dancing"),
    (105, "drop kicking"), (147, "gymnastics tumbling"), (148, "hammer throw"),
    (152, "high kick"), (173, "jumpstyle dancing"), (177, "kitesurfing"),
    (206, "parasailing"), (222, "playing cards"), (228, "playing cymbals"),
    (230, "playing drums"), (235, "playing ice hockey"), (277, "robot dancing"),
    (295, "shining shoes"), (301, "shuffling cards"), (302, "side kick"),
    (307, "ski jumping"), (308, "skiing (not slalom or crosscountry)"),
    (309, "skiing crosscountry"), (310, "skiing slalom"), (322, "snowboarding"),
    (325, "somersaulting"), (349, "tap dancing"), (357, "throwing ball"),
    (358, "throwing discus"), (376, "vault"), (395, "wrestling"),
]

# Static list CORRECTED vs the brief: the brief's transcription dropped "moving furniture"
# (idx 200), which shifted every subsequent name-index pairing by one. Restored from the
# PDF appendix text directly — this also resolves 397/398 cleanly (yawning, yoga; no
# ambiguity, no missing name). Flag to Tom: brief's static list should be replaced by this.
SL_STATIC = [
    (18, "belly dancing"), (20, "bending back"), (23, "blasting sand"),
    (26, "blowing nose"), (53, "changing wheel"), (57, "clapping"),
    (80, "curling hair"), (88, "deadlifting"), (91, "dining"),
    (95, "doing aerobics"), (99, "dribbling basketball"), (113, "eating doughnuts"),
    (126, "filling eyebrows"), (139, "getting a tattoo"), (181, "laying bricks"),
    (182, "long jump"), (183, "lunge"), (186, "making bed"),
    (200, "moving furniture"), (201, "mowing lawn"), (210, "peeling apples"),
    (218, "playing badminton"), (226, "playing controller"), (227, "playing cricket"),
    (255, "pull ups"), (268, "riding camel"), (298, "shot put"),
    (354, "testifying"), (366, "trimming trees"), (388, "waxing eyebrows"),
    (397, "yawning"), (398, "yoga"),
]

CFG = {
    "val_csv": ROOT / "data" / "kinetics400" / "annotations" / "val.csv",
    "checkpoint": "MCG-NJU/videomae-base-finetuned-kinetics",
    "output_path": ROOT / "outputs" / "Laura_SL" / "k400_sl_class_mapping.csv",
}


def normalise(name: str) -> str:
    # Kinetics parentheticals are class-distinguishing (e.g. the three "skiing" variants) —
    # deliberately do NOT strip them, unlike the SSv2 [bracket]-stripping convention.
    return re.sub(r"\s+", " ", name.strip().lower())


def load_csv_label_index(csv_path: Path) -> dict[str, list[str]]:
    """normalised label -> list of original CSV label strings sharing that normalisation."""
    df = pd.read_csv(csv_path)
    index: dict[str, list[str]] = defaultdict(list)
    for label in df["label"].dropna().unique():
        index[normalise(label)].append(label)
    return index


def load_label2id(checkpoint: str) -> dict[str, int]:
    cfg = AutoConfig.from_pretrained(checkpoint)
    return {normalise(k): v for k, v in cfg.label2id.items()}


def match_one(
    sl_idx: int, sl_name: str, category: str,
    csv_index: dict[str, list[str]], label2id: dict[str, int],
) -> dict:
    candidates = csv_index.get(normalise(sl_name), [])
    if len(candidates) == 0:
        csv_label, status = None, "unmatched"
    elif len(candidates) == 1:
        csv_label, status = candidates[0], "exact"
    else:
        csv_label, status = None, "ambiguous"  # never auto-resolve — flag for Tom
        print(f"  AMBIGUOUS: '{sl_name}' matched CSV labels {candidates!r}")

    model_id = label2id.get(normalise(csv_label)) if csv_label else None
    if csv_label and model_id is None:
        print(f"  NOT IN label2id: matched CSV label '{csv_label}' (SL '{sl_name}')")

    return {
        "sl_reference_idx": sl_idx,
        "sl_name": sl_name,
        "sl_category": category,
        "matched_csv_label": csv_label,
        "matched_model_class_id": model_id,
        "match_status": status,
    }


def main() -> None:
    print(f"Loading CSV labels from {CFG['val_csv']}")
    csv_index = load_csv_label_index(CFG["val_csv"])
    print(f"Loading label2id from {CFG['checkpoint']}")
    label2id = load_label2id(CFG["checkpoint"])

    rows = [
        match_one(idx, name, category, csv_index, label2id)
        for category, entries in [("temporal", SL_TEMPORAL), ("static", SL_STATIC)]
        for idx, name in entries
    ]
    df = pd.DataFrame(rows)

    CFG["output_path"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CFG["output_path"], index=False)  # persist before any downstream use

    with pd.option_context("display.max_rows", None, "display.width", 120):
        print("\n" + df.to_string(index=False))

    print("\nCounts by category / match_status:")
    print(df.groupby(["sl_category", "match_status"]).size().to_string())
    print(f"\nSaved -> {CFG['output_path']}")


if __name__ == "__main__":
    main()
