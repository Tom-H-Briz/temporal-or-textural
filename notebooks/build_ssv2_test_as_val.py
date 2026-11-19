"""SSv2 test set in validation.json format (19/11/26): [{"id", "template"}] from test-answers.csv, so drivers that
read VAL_JSON run unchanged on held-out clips (unseen by backbone fine-tuning, SAE training and feature selection).
No placeholders (test has no object labels), so verb classes only.
Usage: uv run python notebooks/build_ssv2_test_as_val.py"""
import csv
import json
from pathlib import Path

CFG = {"labels_dir": Path(__file__).resolve().parents[1] / "data/ssv2/labels", "out": "test_as_val.json"}


def main(cfg):
    d = cfg["labels_dir"]
    rows = [{"id": r[0], "template": r[1]} for r in csv.reader(open(d / "test-answers.csv"), delimiter=";")]
    json.dump(rows, open(d / cfg["out"], "w"))
    print(f"{len(rows)} test clips -> {d / cfg['out']}")


if __name__ == "__main__":
    main(CFG)
