"""
All-174-class SSv2 val manifest for the template-word-overlap DFA run: up to CAP clips per
class, seeded sample from validation.json (sorted by id first, so identical on any machine).
Same schema as manifest_SL_subset.json, read by dfa_mass_delta_vm.py --manifest.

Stdlib only — runs on the Isambard login node:
    VALIDATION_PATH=$HOME/labels/validation.json python3 notebooks/build_all174_manifest.py
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
CFG = {
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "cap":             40,
    "seed":            42,
    "out_path":        ROOT / "outputs/manifests/manifest_ssv2_all174_cap40.json",
}


def main() -> None:
    by_class = defaultdict(list)
    for c in json.load(open(CFG["validation_path"])):
        by_class[c["template"].replace("[", "").replace("]", "")].append(c["id"])
    rng, out = random.Random(CFG["seed"]), []
    for template in sorted(by_class):
        ids = sorted(by_class[template], key=int)
        out += [{"id": i, "template": template} for i in rng.sample(ids, min(CFG["cap"], len(ids)))]
    CFG["out_path"].parent.mkdir(parents=True, exist_ok=True)
    json.dump({"all": out}, open(CFG["out_path"], "w"))
    print(f"{len(by_class)} classes, {len(out)} clips -> {CFG['out_path']}")


if __name__ == "__main__":
    main()
