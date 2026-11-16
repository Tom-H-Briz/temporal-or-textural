"""
Synthetic-motion MLM (16/11/26): the motion_vid_generator clips through UMT's refit MLM head, unedited and with
the L5 2x2 arms (clock / builders incl. #5519 / both; raw stream x' = x5 - sum z_f d_f). Captions: SSv2
direction templates (every word masked alone -> log p(true word), top-5 fills) and a noun list in the object
slot (log p(noun)). Output: per_word.parquet. Usage: uv run python -u src/stage3_analysis/umt_synthetic_motion_mlm.py
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
from umt_mlm_group_ablation import CFG as GA_CFG

SETS = json.load(open(ROOT / "src/stage3_analysis/umt_clock_downstream_sets.json"))["5"]
CFG = dict(GA_CFG, layer=5,
           clip_dir=ROOT / "outputs/synthetic_motion/directional_clips",
           out_dir=Path(os.environ.get("OUT_DIR", str(ROOT / "outputs/analysis/umt_synthetic_motion_mlm"))),
           builders=[5135, 4638, 186, 5519],
           templates=["Moving something down", "Moving something up",
                      "Turning the camera upwards while filming something",
                      "Turning the camera downwards while filming something"],
           nouns=["ball", "box", "laptop", "book", "phone", "cup", "pen", "dvd", "pyramid", "triangle",
                  "cube", "paper", "bottle", "remote"])


def edits(cfg):
    clock = SETS["clock"]
    return [("none", []), ("clock", clock), ("builders", cfg["builders"]), ("both", clock + cfg["builders"])]


def edited_towers(cfg, s, pv):
    """{edit: final vision tokens (1, 2352, 768)}: tower to L5, raw-stream edit, blocks 6-11 + norm."""
    from umt_bridge_phase3 import apply_edit, finish_tower, tower_to_layer
    x5 = tower_to_layer(s["model"], pv.to(cfg["device"]), layer=cfg["layer"])
    with torch.no_grad():
        _, z0 = s["sae"].encode((x5[0] - s["dim_mean"]).float())
        D = s["sae"].get_dictionary()
        return {name: finish_tower(s["model"], apply_edit(x5, z0, D, fs, dose=1.0) if fs else x5,
                                   layer=cfg["layer"])[0].unsqueeze(0)
                for name, fs in edits(cfg)}


def score(cfg, s, cap, vis, k=5):
    """Every masked variant of one caption on vision tokens vis -> (log p(true word), top-k fills) per variant."""
    from umt_mlm_refit import mlm_fusion, mlm_text
    with torch.no_grad():
        h8, add = mlm_text(cap["ids"], cap["attn"], s["w"])
        h = mlm_fusion(h8, add, vis, s["w"])
        lp = s["head"](h[torch.arange(len(cap["pos"])), torch.tensor(cap["pos"])]).log_softmax(-1)
    true = lp.gather(1, torch.tensor(cap["true"], device=lp.device)[:, None])[:, 0]
    tops = [" ".join(s["tok"].convert_ids_to_tokens(r.tolist())) for r in lp.topk(k, dim=-1).indices]
    return true.cpu().tolist(), tops


def captions(cfg, s):
    """[(template, noun, variants)]: each template with 'something' and with each noun in the object slot;
    every word masked alone (the refit head learned real captions with real nouns, so the direction word is
    scored in noun captions too)."""
    from umt_mlm_class_wordmap import all_word_variants
    return [(t, n, all_word_variants(cfg, s, t.replace("something", n), [n]))
            for t in cfg["templates"] for n in ["something"] + cfg["nouns"]]


def main(cfg):
    from umt_mlm_feature_contrast import setup
    meta = json.load(open(cfg["clip_dir"] / "metadata.json"))[: cfg["max_clips"] or None]
    batch = torch.load(cfg["clip_dir"] / "all_umt_clips.pt")              # (N, T, C, H, W)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    s = setup(cfg)
    caps, rows = captions(cfg, s), []
    for i, m in enumerate(meta):
        for edit, vis in edited_towers(cfg, s, batch[i:i + 1]).items():
            for tpl, noun, cap in caps:
                lp, tops = score(cfg, s, cap, vis)
                rows += [dict(clip=m["name"], edit=edit, template=tpl, noun=noun, position=p, word=w, role=r,
                              lp=l, top5=t) for p, w, r, l, t in zip(cap["pos"], cap["word"], cap["role"], lp, tops)]
        print(f"  {i + 1}/{len(meta)} {m['name']}", flush=True)
    pd.DataFrame(rows).to_parquet(cfg["out_dir"] / "per_word.parquet", index=False)
    print(f"Saved {len(rows)} rows -> {cfg['out_dir']}", flush=True)


if __name__ == "__main__":
    main(CFG)
