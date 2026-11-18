"""Single-feature MLM word map at any layer (16/11/26, first use: L5 #5216). Which caption words does ablating
one SAE latent hit? umt_mlm_class_wordmap's method: every caption word-piece masked alone, dCE = -d log p(true
word), the feature alone vs one mass-matched control set; spliced by default (SPLICE=0: raw stream).
Usage: LAYER=5 FEATURE=5216 CLASSES=43,6,41,45 uv run python -u src/stage3_analysis/umt_mlm_feature_wordmap.py
       LAYER=5 FEATURE=4638 NOUN=ball MAX_CLIPS=30 ... (noun clips, out L5_f4638_ball_splice)
"""
import os, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
from umt_mlm_group_ablation import CFG as GA_CFG, load_validation
from umt_mlm_class_wordmap import all_word_variants, build_edits, clip_rows, encode_clips
from umt_mlm_feature_contrast import setup as contrast_setup

CFG = dict(GA_CFG, layer=int(os.environ.get("LAYER", 5)), feature=int(os.environ.get("FEATURE", 5216)),
           classes=[int(c) for c in os.environ.get("CLASSES", "43,6,41,45").split(",")],
           splice=os.environ.get("SPLICE", "1") == "1", num_workers=0,
           dfa=ROOT / "outputs/analysis/dfa_mass_delta_umt/dfa_mass_delta_umt_ssv2_all174_thrfix_l9_job7ep_k64.parquet")
CFG["noun"] = os.environ.get("NOUN", "")   # 18/11/26: clips = the noun's object-contrast pos + test clips
CFG["obj_sets"] = ROOT / "outputs/analysis/umt_object_contrast/sets.csv"
CFG["out_dir"] = ROOT / (f"outputs/analysis/umt_mlm_feature_wordmap/L{CFG['layer']}_f{CFG['feature']}"
                         f"{'_' + CFG['noun'] if CFG['noun'] else ''}{'_splice' if CFG['splice'] else '_raw'}")


def main(cfg):
    t0 = time.time()
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    if cfg["noun"]:                                  # MAX_CLIPS = total object clips
        s = pd.read_csv(cfg["obj_sets"], dtype={"clip_id": str}).query(f"object == '{cfg['noun']}' and role != 'neg'")
        d = s.rename(columns={"true_class": "class_id"}).drop_duplicates("clip_id").head(cfg["max_clips"] or None)
    else:
        d = pd.read_parquet(cfg["dfa"], columns=["clip_id", "class_id"])
        d = d[d.class_id.isin(cfg["classes"])].groupby("class_id").head(cfg["max_clips"] or None)   # MAX_CLIPS per class (18/11/26)
    rows = pd.DataFrame(dict(clip_id=d.clip_id.astype(str).values, class_id=d.class_id.values))
    s = dict(contrast_setup(cfg), splice=cfg["splice"], layer=cfg["layer"])   # layer: edit_logits finishes from L
    enc, val, D = encode_clips(cfg, s, rows), load_validation(cfg), s["sae"].get_dictionary()
    f = cfg["feature"]
    edits = [e for e in build_edits([f], enc) if e[0] != "top"]              # single_f + mass-matched control
    out = []
    for i, (x, z0) in sorted(enc.items()):
        cid = rows.clip_id[i]
        r = clip_rows(cfg, s, cid, all_word_variants(cfg, s, *val[cid]), x, z0, D, edits)
        out += [dict(o, class_id=int(rows.class_id[i]), act=float(z0[:, f].sum())) for o in r]
        print(f"  {cid} ({rows.class_id[i]}) done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(out)
    df.to_parquet(cfg["out_dir"] / "per_word.parquet", index=False)
    print(df.groupby(["class_id", "edit", "role"]).dce.mean().unstack().round(3).to_string())
    print(f"done {(time.time() - t0) / 60:.1f} min -> {cfg['out_dir']}")


if __name__ == "__main__":
    main(CFG)
