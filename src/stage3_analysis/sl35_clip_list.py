"""SL-35 clip list for the PE characterisation runs (25/11/26): every manifest clip of the 35 Sevilla-Lara classes
(Static_Temporal.csv x manifest cap40), not gated on R-correct; same list as the 21/10 VTC clock ablation.
Out: outputs/analysis/sl35_clips.csv (clip_id, class_id, sl_label). Pass it on as CLIP_CSV=outputs/analysis/sl35_clips.csv.
Usage: uv run python src/stage3_analysis/sl35_clip_list.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "stage3_analysis")]
from umt_vtc_clock_ablation import CFG as VCFG, clip_table

CFG = dict(VCFG, max_clips=0, out=ROOT / "outputs/analysis/sl35_clips.csv")

if __name__ == "__main__":
    d = clip_table(CFG)
    d.to_csv(CFG["out"], index=False)
    print(len(d), "clips,", d.class_id.nunique(), "classes:", d.sl_label.value_counts().to_dict())
