"""
Quick baseline-only accuracy check on a random clip sample — calls the exact same
run_spliced_accuracy pipeline used for the real spliced-accuracy runs (baseline_only
skips SAE/dim_mean loading and the splice pass), so this number is directly
comparable rather than a separately-reimplemented approximation of it.

Model- and dataset-agnostic — pass --model-name / --dataset-name. Writes a
baseline report (accuracy, threshold, pass/fail, sampler frame indices for 3
clips), persists the eval-clip sample for the spliced-accuracy job to reuse, and
exits non-zero below threshold so a SLURM dependency chain gates on it.

Usage:
    uv run python notebooks/check_baseline_accuracy.py --model-name timesformer \
        --dataset-name kinetics400 --n-clips 3000 --threshold 0.723
    uv run python notebooks/check_baseline_accuracy.py --dataset-name ssv2 --n-clips 3000
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import av

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import CHECKPOINT_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY, get_frame_sampler
from spliced_accuracy_vm import run_spliced_accuracy


def log_frame_indices(clip_names: list[str], video_dir: Path, model_name: str,
                      dataset_name: str, n_log: int = 3) -> list[str]:
    """Record the exact frame indices the sampler picks for n_log sample clips —
    decoded and resolved through the same get_frame_sampler the eval loop uses,
    not a reimplementation (Gate 4 audit item)."""
    model_cfg = MODEL_REGISTRY[model_name]
    sampler   = get_frame_sampler(dataset_name, model_cfg)
    lines = []
    for name in clip_names[:n_log]:
        with av.open(str(video_dir / name)) as container:
            n = sum(1 for _ in container.decode(video=0))
        lines.append(f"- {name}: {n} decoded frames -> indices {sampler(n, model_cfg['num_frames'])}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="videomae")
    parser.add_argument("--dataset-name", type=str, default="kinetics400")
    parser.add_argument("--n-clips", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, required=True,
                        help="gate: exit non-zero if accuracy < threshold (paper top-1 minus 5pp)")
    parser.add_argument("--report-dir", type=str, default=str(ROOT / "outputs" / "tf_k400"))
    args = parser.parse_args()

    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(os.environ.get("VIDEO_DIR") or DATASET_REGISTRY[args.dataset_name]["video_dir"])
    all_clips = sorted(p.name for ext in ("*.mp4", "*.webm", "*.avi") for p in video_dir.glob(ext))
    rng = random.Random(args.seed)
    eval_clips = rng.sample(all_clips, min(args.n_clips, len(all_clips)))
    print(f"Sampled {len(eval_clips):,} of {len(all_clips):,} clips (seed={args.seed}) — not the "
          f"held-out validation split, a separate random draw for a quick pre-run check")

    sample_path = report_dir / f"eval_clips_{len(eval_clips)}.json"
    sample_path.write_text(json.dumps(eval_clips))
    print(f"Eval-clip sample persisted -> {sample_path}  (the spliced-accuracy job reuses exactly this set)")

    result = run_spliced_accuracy(
        model_name=args.model_name, dataset_name=args.dataset_name,
        eval_clips=eval_clips, baseline_only=True, return_per_clip=True,
    )
    accuracy = result["baseline_accuracy_clip_weighted"]
    result["per_clip_df"].to_csv(report_dir / "baseline_per_clip.csv", index=False)
    passed = accuracy >= args.threshold

    checkpoint = CHECKPOINT_REGISTRY[(args.model_name, args.dataset_name)]
    frame_lines = log_frame_indices(eval_clips, video_dir, args.model_name, args.dataset_name)
    report = [
        "# Baseline gate report", "",
        f"- model: `{args.model_name}`  checkpoint: `{checkpoint}`",
        f"- dataset: `{args.dataset_name}`  n_clips: {len(eval_clips):,}  seed: {args.seed}",
        f"- baseline accuracy (clip-weighted): **{accuracy:.4f}**",
        f"- threshold: {args.threshold:.4f}  ->  {'**PASS**' if passed else '**FAIL**'}", "",
        "## Sampler frame indices (3 clips)", *frame_lines,
    ]
    (report_dir / "baseline_report.md").write_text("\n".join(report) + "\n")
    print(f"Report -> {report_dir / 'baseline_report.md'}")
    if not passed:
        raise SystemExit(f"Baseline {accuracy:.4f} below threshold {args.threshold:.4f} — gate FAILED")


if __name__ == "__main__":
    main()
