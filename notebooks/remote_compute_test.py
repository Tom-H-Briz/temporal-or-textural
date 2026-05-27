"""
Remote compute test for Isambard.

Loads VideoMAE, decodes 5 real SSv2 videos from disk, runs inference,
and validates the same shape checks as the local smoke test.

Usage (on Isambard):
    uv run python notebooks/remote_compute_test.py
"""

import time
import sys
import numpy as np
import torch
import av
from pathlib import Path
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# --- config ---
MODEL_ID = "MCG-NJU/videomae-base-finetuned-ssv2"
VIDEO_DIR = Path("/scratch/b5bg/tomheslin83.b5bg/videos")
OUTPUT_DIR = Path(__file__).parent / "outputs"
VIDEO_IDS = [1, 2, 3, 4, 5]
NUM_FRAMES = 16
HOOK_LAYERS = [5, 7, 9, 11]
EXPECTED_LOGITS_SHAPE = (1, 174)
EXPECTED_ACT_SHAPE = (1, 1568, 768)


def find_video(video_dir: Path, video_id: int) -> Path | None:
    for ext in (".webm", ".mp4", ".avi", ".mkv"):
        p = video_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def decode_frames(video_path: Path, num_frames: int) -> list[np.ndarray]:
    """Uniformly sample `num_frames` RGB frames from a video file."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total = stream.frames or 0

    frames_raw = []
    for frame in container.decode(video=0):
        frames_raw.append(frame.to_ndarray(format="rgb24"))

    container.close()

    if len(frames_raw) == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    indices = np.linspace(0, len(frames_raw) - 1, num_frames, dtype=int)
    return [frames_raw[i] for i in indices]


def run_video(model, processor, frames: list[np.ndarray]) -> tuple[torch.Tensor, dict]:
    activations: dict[int, torch.Tensor] = {}
    hooks = []

    for layer_idx in HOOK_LAYERS:
        def _make_hook(idx: int):
            def _hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activations[idx] = hidden.detach()
            return _hook
        hooks.append(
            model.videomae.encoder.layer[layer_idx].register_forward_hook(_make_hook(layer_idx))
        )

    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    for h in hooks:
        h.remove()

    return outputs.logits, activations


def check_shapes(logits: torch.Tensor, activations: dict) -> dict:
    results = {}
    logits_shape = tuple(logits.shape)
    results["logits_shape"] = {
        "pass": logits_shape == EXPECTED_LOGITS_SHAPE,
        "expected": EXPECTED_LOGITS_SHAPE,
        "got": logits_shape,
    }
    for layer_idx in HOOK_LAYERS:
        captured = activations.get(layer_idx)
        act_shape = tuple(captured.shape) if captured is not None else None
        results[f"layer_{layer_idx}_activations"] = {
            "pass": act_shape == EXPECTED_ACT_SHAPE,
            "expected": EXPECTED_ACT_SHAPE,
            "got": act_shape,
        }
    return results


def main() -> bool:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "remote_compute_test.txt"

    def log(msg: str = "") -> None:
        print(msg)
        out_file.write(msg + "\n")

    with open(out_path, "w") as out_file:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Device: {device}")
        log(f"Loading model: {MODEL_ID}")

        t0 = time.perf_counter()
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
        model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
        model.to(device)
        model.eval()
        log(f"Model loaded in {time.perf_counter() - t0:.1f}s\n")

        all_pass = True
        per_video_results = {}

        for vid_id in VIDEO_IDS:
            path = find_video(VIDEO_DIR, vid_id)
            if path is None:
                log(f"[SKIP] video {vid_id}: not found in {VIDEO_DIR}")
                per_video_results[vid_id] = None
                continue

            log(f"Video {vid_id}: {path.name}")
            try:
                t1 = time.perf_counter()
                frames = decode_frames(path, NUM_FRAMES)
                log(f"  decoded {len(frames)} frames ({frames[0].shape}) in {time.perf_counter() - t1:.2f}s")

                t2 = time.perf_counter()
                logits, activations = run_video(model, processor, frames)
                log(f"  inference in {time.perf_counter() - t2:.2f}s")

                results = check_shapes(logits, activations)
                per_video_results[vid_id] = results

                top5 = logits[0].topk(5)
                top5_ids = top5.indices.tolist()
                top5_scores = top5.values.tolist()
                labels: dict = model.config.id2label or {}
                log(f"  top-5 predictions:")
                for rank, (idx, score) in enumerate(zip(top5_ids, top5_scores), 1):
                    log(f"    {rank}. {labels.get(idx, str(idx))} ({score:.3f})")

            except Exception as e:
                log(f"  ERROR: {e}")
                per_video_results[vid_id] = None
                all_pass = False

        log("\n=== Remote Compute Test Summary ===")
        for vid_id, results in per_video_results.items():
            if results is None:
                log(f"  [SKIP] video {vid_id}")
                continue
            for name, result in results.items():
                status = "PASS" if result["pass"] else "FAIL"
                if not result["pass"]:
                    all_pass = False
                    log(f"  [{status}] video {vid_id} / {name}: expected {result['expected']}, got {result['got']}")
                else:
                    log(f"  [{status}] video {vid_id} / {name}: {result['got']}")

        log(f"\nResult: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    print(f"\nOutput written to: {out_path}")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
