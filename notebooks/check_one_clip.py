"""
Loads one real SSv2 clip, runs it through the model, and prints the top prediction
+ its logit — for comparing behavior between transformers versions directly
(uv run --with "transformers==X" python notebooks/check_one_clip.py) rather than
just checking the load report.

Usage:
    uv run python notebooks/check_one_clip.py
    uv run --with "transformers==5.5.0" python notebooks/check_one_clip.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import CHECKPOINT_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY, load_metadata, sample_frames_ssv2


def main() -> None:
    model_cfg  = MODEL_REGISTRY["videomae"]
    checkpoint = CHECKPOINT_REGISTRY[("videomae", "ssv2")]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint).eval()

    d = DATASET_REGISTRY["ssv2"]
    label_map, clips, id2label = load_metadata(str(d["labels_path"]), str(d["validation_path"]))
    video_dir = Path(d["video_dir"])
    clip = next(c for c in clips if (video_dir / f"{c['id']}.webm").exists())
    path = video_dir / f"{clip['id']}.webm"
    print(f"Clip: {path.name}  true label: {clip['template']!r}")

    import av
    container = av.open(str(path))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    idx = sample_frames_ssv2(len(frames), model_cfg["num_frames"])
    sampled = [frames[i] for i in idx]
    pixel_values = processor(sampled, return_tensors="pt")["pixel_values"]

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits[0]
    pred_id = logits.argmax().item()
    probs = torch.softmax(logits, dim=-1)
    print(f"Predicted: {id2label[pred_id]!r}  logit={logits[pred_id]:.4f}  prob={probs[pred_id]:.4f}")
    true_id = label_map[clip["template"].replace("[", "").replace("]", "")]
    print(f"True-label logit={logits[true_id]:.4f}  prob={probs[true_id]:.4f}  rank="
          f"{(logits > logits[true_id]).sum().item() + 1}/{len(logits)}")


if __name__ == "__main__":
    main()
