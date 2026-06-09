"""
Sanity check: compare top firing features between 8x and 16x SAE checkpoints.

Stdout only. Picks the first 3 clips from the validation set.
"""

import sys
from pathlib import Path

import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from sae import BatchTopKSAE
from ToT_utils import load_metadata, _strip_brackets
from stage3_analysis.dfa_engine import _preprocess_videomae

SAE_8X_PATH  = ROOT / "outputs/sae/sae_layer7_job128.pt"
SAE_16X_PATH = ROOT / "outputs/sae/sae_layer7_job128_16x.pt"
DIM_MEAN_PATH = ROOT / "outputs/sae/layer7_dim_mean.pt"
LABELS_PATH     = ROOT / "data/ssv2/labels/labels.json"
VALIDATION_PATH = ROOT / "data/ssv2/labels/validation.json"
VIDEO_DIR       = ROOT / "data/ssv2/20bn-something-something-v2"
MODEL_ID = "MCG-NJU/videomae-base-finetuned-ssv2"
LAYER    = 7
TOP_N    = 15  # top features to print per SAE
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def load_sae(path: Path, dim_mean: torch.Tensor) -> BatchTopKSAE:
    ckpt = torch.load(path, weights_only=True, map_location=DEVICE)
    if isinstance(ckpt, dict) and "sae_state_dict" in ckpt:
        ckpt = ckpt["sae_state_dict"]
    nb_concepts = ckpt["dictionary._weights"].shape[0]
    k_total = ckpt["dictionary._weights"].shape[0] // 6  # rough default; overridden below
    # Infer k from top_k stored in running_threshold context — use nb_concepts / 48 heuristic
    # Actually just use 128*1568 for both (same k)
    sae = BatchTopKSAE(input_shape=768, nb_concepts=nb_concepts, top_k=128*1568, device=DEVICE)
    sae.load_state_dict(ckpt)
    sae.train()
    dummy = torch.zeros(1568, 768, device=DEVICE)
    with torch.no_grad():
        sae.encode((dummy - dim_mean).float())
    sae.eval()
    return sae


def print_sae_dims(label: str, path: Path) -> None:
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    if isinstance(ckpt, dict) and "sae_state_dict" in ckpt:
        sd = ckpt["sae_state_dict"]
        epoch = ckpt.get("epoch", "?")
    else:
        sd = ckpt
        epoch = "?"
    w = sd["dictionary._weights"]
    print(f"  {label}: dictionary._weights={tuple(w.shape)}  (nb_concepts={w.shape[0]}, hidden={w.shape[1]})  epoch={epoch}")


def top_features(z: torch.Tensor, n: int) -> list[tuple[int, float]]:
    """Mean activation per feature across tokens, top-n by magnitude."""
    mean_act = z.mean(dim=0)  # (nb_concepts,)
    vals, idxs = mean_act.topk(n)
    return [(int(i), float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]


def main() -> None:
    print("=" * 60)
    print("SAE DIMENSIONS")
    print_sae_dims("8x  (job128)", SAE_8X_PATH)
    print_sae_dims("16x (job128_16x)", SAE_16X_PATH)
    print()

    dim_mean = torch.load(DIM_MEAN_PATH, weights_only=True).to(DEVICE)

    print("Loading SAEs...")
    sae_8x  = load_sae(SAE_8X_PATH,  dim_mean)
    sae_16x = load_sae(SAE_16X_PATH, dim_mean)
    print(f"  SAE-8x:  nb_concepts={sae_8x.nb_concepts}")
    print(f"  SAE-16x: nb_concepts={sae_16x.nb_concepts}")
    print()

    print("Loading model...")
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook_store: dict = {}
    def _hook(module, inp, out):
        hook_store["acts"] = (out[0] if isinstance(out, tuple) else out).detach()
    model.videomae.encoder.layer[LAYER].register_forward_hook(_hook)

    _, clips, _ = load_metadata(str(LABELS_PATH), str(VALIDATION_PATH))
    clips_to_run = clips[:3]

    print("=" * 60)
    for clip_info in clips_to_run:
        clip_id   = str(clip_info["id"])
        class_name = _strip_brackets(clip_info["template"])
        clip_path = VIDEO_DIR / f"{clip_id}.webm"

        pixel_values = _preprocess_videomae(clip_path, 16, processor, DEVICE)
        with torch.no_grad():
            model(pixel_values=pixel_values)

        acts = hook_store["acts"]  # (1, 1568, 768)
        tokens = (acts.squeeze(0) - dim_mean).float()  # (1568, 768)

        with torch.no_grad():
            _, z_8x  = sae_8x.encode(tokens)
            _, z_16x = sae_16x.encode(tokens)

        top_8x  = top_features(z_8x,  TOP_N)
        top_16x = top_features(z_16x, TOP_N)

        print(f"Clip {clip_id}  |  {class_name}")
        print(f"  {'SAE-8x  (feat / mean_act)':<30}  {'SAE-16x (feat / mean_act)'}")
        print(f"  {'-'*28}  {'-'*28}")
        for (i8, v8), (i16, v16) in zip(top_8x, top_16x):
            print(f"  8x-feat-{i8:<6} {v8:>8.4f}          16x-feat-{i16:<6} {v16:>8.4f}")
        print()


if __name__ == "__main__":
    main()
