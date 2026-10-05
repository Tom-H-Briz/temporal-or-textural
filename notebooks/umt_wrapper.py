"""
UMT-B/16 (SSv2-template retrieval checkpoint) as a classifier: video -> 174 template logits.

Vision tower only (repo's vit.py + vision_proj); the 174 template text embeddings are
pre-computed and loaded from disk, so no BERT / old transformers is needed.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, Resize

ROOT = Path(__file__).parent.parent

CFG = {
    "ckpt_path":     ROOT / "models/umt_ckpts/ret_ssv2_tpl_b16_25m.pth",
    # (174, 512) text_proj outputs from the spike's CPU run (outputs/umt_ssv2_tpl_cls/raw/
    # text_emb.npy), rows RE-ORDERED to labels.json class id — row i == logit i == class i.
    "text_emb_path": ROOT / "models/umt_ckpts/ssv2_template_text_emb.npy",
    "vit_path":      ROOT / "models/unmasked_teacher/multi_modality/models/backbones/vit/vit.py",
    "num_frames":    12,
    "img_size":      224,
    "mean":          (0.485, 0.456, 0.406),
    "std":           (0.229, 0.224, 0.225),
}


def sample_frames_umt(n: int, num_frames: int) -> list[int]:
    """UMT's 'middle' sampling: midpoint of each of num_frames equal intervals
    (repo dataset/video_utils.py get_frame_indices, sample='middle')."""
    edges = np.linspace(0, n, num_frames + 1).astype(int)
    idx = [(edges[i] + edges[i + 1] - 1) // 2 for i in range(num_frames)]
    return idx


class UMTProcessor:
    """Same call shape as the HF processors: processor(frames, return_tensors="pt")
    -> {"pixel_values": (1, T, C, H, W)}. Repo eval transform: bicubic squash-resize
    on uint8, then /255, then ImageNet mean/std."""

    @classmethod
    def from_pretrained(cls, checkpoint=None):
        return cls()

    def __call__(self, frames, return_tensors="pt"):
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)   # (T, C, H, W) uint8
        size = (CFG["img_size"], CFG["img_size"])
        x = Resize(size, interpolation=InterpolationMode.BICUBIC)(x).float().div(255.0)
        mean = torch.tensor(CFG["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(CFG["std"]).view(1, 3, 1, 1)
        return {"pixel_values": ((x - mean) / std).unsqueeze(0)}


def build_vision_tower(sd: dict):
    """Repo's vit.py (loaded by path: its package __init__ imports clip.py), config
    from exp/finetuning/ret_ssv2_tpl/b16_25m.py. Strict-loads vision_encoder.* keys."""
    from easydict import EasyDict
    spec = importlib.util.spec_from_file_location("umt_vit", CFG["vit_path"])
    vit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vit)
    ve = EasyDict(
        name="vit_b16", img_size=224, patch_size=16, encoder_embed_dim=768, encoder_depth=12,
        encoder_num_heads=12, drop_path_rate=0.0, num_frames=CFG["num_frames"], tubelet_size=1,
        use_checkpoint=False, checkpoint_num=0, clip_decoder_embed_dim=768, clip_output_dim=512,
        clip_return_layer=0, clip_student_return_interval=1, pretrained=None, clip_teacher="none",
        clip_img_size=224, clip_return_interval=1,
    )
    tower = vit.build_vit(EasyDict(vision_encoder=ve))
    weights = {k[15:]: v for k, v in sd.items() if k.startswith("vision_encoder.")}
    tower.load_state_dict(weights, strict=True)
    return tower


class UMTClassifier(torch.nn.Module):
    """pixel_values (B, T, C, H, W) -> .logits (B, 174): per-frame cosine to each
    template embedding, mean over frames, divided by the learned temperature."""

    def __init__(self, tower, vision_proj, text_emb, temp):
        super().__init__()
        self.vision, self.vision_proj, self.temp = tower, vision_proj, temp
        self.register_buffer("text_emb", F.normalize(text_emb, dim=-1))   # (174, 512)

    @classmethod
    def from_pretrained(cls, checkpoint=None):
        sd = torch.load(CFG["ckpt_path"], map_location="cpu", weights_only=False)
        proj = torch.nn.Linear(768, 512)
        proj.load_state_dict({"weight": sd["vision_proj.weight"], "bias": sd["vision_proj.bias"]})
        text = torch.from_numpy(np.load(CFG["text_emb_path"]))
        return cls(build_vision_tower(sd), proj, text, float(sd["temp"]))

    def forward(self, pixel_values):
        x = pixel_values.permute(0, 2, 1, 3, 4)                     # (B, C, T, H, W)
        _, pooled, _ = self.vision(x, keep_temporal=True)           # (B, T, 768), post pool_norm
        frames = F.normalize(self.vision_proj(pooled), dim=-1)      # (B, T, 512)
        sim = torch.einsum("btd,cd->bc", frames, self.text_emb) / frames.shape[1]
        return type("Out", (), {"logits": sim / self.temp})()
