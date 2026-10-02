"""Part A: structural check of the UMT-B MLM head (FT vs S2). Read-only, no forwards."""
import sys
import json
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent.parent
UMT_MM = ROOT / "models" / "unmasked_teacher" / "multi_modality"
sys.path.insert(0, str(UMT_MM))

CONFIG = dict(
    ft_ckpt=ROOT / "models/umt_ckpts/ret_ssv2_tpl_b16_25m.pth",
    s2_ckpt=ROOT / "models/umt_ckpts/b16_25m.pth",
    out_dir=ROOT / "outputs/umt_mlm_check",
    fusion_layer=9,
    num_bert_layers=12,
)


def model_config_dict(is_pretrain):
    """Model section of exp/finetuning/ret_ssv2_tpl/b16_25m.py, pretrained=None.

    - pretrained=None: build_vit skips loading the (absent) single-modality
      .pth; the UMT checkpoint fills all weights afterwards. Config-level
      change only, no repo code edited.
    """
    return dict(
        model_cls="UMT",
        vision_encoder=dict(
            name="vit_b16", img_size=224, patch_size=16, d_model=768,
            encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
            drop_path_rate=0.1, num_frames=12, tubelet_size=1,
            use_checkpoint=False, checkpoint_num=12,
            clip_decoder_embed_dim=768, clip_output_dim=512,
            clip_return_layer=0, clip_student_return_interval=1,
            pretrained=None, clip_teacher="none", clip_img_size=224,
            clip_return_interval=1,
            video_mask_type="attention", video_mask_ratio=0.,
            video_double_mask_ratio=0., image_mask_type="attention",
            image_mask_ratio=0., image_double_mask_ratio=0.,
            keep_temporal=True,
        ),
        text_encoder=dict(
            name="bert_base", pretrained="bert-base-uncased",
            config=str(UMT_MM / "configs/config_bert.json"),
            d_model=768, fusion_layer=CONFIG["fusion_layer"],
        ),
        multimodal=dict(enable=True),
        embed_dim=512,
        temp=0.07,
        criterion=dict(
            loss_weight=dict(vtc=1.0, mlm=0.0, vtm=1.0, uta=0.0),
            vtm_hard_neg=True, mlm_masking_prob=0.5,
            uta_norm_type="l2", uta_loss_type="l2",
        ),
        gradient_checkpointing=False,
    )


def build_umt(is_pretrain):
    """Instantiate the repo's own UMT class (models/umt.py), as tasks/retrieval.py does."""
    from easydict import EasyDict
    from models.umt import UMT
    from models.backbones.bert.tokenization_bert import BertTokenizer

    model_cfg = model_config_dict(is_pretrain)
    criterion = model_cfg.pop("criterion")
    cfg = EasyDict(
        model=EasyDict(model_cfg),
        criterion=EasyDict(criterion),
        gradient_checkpointing=False,
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = UMT(config=cfg, tokenizer=tokenizer, is_pretrain=is_pretrain)
    return model, tokenizer


def load_state_dict(path):
    """Return the model state dict inside a checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def load_report(model, sd, label, out):
    """Load checkpoint into the repo model class, record missing/unexpected keys."""
    msg = model.load_state_dict(sd, strict=False)
    out[f"{label}_missing_keys"] = sorted(msg.missing_keys)
    out[f"{label}_unexpected_keys"] = sorted(msg.unexpected_keys)
    groups = {}
    for k in msg.unexpected_keys:
        groups.setdefault(k.split(".")[0] + "." + k.split(".")[1], 0)
        groups[k.split(".")[0] + "." + k.split(".")[1]] += 1
    out[f"{label}_unexpected_groups"] = dict(sorted(groups.items()))
    print(f"[{label}] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")


def mlm_head_keys(sd):
    """All text-side MLM head keys (BERT cls.predictions.*)."""
    return sorted(k for k in sd if "cls.predictions" in k)


def tensor_diff(a, b):
    """Max abs diff between two tensors, or None if shapes differ."""
    if a.shape != b.shape:
        return None
    return (a.float() - b.float()).abs().max().item()


def check_tying(dec_key, emb_key, sd):
    """Is the MLM output weight identical to the word-embedding matrix?"""
    dec, emb = sd.get(dec_key), sd.get(emb_key)
    if dec is None or emb is None:
        return "n/a (tensor absent)"
    identical = dec.shape == emb.shape and torch.equal(dec, emb)
    return f"{'tied (bitwise identical)' if identical else f'not tied (max abs diff {tensor_diff(dec, emb):.6f})'}"


def upstream_drift(ft_sd, s2_sd, out):
    """Per-layer max abs diff of text-side layers, FT vs S2 (key prefixes differ)."""
    rows = {}
    for n in range(CONFIG["num_bert_layers"]):
        layer_max, n_keys = 0.0, 0
        for sub in ("attention", "crossattention", "intermediate", "output"):
            for k in s2_sd:
                if f"text_encoder.bert.encoder.layer.{n}.{sub}." not in k:
                    continue
                ft_k = k.replace("text_encoder.bert.", "text_encoder.")
                if ft_k not in ft_sd:
                    continue
                d = tensor_diff(ft_sd[ft_k], s2_sd[k])
                layer_max = max(layer_max, d if d is not None else 0.0)
                n_keys += 1
        rows[f"layer_{n:02d}"] = dict(max_abs_diff=layer_max, n_tensors=n_keys)
    out["upstream_drift"] = rows


def main():
    out = dict(device="cpu (M4 MacBook, no CUDA; structural check is tensor arithmetic only)")
    ft_sd, s2_sd = load_state_dict(CONFIG["ft_ckpt"]), load_state_dict(CONFIG["s2_ckpt"])

    ft_model, _ = build_umt(is_pretrain=False)
    s2_model, _ = build_umt(is_pretrain=True)
    load_report(ft_model, ft_sd, "ft_model_load", out)
    load_report(s2_model, s2_sd, "s2_model_load", out)

    ft_mlm = mlm_head_keys(ft_sd)
    s2_mlm = mlm_head_keys(s2_sd)
    out["mlm_head_keys_s2"] = {k: list(s2_sd[k].shape) for k in s2_mlm}
    out["mlm_head_keys_ft"] = {k: list(ft_sd[k].shape) for k in ft_mlm}

    out["diff_ft_vs_s2_mlm_head"] = {
        "note": "FT checkpoint contains no cls.predictions.* tensors; per-tensor diff undefined",
        "per_tensor": {},
    }
    for k in s2_mlm:
        out["diff_ft_vs_s2_mlm_head"]["per_tensor"][k] = "absent in FT"

    out["tying_s2"] = check_tying(
        "text_encoder.cls.predictions.decoder.weight",
        "text_encoder.bert.embeddings.word_embeddings.weight", s2_sd)
    out["tying_ft"] = check_tying(
        "text_encoder.cls.predictions.decoder.weight",
        "text_encoder.embeddings.word_embeddings.weight", ft_sd)
    upstream_drift(ft_sd, s2_sd, out)

    ft_emb = ft_sd.get("text_encoder.embeddings.word_embeddings.weight")
    s2_dec = s2_sd.get("text_encoder.cls.predictions.decoder.weight")
    out["ft_wordemb_vs_s2_mlm_decoder"] = (
        f"max abs diff {tensor_diff(ft_emb, s2_dec):.6f}"
        if ft_emb is not None and s2_dec is not None else "n/a"
    )
    return out


if __name__ == "__main__":
    result = main()
    CONFIG["out_dir"].mkdir(parents=True, exist_ok=True)
    json_path = CONFIG["out_dir"] / "part_a_raw.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"saved {json_path}")
    print("FT MLM head:", result["mlm_head_keys_ft"] or "ABSENT")
    print("S2 MLM head tensors:", len(result["mlm_head_keys_s2"]))
    print("tying S2:", result["tying_s2"], "| tying FT:", result["tying_ft"])
