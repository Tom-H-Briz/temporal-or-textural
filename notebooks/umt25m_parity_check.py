"""Parity check (umt25m_brief_241126.md §2, agent B): the official repo's UMT class
(xbert, transformers 4.24) against our umt25m_wrapper on ~20 MSRVTT clips + captions.
Reports max |diff| for the layer-8 text states, pooled vision tokens, VTC similarities
and VTM scores. Run with the 4.24 venv, NOT the uv env:
  models/umt_venv/bin/python notebooks/umt25m_parity_check.py
Env: MSRVTT_ANNO, MSRVTT_VIDEO_DIR, UMT25M_CKPT, N_CLIPS (default 20). TF32 is forced off.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "notebooks"), str(ROOT / "models/unmasked_teacher/multi_modality")]
from umt25m_msrvtt_eval import CFG as EVAL_CFG                      # noqa: E402
from umt25m_msrvtt_eval import decode_frames, load_annotations      # noqa: E402
from umt_wrapper import UMTProcessor                               # noqa: E402

CFG = {
    "n_clips": int(os.environ.get("N_CLIPS", 20)),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "out": ROOT / "outputs/analysis/umt25m_msrvtt_gate/parity.json",
}


def build_official(device):
    """Official UMT exactly as exp/zero_shot/ret_msrvtt/b16.py configures it; the 25M
    weights land via strict=False (clip_teacher.*, text_encoder.cls.* dropped, reported)."""
    from easydict import EasyDict
    from models.umt import UMT
    from transformers import AutoTokenizer
    ve = EasyDict(
        name="vit_b16", img_size=224, patch_size=16, d_model=768, encoder_embed_dim=768,
        encoder_depth=12, encoder_num_heads=12, drop_path_rate=0.0, num_frames=4,
        tubelet_size=1, use_checkpoint=False, checkpoint_num=0, clip_decoder_embed_dim=768,
        clip_output_dim=512, clip_return_layer=0, clip_student_return_interval=1,
        pretrained=None, clip_teacher="none", clip_img_size=224, clip_return_interval=1,
        video_mask_type="attention", video_mask_ratio=0.0, video_double_mask_ratio=0.0,
        image_mask_type="attention", image_mask_ratio=0.0, image_double_mask_ratio=0.0,
        keep_temporal=True,
    )
    cfg = EasyDict(
        model=EasyDict(vision_encoder=ve, embed_dim=512, temp=0.07,
                       text_encoder=EasyDict(name="bert_base", pretrained="bert-base-uncased",
                                             config=str(ROOT / "models/unmasked_teacher/multi_modality/configs/config_bert.json"),
                                             d_model=768, fusion_layer=9),
                       multimodal=EasyDict(enable=True)),
        criterion=EasyDict(loss_weight=EasyDict(vtc=1.0, mlm=0.0, vtm=1.0, uta=0.0),
                           vtm_hard_neg=True, mlm_masking_prob=0.5,
                           uta_norm_type="l2", uta_loss_type="l2"),
        gradient_checkpointing=False,
    )
    model = UMT(cfg, AutoTokenizer.from_pretrained("bert-base-uncased"), is_pretrain=False)
    sd = torch.load(os.environ.get("UMT25M_CKPT", str(EVAL_CFG["ckpt"])), map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    # shared_utils.load_checkpoint's zero-shot branch: text_encoder.bert.* -> text_encoder.*
    sd = {k.replace("bert.", "") if "bert" in k else k: v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"official load: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    print(f"  unexpected prefixes: {sorted({'.'.join(k.split('.')[:2]) for k in msg.unexpected_keys})}")
    return model.to(device).eval()


@torch.no_grad()
def compare(device):
    """Same decoded frames + tokens into both models; per-output max |diff|."""
    from models.criterions import get_sim
    from transformers import AutoTokenizer
    from umt25m_msrvtt_eval import resolve_videos
    from umt25m_wrapper import build_umt25m
    vids, texts, *_ = load_annotations(EVAL_CFG["anno"], CFG["n_clips"])
    paths = resolve_videos(EVAL_CFG["video_dir"], vids)
    proc = UMTProcessor()
    pv = torch.stack([proc(decode_frames(p, 4))["pixel_values"][0] for p in paths]).to(device)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = tok(texts, padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)
    off, ours = build_official(device), build_umt25m(str(EVAL_CFG["ckpt"]), 4, device)
    o_vis, o_pool = off.encode_vision(pv, test=True)
    if o_vis.dim() == 4:   # official rearranges "b t l c -> b (t l) c" in retrieval_utils
        o_vis = o_vis.flatten(1, 2)
    x_vis, pooled = ours.encode_video(pv)
    l8, add_mask = ours.encode_text(enc.input_ids, enc.attention_mask)
    o_txt = off.encode_text(enc)[0]
    o_tproj, o_vproj = off.text_proj(o_txt[:, 0]), off.vision_proj(o_pool)
    s_off = get_sim(o_vproj, o_tproj, off.temp)[0]
    s_ours = ours.vtc_logits(ours.vision_proj(pooled), ours.text_proj(l8[:, 0]))
    o_att = torch.ones(o_vis.shape[:-1], dtype=torch.long, device=device)
    o_fus = off.text_encoder(encoder_embeds=o_txt, attention_mask=enc.attention_mask,
                             encoder_hidden_states=o_vis,          # 1:1 pairs, no repeat
                             encoder_attention_mask=o_att, return_dict=True, mode="fusion")
    o_itm = off.itm_head(o_fus.last_hidden_state[:, 0])[:, 1]
    ours_itm = ours.vtm_score(l8, add_mask, x_vis)
    return dict(
        vision_tokens=(o_vis - x_vis).abs().max().item(),
        pooled_vision=(o_pool - pooled).abs().max().item(),
        text_l8=(o_txt - l8).abs().max().item(),
        text_proj=(o_tproj - ours.text_proj(l8[:, 0])).abs().max().item(),
        vision_proj=(o_vproj - ours.vision_proj(pooled)).abs().max().item(),
        vtc_sim=(s_off - s_ours).abs().max().item(),
        vtm_score=(o_itm - ours_itm).abs().max().item(),
    )


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    res = compare(CFG["device"])
    for k, v in res.items():
        print(f"{k:15s} max |diff| = {v:.3e}")
    res["pass"] = bool(max(res.values()) < 1e-3)
    CFG["out"].parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(CFG["out"], "w"), indent=2)
    print(f"PARITY {'PASS' if res['pass'] else 'FAIL'} (threshold 1e-3) -> {CFG['out']}")


if __name__ == "__main__":
    main()
