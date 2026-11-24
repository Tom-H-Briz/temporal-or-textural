"""
UMT-B/16 25M (stage-2 pretrained, never SSv2-fine-tuned) for the MSRVTT zero-shot gate
(umt25m_brief_241126.md §2). Vision path mirrors umt_wrapper.py; the text tower and VTM
fusion are plain torch (the repo's xbert cannot import under transformers 5.5 on Isambard,
same strategy as umt_vtm_gate.py, which matches xbert to 1.9e-6). Scoring ports
criterions.get_sim (VTC) and tasks/retrieval_utils.py evaluation() (VTM rerank, itm_head[:, 1]).

Frames: b16_25m.pth stores no pos_embed - vit.py builds it at load as ONE 1D sinusoid over
the flattened token index (frame*196 + patch), so building at 12 frames does not interpolate:
frames 4-11 get positions 784-2351 that 25M never saw in training. The gate therefore stays
at 4 frames (the training/eval protocol); the analysis frame count is an open user decision.
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent

CFG = {
    "ckpt":      ROOT / "models/umt_ckpts/b16_25m.pth",
    "vit_path":  ROOT / "models/unmasked_teacher/multi_modality/models/backbones/vit/vit.py",
    "num_frames": 4,          # 25M pretrained and evaluated at 4 frames (exp/zero_shot b16.py)
    "img_size":  224,
    "embed_dim": 512,
    "text":      dict(hidden=768, heads=12, head_dim=64, layers=12, fusion_layer=9, ln_eps=1e-12),
    "add_mask_scale": -10000.0,   # xbert extended-attention-mask convention
}


def load_sd(ckpt_path):
    """25M checkpoint with the saved CLIP teacher dropped: umt.py only builds the teacher
    when clip_teacher != 'none', so the retrieval model never has those weights. The
    vision_encoder.clip_decoder.* keys (UTA alignment heads, unused by the retrieval
    forward) are dropped too - the SSv2 ckpt never had them, the 25M pretrain save does."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    drop = [k for k in sd if k.startswith("clip_teacher.") or k.startswith("vision_encoder.clip_decoder.")]
    for k in drop:
        del sd[k]
    print(f"dropped {len(drop)} clip_teacher.*/vision_encoder.clip_decoder.* keys")
    return sd


def build_vision_tower(sd, num_frames):
    """Repo's vit.py, config from exp/zero_shot/ret_msrvtt/b16.py (eval: no drop path).
    strict=True load => 0 missing / 0 unexpected vision_encoder.* keys or it raises."""
    from easydict import EasyDict
    spec = importlib.util.spec_from_file_location("umt_vit", CFG["vit_path"])
    vit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vit)
    ve = EasyDict(
        name="vit_b16", img_size=CFG["img_size"], patch_size=16, encoder_embed_dim=768,
        encoder_depth=12, encoder_num_heads=12, drop_path_rate=0.0, num_frames=num_frames,
        tubelet_size=1, use_checkpoint=False, checkpoint_num=0, clip_decoder_embed_dim=768,
        clip_output_dim=CFG["embed_dim"], clip_return_layer=0, clip_student_return_interval=1,
        pretrained=None, clip_teacher="none", clip_img_size=CFG["img_size"],
        clip_return_interval=1,
    )
    tower = vit.build_vit(EasyDict(vision_encoder=ve))
    weights = {k[15:]: v for k, v in sd.items() if k.startswith("vision_encoder.")}
    missing = set(tower.state_dict()) - set(weights)
    unexpected = set(weights) - set(tower.state_dict())
    print(f"vision_encoder keys: {len(weights)} matched, missing={len(missing)}, unexpected={len(unexpected)}")
    tower.load_state_dict(weights, strict=True)
    return tower


# ---- plain-torch xbert BERT (post-LN BertLayer), same helpers as umt_vtm_gate.py ----
def _lin(x, w, name):
    return F.linear(x, w[name + ".weight"], w[name + ".bias"])


def _heads(x):   # (N, L, 768) -> (N, 12, L, 64)
    t = CFG["text"]
    return x.view(*x.shape[:2], t["heads"], t["head_dim"]).transpose(1, 2)


def _attend(x, k, v, w, pre, add_mask):
    """xbert BertAttention: q from x, k/v given (cross-attn k/v broadcast over N); add & LayerNorm."""
    s = _heads(_lin(x, w, pre + ".self.query")) @ k.transpose(-1, -2) / CFG["text"]["head_dim"] ** 0.5
    if add_mask is not None:
        s = s + add_mask
    ctx = (s.softmax(-1) @ v).transpose(1, 2).reshape(x.shape)
    return F.layer_norm(_lin(ctx, w, pre + ".output.dense") + x, (768,),
                        w[pre + ".output.LayerNorm.weight"], w[pre + ".output.LayerNorm.bias"], 1e-12)


def _kv(x, w, pre):
    return _heads(_lin(x, w, pre + ".self.key")), _heads(_lin(x, w, pre + ".self.value"))


def _ffn(h, w, p):
    ff = _lin(F.gelu(_lin(h, w, p + ".intermediate.dense")), w, p + ".output.dense")
    return F.layer_norm(ff + h, (768,), w[p + ".output.LayerNorm.weight"], w[p + ".output.LayerNorm.bias"], 1e-12)


def bert_embeddings(ids, w):
    """xbert BertEmbeddings (eval): word + position + token_type, then LayerNorm."""
    t = CFG["text"]
    e = "text_encoder.embeddings"
    h = (w[e + ".word_embeddings.weight"][ids]
         + w[e + ".position_embeddings.weight"][: ids.shape[1]][None]
         + w[e + ".token_type_embeddings.weight"][torch.zeros_like(ids)])
    return F.layer_norm(h, (t["hidden"],), w[e + ".LayerNorm.weight"], w[e + ".LayerNorm.bias"], t["ln_eps"])


def encode_text_25m(ids, att, w):
    """BERT layers 0-8 only (xbert mode='text' stops at fusion_layer, so the returned
    states are exactly the official text_feats) -> (hidden (B,L,768), add_mask)."""
    h = bert_embeddings(ids, w)
    add_mask = ((1.0 - att.float()) * CFG["add_mask_scale"])[:, None, None, :]
    for i in range(CFG["text"]["fusion_layer"]):
        p = f"text_encoder.encoder.layer.{i}"
        h = _attend(h, *_kv(h, w, p + ".attention"), w, p + ".attention", add_mask)
        h = _ffn(h, w, p)
    return h, add_mask


def fusion_scores(l8_h, add_mask, x_vis, w):
    """UMT eval rerank score: fusion layers 9-11 (self-attn text padding-masked, cross-attn
    to the video tokens unmasked, as in retrieval_utils' all-ones encoder_attention_mask),
    then itm_head[:, 1]. l8_h (K, L, 768) per-pair text states; x_vis (K, N, 768) per-pair video tokens."""
    t = CFG["text"]
    kv = {i: _kv(x_vis, w, f"text_encoder.encoder.layer.{i}.crossattention") for i in (9, 10, 11)}
    h = l8_h
    for i in (9, 10, 11):
        p = f"text_encoder.encoder.layer.{i}"
        h = _attend(h, *_kv(h, w, p + ".attention"), w, p + ".attention", add_mask)
        h = _attend(h, *kv[i], w, p + ".crossattention", None)
        h = _ffn(h, w, p)
    return _lin(h[:, 0], w, "itm_head")[:, 1]


def vtc_scores(pooled_v, cls_t, temp):
    """criterions.get_sim: normalized per-frame dots, mean over frames, / temp.
    pooled_v (Bv, T, 512); cls_t (Bt, 512). Returns (Bv, Bt)."""
    v = F.normalize(pooled_v, dim=-1)
    t = F.normalize(cls_t, dim=-1)
    return torch.einsum("btd,nd->btn", v, t).mean(1) / temp


class UMT25M(torch.nn.Module):
    """Thin container over the 25M weights. encode_video(pixel_values (B,T,C,H,W))
    -> (x_vis (B,T*196,768), pooled (B,T,768)); encode_text -> (l8 hidden, add_mask);
    vtc_logits(pooled_v, cls_t); vtm_score(l8_h, add_mask, x_vis)."""

    def __init__(self, tower, sd, temp):
        super().__init__()
        self.vision = tower
        self.vision_proj = torch.nn.Linear(768, CFG["embed_dim"])
        self.text_proj = torch.nn.Linear(768, CFG["embed_dim"])
        self.vision_proj.load_state_dict({"weight": sd["vision_proj.weight"], "bias": sd["vision_proj.bias"]})
        self.text_proj.load_state_dict({"weight": sd["text_proj.weight"], "bias": sd["text_proj.bias"]})
        self.temp = float(temp)
        self.w = {k.replace("text_encoder.bert.", "text_encoder."): v for k, v in sd.items()
                  if k.startswith("text_encoder.bert.")
                  and not k.startswith("text_encoder.bert.embeddings.position_ids")}
        self.w.update({k: v for k, v in sd.items() if k.startswith("itm_head.")})
    def encode_video(self, pixel_values):
        x_vis, pooled, _ = self.vision(pixel_values.permute(0, 2, 1, 3, 4), keep_temporal=True)
        return x_vis, pooled

    def encode_text(self, input_ids, attention_mask):
        return encode_text_25m(input_ids, attention_mask, self.w)

    def vtc_logits(self, pooled_v, cls_t):
        return vtc_scores(pooled_v, cls_t, self.temp)

    def vtm_score(self, l8_h, add_mask, x_vis):
        return fusion_scores(l8_h, add_mask, x_vis, self.w)


def build_umt25m(ckpt_path=None, num_frames=None, device="cpu"):
    """Build + weight the 25M retrieval model. Prints the vision key counts and the
    number of text/fusion weights kept; MLM head keys are simply never loaded."""
    sd = load_sd(ckpt_path or CFG["ckpt"])
    model = UMT25M(build_vision_tower(sd, num_frames or CFG["num_frames"]), sd, sd["temp"])
    print(f"text/fusion weights kept: {len(model.w)} (incl. {sum(k.startswith('itm_head.') for k in model.w)} itm_head)")
    model = model.to(device).eval()
    model.w = {k: v.to(device) for k, v in model.w.items()}   # plain dict: .to() doesn't move it
    return model
