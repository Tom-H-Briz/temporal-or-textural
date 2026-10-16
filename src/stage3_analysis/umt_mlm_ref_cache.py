"""One-off, LOCAL, in models/umt_venv (transformers 4.24 for UMT's xbert):
ground-truth reference for the plain-torch MLM port in umt_mlm_refit.py.

Dumps notebooks/umt_mlm_ref.npz:
  - N_REF val captions, every word-piece masked once, fixed random vision tokens,
    and xbert's MLM logits at the masked position for BOTH BERTs
    (finetuned = graft config: pretrain head, decoder tied to finetuned embeddings);
  - umt_mlm_graft_check.py's exact masked ids/labels/is_obj (seed 0, 300 captions),
    so the port's reproduction check runs on identical inputs;
  - token ids for all 174 templates + the ref captions, to verify the pipeline env's
    tokenizer (transformers 5.5 BertTokenizer) matches xbert's before D2 uses it.

Usage (from repo root):
  PYTHONPATH=models/unmasked_teacher/multi_modality models/umt_venv/bin/python src/stage3_analysis/umt_mlm_ref_cache.py
"""
import json
from pathlib import Path
import numpy as np
import torch

from umt_mlm_graft_check import CFG as GRAFT_CFG, build_mlm, strip_prefix

CFG = {
    "pretrain_ckpt": Path("models/umt_ckpts/b16_25m.pth"),
    "finetune_ckpt": Path("models/umt_ckpts/ret_ssv2_tpl_b16_25m.pth"),
    "labels":        Path("data/ssv2/labels/labels.json"),
    "captions":      Path("data/ssv2/labels/validation.json"),
    "out":           Path("notebooks/umt_mlm_ref.npz"),
    "n_ref_clips":   10,    # ~60 masked variants; logits (n, 30522) fp32 x2 models
    "seed":          7,
    "max_txt_l":     32,    # graft-check settings
    "n_ref_vis":     32,    # random vision tokens for the cross-attention input
}


def ref_examples(tok):
    """N_REF val captions (seed CFG), each word-piece masked once -> padded ids,
    attention mask, mask positions, true ids, is_obj (graft-check rules)."""
    rows = json.load(open(CFG["captions"]))
    rng = np.random.default_rng(CFG["seed"])
    picked = rng.choice(rows, CFG["n_ref_clips"], replace=False)
    ids, pos, true, is_obj, caps = [], [], [], [], []
    for r in picked:
        caps.append(r["label"])
        enc = tok(r["label"].lower(), truncation=True, max_length=CFG["max_txt_l"])["input_ids"]
        obj = set(tok(" ".join(r["placeholders"]).lower(), add_special_tokens=False)["input_ids"])
        for p in range(1, len(enc) - 1):
            ids.append(enc[:p] + [tok.mask_token_id] + enc[p + 1:])
            pos.append(p)
            true.append(enc[p])
            is_obj.append(enc[p] in obj)
    L = max(map(len, ids))
    x = torch.tensor([b + [0] * (L - len(b)) for b in ids])
    return x, (x != 0).long(), pos, true, is_obj, caps


@torch.no_grad()
def ref_logits(mlm, x, attn, vis, pos):
    """xbert MLM logits at each example's masked position, given shared vision tokens."""
    out = []
    for s in range(0, len(x), GRAFT_CFG["batch_size"]):
        xb, ab = x[s:s + GRAFT_CFG["batch_size"]], attn[s:s + GRAFT_CFG["batch_size"]]
        enc = vis.unsqueeze(0).expand(len(xb), -1, -1).contiguous()
        lg = mlm(xb, attention_mask=ab, encoder_hidden_states=enc,
                 encoder_attention_mask=torch.ones(enc.shape[:2], dtype=torch.long),
                 return_dict=True, return_logits=True, mode="multi_modal")
        out.append(lg[torch.arange(len(xb)), torch.tensor(pos[s:s + GRAFT_CFG["batch_size"]])])
    return torch.cat(out).numpy().astype(np.float32)


@torch.no_grad()
def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models.backbones.bert.tokenization_bert import BertTokenizer
    from umt_mlm_graft_check import masked_examples
    load = lambda p: (lambda sd: sd.get("model", sd))(torch.load(p, map_location="cpu", weights_only=False))
    pre, fin = load(CFG["pretrain_ckpt"]), load(CFG["finetune_ckpt"])
    head = strip_prefix(pre, "text_encoder.cls.")
    mlms = {"pretrain": build_mlm(strip_prefix(pre, "text_encoder.bert."), head),
            "graft":    build_mlm(strip_prefix(fin, "text_encoder."), head)}
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    x, attn, pos, true, is_obj, caps = ref_examples(tok)
    vis = torch.randn(CFG["n_ref_vis"], 768, generator=torch.Generator().manual_seed(0))
    refs = {k: ref_logits(m, x, attn, vis, pos) for k, m in mlms.items()}
    g_ids, g_labels, g_is_obj = masked_examples(tok)          # graft check, seed 0, 300 captions
    gL = max(map(len, g_ids))
    gx = np.array([b + [0] * (gL - len(b)) for b in g_ids], dtype=np.int32)
    templates = list(json.load(open(CFG["labels"])).keys())
    tpl_ids = tok([" ".join(t.lower().replace("[", "").replace("]", "").split())
                   for t in templates], add_special_tokens=False)["input_ids"]
    np.savez(CFG["out"], ref_ids=x.numpy(), ref_attn=attn.numpy(), ref_pos=np.array(pos),
             ref_true=np.array(true), ref_is_obj=np.array(is_obj), ref_vis=vis.numpy(),
             ref_logits_pretrain=refs["pretrain"], ref_logits_graft=refs["graft"],
             graft_ids=gx, graft_pos=np.array([p for p, _ in g_labels], dtype=np.int32),
             graft_true=np.array([t for _, t in g_labels], dtype=np.int32), graft_is_obj=g_is_obj,
             tpl_ids=np.array(tpl_ids, dtype=object),
             ref_caps=np.array(caps, dtype=object))
    print(f"saved {CFG['out']}: {len(x)} ref variants, graft {len(gx)}; "
          f"logits {refs['pretrain'].shape} + {refs['graft'].shape}")


if __name__ == "__main__":
    main()
