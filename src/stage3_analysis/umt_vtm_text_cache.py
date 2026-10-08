"""One-off, LOCAL, in models/umt_venv (transformers 4.24 for UMT's xbert):
cache BERT layer 0-8 outputs for the 174 SSv2 templates (class-id order) + fusion-layer
reference outputs, so umt_vtm_gate.py can run layers 9-11 in plain torch under the pipeline env.

Usage (from repo root):
  PYTHONPATH=models/unmasked_teacher/multi_modality models/umt_venv/bin/python src/stage3_analysis/umt_vtm_text_cache.py
"""
import json
from pathlib import Path
import numpy as np
import torch

CFG = {
    "ckpt":       Path("models/umt_ckpts/ret_ssv2_tpl_b16_25m.pth"),
    "bert_cfg":   Path("models/unmasked_teacher/multi_modality/configs/config_bert.json"),
    "ds_utils":   Path("models/unmasked_teacher/multi_modality/dataset/utils.py"),  # pre_text
    "spike_maps": Path("outputs/umt_ssv2_tpl_cls/raw/gt_maps.json"),   # UMT's own template strings
    "labels":     Path("data/ssv2/labels/labels.json"),
    "text_emb":   Path("notebooks/umt_ssv2_template_text_emb.npy"),    # VTC cache to verify against
    "out":        Path("notebooks/umt_ssv2_template_bert_l8.npz"),     # git-tracked (*.pt is gitignored), ~13 MB
    "max_txt_l":  25,
    "n_ref_tok":  64,   # random vision tokens for the plain-torch fusion check
}


def templates_in_class_order():
    """UMT's template strings (exact tokenizer input), re-ordered so row i == labels.json class i."""
    import importlib.util   # by path: dataset/__init__ imports decord, which we don't need
    spec = importlib.util.spec_from_file_location("umt_ds_utils", CFG["ds_utils"])
    ds_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds_utils)
    strip = lambda s: ds_utils.pre_text(s).replace("[", "").replace("]", "")
    labels = json.load(open(CFG["labels"]))
    cap2id = {strip(k): int(v) for k, v in labels.items()}
    texts = json.load(open(CFG["spike_maps"]))["texts"]
    ids = [cap2id[strip(t)] for t in texts]
    assert sorted(ids) == list(range(174)), "template -> class id mapping is not a bijection"
    return [t for _, t in sorted(zip(ids, texts))]


def build_bert(sd):
    """UMT's xbert BertModel (fusion_layer=9), strict-loaded from the checkpoint's text_encoder.*."""
    from models.backbones.bert.xbert import BertConfig, BertModel
    cfg = BertConfig.from_json_file(str(CFG["bert_cfg"]))
    cfg.encoder_width, cfg.fusion_layer = 768, 9
    bert = BertModel(cfg, add_pooling_layer=False)
    bert.load_state_dict({k[13:]: v for k, v in sd.items() if k.startswith("text_encoder.")}, strict=True)
    return bert.eval()


@torch.no_grad()
def reference_itm(bert, itm_head, hidden, mask):
    """xbert fusion (layers 9-11) + itm_head[:,1] on fixed random vision tokens: ground truth
    the gate script's plain-torch fusion must reproduce."""
    vis = torch.randn(2, CFG["n_ref_tok"], 768, generator=torch.Generator().manual_seed(0))
    ref = []
    for v in vis:
        enc = v.unsqueeze(0).repeat(len(hidden), 1, 1)
        out = bert(encoder_embeds=hidden, attention_mask=mask, encoder_hidden_states=enc,
                   encoder_attention_mask=torch.ones(enc.shape[:2], dtype=torch.long),
                   return_dict=True, mode="fusion")
        ref.append(itm_head(out.last_hidden_state[:, 0])[:, 1])
    return vis, torch.stack(ref)   # (2, n_tok, 768), (2, 174)


@torch.no_grad()
def main():
    from models.backbones.bert.tokenization_bert import BertTokenizer
    sd = torch.load(CFG["ckpt"], map_location="cpu", weights_only=False)
    bert, texts = build_bert(sd), templates_in_class_order()
    tok = BertTokenizer.from_pretrained("bert-base-uncased")(
        texts, padding="max_length", truncation=True, max_length=CFG["max_txt_l"], return_tensors="pt")
    hidden = bert(tok.input_ids, attention_mask=tok.attention_mask, return_dict=True, mode="text").last_hidden_state
    # Check 1: CLS -> text_proj reproduces the cached VTC text embeddings row-for-row.
    proj = hidden[:, 0] @ sd["text_proj.weight"].T + sd["text_proj.bias"]
    cos = torch.nn.functional.cosine_similarity(proj, torch.from_numpy(np.load(CFG["text_emb"])), dim=-1)
    print(f"text_proj(CLS) vs cached text_emb: min cos {cos.min():.6f}")
    itm_head = torch.nn.Linear(768, 2)
    itm_head.load_state_dict({"weight": sd["itm_head.weight"], "bias": sd["itm_head.bias"]})
    ref_vis, ref_itm = reference_itm(bert, itm_head, hidden, tok.attention_mask)
    np.savez(CFG["out"], texts=np.array(texts), hidden=hidden.numpy(), mask=tok.attention_mask.numpy(),
             ref_vision=ref_vis.numpy(), ref_itm=ref_itm.numpy())
    print(f"saved {CFG['out']}: hidden {tuple(hidden.shape)}, ref_itm {tuple(ref_itm.shape)}")


if __name__ == "__main__":
    main()
