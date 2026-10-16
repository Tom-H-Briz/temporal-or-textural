"""Viability check for option B: graft UMT's pretrain MLM head onto the SSv2-finetuned BERT.
Masks each token of N SSv2 val captions one at a time; compares top-1 accuracy / log-prob of the
true token for (pretrain BERT + head) vs (finetuned BERT + grafted head). Null video for both
(one zero vision token): the fusion layers 9-11 always cross-attend, so there is no text-only pass.

Usage (from repo root, LOCAL CPU, UMT's xbert needs transformers 4.24):
  PYTHONPATH=models/unmasked_teacher/multi_modality models/umt_venv/bin/python src/stage3_analysis/umt_mlm_graft_check.py
"""
import json
from pathlib import Path
import numpy as np
import torch

CFG = {
    "pretrain_ckpt": Path("models/umt_ckpts/b16_25m.pth"),
    "finetune_ckpt": Path("models/umt_ckpts/ret_ssv2_tpl_b16_25m.pth"),
    "bert_cfg":      Path("models/unmasked_teacher/multi_modality/configs/config_bert.json"),
    "captions":      Path("data/ssv2/labels/validation.json"),
    "n_captions":    300,
    "seed":          0,
    "max_txt_l":     32,
    "batch_size":    64,
}


def build_mlm(bert_sd, head_sd):
    """xbert BertForMaskedLM (fusion_layer=9) from a BERT state dict + an MLM head state dict.
    Decoder weight is tied to the BERT's own word embeddings (as in UMT pretraining)."""
    from models.backbones.bert.xbert import BertConfig, BertForMaskedLM
    cfg = BertConfig.from_json_file(str(CFG["bert_cfg"]))
    cfg.encoder_width, cfg.fusion_layer = 768, 9
    mlm = BertForMaskedLM(cfg)
    sd = {"bert." + k: v for k, v in bert_sd.items()} | {"cls." + k: v for k, v in head_sd.items()}
    sd["cls.predictions.decoder.weight"] = bert_sd["embeddings.word_embeddings.weight"]
    mlm.load_state_dict(sd, strict=True)
    return mlm.eval()


def strip_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def masked_examples(tok):
    """One example per (caption, real token): that token replaced by [MASK].
    is_obj = the token is a word-piece of one of the caption's placeholders (object nouns)."""
    rows = json.load(open(CFG["captions"]))
    rng = np.random.default_rng(CFG["seed"])
    ids, labels, is_obj = [], [], []
    for r in rng.choice(rows, CFG["n_captions"], replace=False):
        enc = tok(r["label"].lower(), truncation=True, max_length=CFG["max_txt_l"])["input_ids"]
        obj = set(tok(" ".join(r["placeholders"]).lower(), add_special_tokens=False)["input_ids"])
        for pos in range(1, len(enc) - 1):   # skip [CLS] / [SEP]
            ids.append(enc[:pos] + [tok.mask_token_id] + enc[pos + 1:])
            labels.append((pos, enc[pos]))
            is_obj.append(enc[pos] in obj)
    return ids, labels, np.array(is_obj)


@torch.no_grad()
def score(mlm, ids, labels):
    """Per example: (top-1 correct, log-prob of the true token), null video (one zero token)."""
    hit, logp = [], []
    for s in range(0, len(ids), CFG["batch_size"]):
        batch = ids[s:s + CFG["batch_size"]]
        L = max(map(len, batch))
        x = torch.tensor([b + [0] * (L - len(b)) for b in batch])
        vis = torch.zeros(len(batch), 1, 768)
        logits = mlm(x, attention_mask=(x != 0).long(), encoder_hidden_states=vis,
                     encoder_attention_mask=torch.ones(len(batch), 1, dtype=torch.long),
                     return_dict=True, return_logits=True, mode="multi_modal")
        for i, (pos, true) in enumerate(labels[s:s + CFG["batch_size"]]):
            lp = logits[i, pos].log_softmax(-1)
            hit.append(int(lp.argmax() == true))
            logp.append(lp[true].item())
    return np.array(hit), np.array(logp)


def main():
    from models.backbones.bert.tokenization_bert import BertTokenizer
    load = lambda p: (lambda sd: sd.get("model", sd))(torch.load(p, map_location="cpu", weights_only=False))
    pre, fin = load(CFG["pretrain_ckpt"]), load(CFG["finetune_ckpt"])
    head = strip_prefix(pre, "text_encoder.cls.")
    models = {"pretrain": build_mlm(strip_prefix(pre, "text_encoder.bert."), head),
              "graft":    build_mlm(strip_prefix(fin, "text_encoder."), head)}
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    ids, labels, is_obj = masked_examples(tok)
    print(f"{CFG['n_captions']} captions, {len(ids)} masked tokens ({is_obj.sum()} object)")
    for name, mlm in models.items():
        hit, logp = score(mlm, ids, labels)
        for part, m in [("all", slice(None)), ("object", is_obj), ("other", ~is_obj)]:
            print(f"{name:9s} {part:7s} top-1 {hit[m].mean():.3f}  mean log-p {logp[m].mean():.2f}")


if __name__ == "__main__":
    main()
