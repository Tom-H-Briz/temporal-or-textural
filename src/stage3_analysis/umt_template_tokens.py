"""Local one-off (models/umt_venv): tokenize the 174 templates with UMT's vendored
BertTokenizer exactly as umt_vtm_text_cache.py did, dumping tokens + word starts
so the bridge script needs no tokenizer dependency. Run from repo root:
  PYTHONPATH=models/unmasked_teacher/multi_modality models/umt_venv/bin/python src/stage3_analysis/umt_template_tokens.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, "models/unmasked_teacher/multi_modality")
from models.backbones.bert.tokenization_bert import BertTokenizer

texts = [str(t) for t in np.load("notebooks/umt_ssv2_template_bert_l8.npz")["texts"]]
tok = BertTokenizer.from_pretrained("bert-base-uncased")
enc = tok(texts, padding="max_length", truncation=True, max_length=25)
mask = np.array(enc["attention_mask"])
cache_mask = np.load("notebooks/umt_ssv2_template_bert_l8.npz")["mask"]
assert (mask == cache_mask).all(), "vendored tokenizer no longer reproduces the cache"
rows = []
for i, t in enumerate(texts):
    ids = [int(x) for x in enc["input_ids"][i][: int(mask[i].sum())]]
    rows.append({"text": t, "tokens": tok.convert_ids_to_tokens(ids)})
json.dump(rows, open("notebooks/umt_ssv2_template_tokens.json", "w"), indent=1)
print(f"saved notebooks/umt_ssv2_template_tokens.json; mask==cache for all 174")
print("example:", rows[0])
