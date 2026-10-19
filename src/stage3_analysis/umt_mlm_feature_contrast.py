"""UMT MLM feature contrast (brief mlm_feature_contrast_brief_191026.md): how much each
L9 vision SAE feature contributes to the refit MLM head's masked pair-word decision
T_ab = logit(word_a) - logit(word_b), under R / C / Rev, on Phase 3's six pairs' clips
(both splits), then Phase 3's selection on the R selection clips and a comparison with
Phase 3's L9 T_vtm selections.

Per clip and condition (batch 1): masked caption (first occurrence of the clip's class
word -> [MASK]) -> mlm_text (0-8, no grad) -> z leaf at the L9 SAE trained threshold
(x = x9 + decode(z - z0), Phase 3 convention) -> finish_tower (10-11 + norm) ->
mlm_fusion (9-11) -> RefitHead at the mask; contribution c = dT_ab/dz * z per
(token, latent), stored token-level sparse exactly as umt_vision_pair_contrast.

Modes (uv run python src/stage3_analysis/umt_mlm_feature_contrast.py <mode>):
  check    CPU: no-grad equivalence + FD gradient check (2-3 clips), mask counts (all clips)
  extract  GPU/SLURM: sparse/{cond}/{clip_id}.npz + per_clip.parquet + skipped_clips.csv
  select   CPU: discriminative/largest/mass_controls unchanged -> mlm_pair_selection.csv
  compare  CPU: vs Phase 3 T_vtm -> comparison_features/overlap + summary_by_condition
Usage: MAX_CLIPS=20 uv run python src/stage3_analysis/umt_mlm_feature_contrast.py all
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
from umt_mlm_refit import (bert_weights, encode, get_tokenizer, load_refit_head,
                           mlm_fusion, mlm_hidden, mlm_text, umt_pre_text)
from umt_bridge_phase3 import (discriminative, finish_tower, largest, mass_controls,
                               tower_to_layer)
from umt_vision_pair_contrast import apply_condition, to_sparse

CFG = {
    "layer": 9, "sae_k": 64, "job_label": "7ep",      # the vision SAE Phase 3 used
    "pairs": {  # (a, b): (word_a, word_b) — §1 table; both must be single word-pieces
        (46, 5): ("opening", "closing"), (168, 165): ("upwards", "downwards"),
        (45, 43): ("up", "down"), (44, 41): ("towards", "away"),
        (42, 40): ("closer", "away"), (37, 36): ("closer", "away")},
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "max_clips":       int(os.environ.get("MAX_CLIPS", 0)),   # smoke: half per class x split
    "pairs_env":       os.environ.get("PAIRS", ""),           # subset, e.g. "46_5;45_43"
    "n_sel": 10, "max_other": 1 / 3,                          # Phase 3 selection parameters
    "phase3_dir": ROOT / "outputs/analysis/umt_vision_text_bridge/phase3",
    "refit_dir":   ROOT / "outputs/analysis/umt_mlm_refit",
    "out_dir":     ROOT / "outputs/analysis/umt_mlm_feature_contrast",
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
}


def pair_rows(cfg):
    """Phase 3's clips for the six pairs with split tags: per_clip_ap.parquet joined with
    g34_clips.csv (g34 = held-out test, never read by selection). PAIRS / MAX_CLIPS
    (smoke: half the cap per class AND split — Phase 3's held-out set is the first ~37
    ids per pair, so a class-only cap would pick g34 only) restrict."""
    df = pd.read_parquet(cfg["phase3_dir"] / "per_clip_ap.parquet")
    df = df[df.apply(lambda r: (r.a, r.b) in cfg["pairs"], axis=1)].copy()
    g34 = set(pd.read_csv(cfg["phase3_dir"] / "g34_clips.csv", dtype={"clip_id": str}).clip_id)
    df["split"] = np.where(df.clip_id.isin(g34), "g34", "selection")
    if cfg["pairs_env"]:
        keep = {tuple(map(int, p.split("_"))) for p in cfg["pairs_env"].split(";")}
        df = df[df.apply(lambda r: (r.a, r.b) in keep, axis=1)]
    if cfg["max_clips"]:
        n = max(1, cfg["max_clips"] // 2)
        df = df.sort_values("clip_id").groupby(["a", "b", "true_class", "split"],
                                               group_keys=False).head(n)
    return df.sort_values("clip_id").reset_index(drop=True)


def word_ids(cfg, tok):
    """{(a, b): (id_a, id_b)}: each pair's class words as single word-pieces (§1)."""
    out = {}
    for (a, b), (wa, wb) in cfg["pairs"].items():
        pieces = [tok(w, add_special_tokens=False)["input_ids"] for w in (wa, wb)]
        assert all(len(p) == 1 for p in pieces), f"multi-piece class word in pair {(a, b)}"
        out[(a, b)] = (pieces[0][0], pieces[1][0])
    return out


def masked_pos(enc, word_id):
    """First occurrence (positions 1.., past [CLS]) of the class word's word-piece.
    -> (pos, status): ok | missing | repeated (the clip is skipped, never guessed)."""
    hits = [p for p in range(1, len(enc)) if enc[p] == word_id]
    return (hits[0], "ok") if len(hits) == 1 else (None, "repeated" if hits else "missing")


def clip_texts(cfg, rows, s):
    """Masked caption tensors per clip + skip records. word_a on A clips, word_b on B
    clips; -> {clip_id: (ids (1, L), attn, mask_pos, id_a, id_b)} on the device."""
    labels = {c["id"]: c["label"] for c in json.load(open(cfg["validation_path"]))}
    wid = word_ids(cfg, s["tok"])
    texts, skips = {}, []
    for r in rows.itertuples():
        is_a = r.true_class == r.a
        id_a, id_b = wid[(r.a, r.b)]
        word = id_a if is_a else id_b
        enc, _ = encode(s["tok"], s["pre_text"](labels[r.clip_id]), 25)
        pos, st = masked_pos(enc, word)
        if st != "ok":
            skips.append(dict(clip_id=r.clip_id, a=r.a, b=r.b, true_class=r.true_class,
                              split=r.split, reason=st))
            continue
        ids = torch.tensor([[*enc[:pos], s["tok"].mask_token_id, *enc[pos + 1:]]],
                           device=cfg["device"])
        texts[r.clip_id] = (ids, torch.ones_like(ids), pos, id_a, id_b)
    return texts, skips


def setup(cfg):
    """UMT tower, finetuned BERT weights + refit head (umt_mlm_refit loaders), and the
    L9 vision SAE with trained threshold + dim_mean, resolved as phase3_setup does."""
    from ToT_utils import load_sae_eval, resolve_sae_checkpoint
    from umt_wrapper import UMTClassifier
    res = resolve_sae_checkpoint("umt", cfg["layer"], dataset_name="ssv2",
                                 sae_k=cfg["sae_k"], job_label=cfg["job_label"])
    sae = load_sae_eval(res["sae_path"], input_dim=768, tokens_per_clip=2352,
                        device=cfg["device"], sae_k=cfg["sae_k"])
    return dict(model=UMTClassifier.from_pretrained().to(cfg["device"]).eval(), sae=sae,
                dim_mean=torch.load(res["dim_mean_path"], weights_only=True).to(cfg["device"]),
                head=load_refit_head(cfg["refit_dir"])[0],
                w={k: v.to(cfg["device"]) for k, v in bert_weights("finetune").items()},
                tok=get_tokenizer(), pre_text=umt_pre_text())


def t_of_z(s, x9, z0, h8, add, pos, id_a, id_b, z):
    """T_ab through the grad path with codes z: x = x9 + decode(z - z0) -> finish_tower
    (blocks 10-11 + norm) -> mlm_fusion (9-11) -> RefitHead at the masked position."""
    x = x9 + s["sae"].decode(z - z0).unsqueeze(0)
    vis = finish_tower(s["model"], x, layer=9)[0].unsqueeze(0)   # (1, 2352, 768) for _kv
    logits = s["head"](mlm_fusion(h8, add, vis, s["w"])[0, pos])
    return logits[id_a] - logits[id_b], logits


def clip_pass(s, x9, text):
    """One clip's grad pass (Phase 3's z-leaf convention): eval codes z0, z graph leaf,
    -> (T_ab, logits, z, c) with per-(token, latent) contribution c = dT_ab/dz * z."""
    ids, attn, pos, id_a, id_b = text
    with torch.no_grad():
        h8, add = mlm_text(ids, attn, s["w"])
        _, z0 = s["sae"].encode((x9[0] - s["dim_mean"]).float())
    z = z0.detach().clone().float().requires_grad_(True)
    t, logits = t_of_z(s, x9, z0, h8, add, pos, id_a, id_b, z)
    g = torch.autograd.grad(t, z)[0].detach()
    return float(t.detach()), logits.detach(), z.detach(), g * z.detach()


def clip_loader(rows, cfg, condition):
    """Batch-1 (pv, row-index) loader: same dataset class, sampler and processor as
    umt_vision_pair_contrast.clip_loader (R / C = UMT native-frame shuffle), but labels
    are row indices so the dataset's decode-retry skip-to-next is detectable."""
    from ToT_utils import MODEL_REGISTRY, get_frame_sampler
    from perturb_accuracy_tf_kinetics import PerturbedKineticsDataset
    from umt_wrapper import UMTProcessor
    mcfg = MODEL_REGISTRY["umt"]
    paths = [Path(cfg["video_dir"]) / f"{i}.webm" for i in rows.clip_id]
    ds = PerturbedKineticsDataset(paths, rows.clip_id.tolist(), rows.index.tolist(),
                                  UMTProcessor(), mcfg["num_frames"],
                                  get_frame_sampler("ssv2", mcfg),
                                  "C" if condition == "C" else "R")
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                       num_workers=cfg["num_workers"])


def clip_row(r, cond, text, t, logits, z):
    """per_clip.parquet fields for one clip-condition (§3 D1)."""
    _, _, pos, id_a, id_b = text
    lp = logits.float().log_softmax(-1)
    return dict(clip_id=r.clip_id, a=r.a, b=r.b, true_class=r.true_class, condition=cond,
                split=r.split, mask_pos=pos, word_a_id=id_a, word_b_id=id_b, T_ab=t,
                logp_a=float(lp[id_a]), logp_b=float(lp[id_b]),
                top1_id=int(logits.argmax()), l0_per_token=float((z > 0).sum(1).float().mean()))


def extract(cfg, s):
    """D1: sparse/{cond}/{clip_id}.npz + one per_clip.parquet row per clip-condition.
    R and C come from the dataset (C = UMT native-frame shuffle), Rev = apply_condition;
    the masked caption is identical in all three conditions."""
    rows = pair_rows(cfg)
    texts, skips = clip_texts(cfg, rows, s)
    keep = rows[rows.clip_id.isin(texts)].reset_index(drop=True)
    print(f"{len(rows)} clips: {len(keep)} maskable, {len(skips)} skipped (word missing/repeated)")
    recs, undec, t0 = [], [], time.time()
    for cond in ("R", "C", "Rev"):
        (cfg["out_dir"] / "sparse" / cond).mkdir(parents=True, exist_ok=True)
        filled = np.zeros(len(keep), bool)
        for pv, y in clip_loader(keep, cfg, cond):
            i = int(y)
            if filled[i]:                      # decode-retry duplicate of a written clip
                continue
            r = keep.iloc[i]
            pv = apply_condition(pv, cond).to(cfg["device"])
            x9 = tower_to_layer(s["model"], pv, layer=cfg["layer"])
            t, logits, z, c = clip_pass(s, x9, texts[r.clip_id])
            np.savez_compressed(cfg["out_dir"] / "sparse" / cond / f"{r.clip_id}.npz",
                                **to_sparse(z, c, "umt"))
            recs.append(clip_row(r, cond, texts[r.clip_id], t, logits, z))
            filled[i] = True
        undec += [dict(clip_id=r.clip_id, a=r.a, b=r.b, true_class=r.true_class,
                       split=r.split, condition=cond, reason="undecodable")
                  for _, r in keep[~filled].iterrows()]
        done, el = len(recs), time.time() - t0
        print(f"  {cond}: {int(filled.sum())}/{len(keep)} clips  {el / max(done, 1) * 1e3:.0f} ms/clip-cond"
              f"  projected full {(2418 * 3) * el / max(done, 1) / 3600:.1f} h", flush=True)
    finish_extract(cfg, skips, undec, recs, t0)


def finish_extract(cfg, skips, undec, recs, t0):
    """Write per_clip.parquet + skipped_clips.csv; report per-pair skip counts and the
    actual sparse size per clip-condition (the §6.2 size check)."""
    pd.DataFrame(recs).to_parquet(cfg["out_dir"] / "per_clip.parquet", index=False)
    all_sk = pd.DataFrame(skips + undec)
    all_sk.to_csv(cfg["out_dir"] / "skipped_clips.csv", index=False)
    if len(all_sk):
        print(all_sk.groupby(["a", "b", "reason"]).size().rename("n").to_string())
    files = list((cfg["out_dir"] / "sparse").rglob("*.npz"))
    mb = sum(f.stat().st_size for f in files) / 1e6 / max(len(files), 1)
    print(f"extract done: {len(recs)} clip-conditions, {len(files)} sparse files "
          f"({mb:.2f} MB each), {(time.time() - t0) / 60:.1f} min")


def first_clips(cfg, rows, texts, n):
    """The first n maskable clips (clip_id order), index-reset for the loader."""
    sub = rows[rows.clip_id.isin(texts)].head(n).reset_index(drop=True)
    return sub, clip_loader(sub, cfg, "R")


def check_equiv(cfg, s, rows, texts, n=3):
    """§7.1: T_ab from the gradient path (z = z0) vs the no-grad path (model.vision
    tokens -> mlm_hidden -> RefitHead) on the same decoded clips."""
    sub, loader = first_clips(cfg, rows, texts, n)
    errs = []
    for pv, y in loader:
        r = sub.iloc[int(y)]
        _, attn, pos, id_a, id_b = texts[r.clip_id]
        with torch.no_grad():
            vis = s["model"].vision(pv.to(cfg["device"]).permute(0, 2, 1, 3, 4),
                                    keep_temporal=True)[0]
            lg = s["head"](mlm_hidden(texts[r.clip_id][0], attn, vis, s["w"])[0, pos])
            ref = float(lg[id_a] - lg[id_b])
        x9 = tower_to_layer(s["model"], pv.to(cfg["device"]), layer=cfg["layer"])
        t = clip_pass(s, x9, texts[r.clip_id])[0]
        errs.append(abs(t - ref))
        print(f"  {r.clip_id}: grad-path T_ab {t:.4f}  no-grad {ref:.4f}  |diff| {errs[-1]:.2e}")
    assert errs and max(errs) < 1e-3, f"equivalence failed: {errs}"
    print(f"equivalence ok (n={len(errs)}): max |diff| = {max(errs):.2e}")


def fp64_setup(s):
    """Float64 copies of the t_of_z path, as umt_bridge_phase3.g1b_check_b builds them
    (the eval SAE's cached fused dictionary is stripped, deep-copied, reattached fp64)."""
    from copy import deepcopy
    sae = s["sae"]
    fused, sae.dictionary._fused_dictionary = sae.dictionary._fused_dictionary, None
    s64 = dict(s, model=deepcopy(s["model"]).double(), sae=deepcopy(sae).double(),
               head=deepcopy(s["head"]).double(),
               w={k: v.double() for k, v in s["w"].items()},
               dim_mean=s["dim_mean"].double())
    sae.dictionary._fused_dictionary = fused
    s64["sae"].dictionary._fused_dictionary = fused.detach().double()
    s64["sae"].running_threshold = torch.as_tensor(sae.running_threshold).double()
    return s64


def check_fd(cfg, s, rows, texts, n_clips=1):
    """§7.2 (updated brief): _fd_check on f(z) = T_ab in float64 — the gradient direction
    plus 4 random unit directions, restricted to the active support by running the check
    in the active subspace (scatter via index_put), eps 1e-2 as _fd_check defaults."""
    from umt_fusion_grad import _fd_check
    s64 = fp64_setup(s)
    sub, loader = first_clips(cfg, rows, texts, n_clips)
    for pv, y in loader:
        r = sub.iloc[int(y)]
        ids_t, attn, pos, id_a, id_b = texts[r.clip_id]
        x9 = tower_to_layer(s["model"], pv.to(cfg["device"]), layer=cfg["layer"]).double()
        with torch.no_grad():
            h8, add = mlm_text(ids_t, attn, s64["w"])
            _, z0 = s64["sae"].encode((x9[0] - s64["dim_mean"]).double())
        flat = z0.flatten()
        idx = (z0 > 0).flatten().nonzero(as_tuple=True)[0]

        def f(xa):
            zz = flat.index_put((idx,), xa).view(z0.shape)
            return t_of_z(s64, x9, z0, h8, add, pos, id_a, id_b, zz)[0]
        out = _fd_check(f, flat[idx].clone().requires_grad_(True), n_dir=4)
        for ad, fd, rel in out:
            print(f"  dir: autograd {ad:.6f}  fd {fd:.6f}  rel err {rel:.2e}")
        assert max(r_[2] for r_ in out) <= 1e-3, f"FD check failed on {r.clip_id}"
        print(f"FD check ok on {r.clip_id}: max rel err {max(r_[2] for r_ in out):.2e}")


def check_masks(cfg, s):
    """§7.4: per pair, found / missing / repeated counts over ALL six pairs' clips
    (caption side only; MAX_CLIPS/PAIRS ignored). The masked token is the class word's
    word-piece on every kept clip by construction (masked_pos)."""
    rows = pair_rows(dict(cfg, max_clips=0, pairs_env=""))
    texts, skips = clip_texts(cfg, rows, s)
    tab = rows.groupby(["a", "b"]).size().rename("clips").to_frame().join(
        rows[rows.clip_id.isin(texts)].groupby(["a", "b"]).size().rename("found"))
    if skips:
        df = pd.DataFrame(skips)
        tab = tab.join(pd.crosstab([df.a, df.b], df.reason))
    tab = tab.fillna(0).astype(int)
    n_m = sum(x["reason"] == "missing" for x in skips)
    n_r = sum(x["reason"] == "repeated" for x in skips)
    print(tab.to_string())
    print(f"mask counts: {len(rows)} clips -> {len(texts)} found, {n_m} missing, {n_r} repeated")
    return tab


def run_check(cfg):
    """§6.1 checkpoint (CPU): §7.1 equivalence + §7.2 FD on 2-3 clips, then §7.4 mask
    counts over all clips."""
    s = setup(cfg)
    rows = pair_rows(cfg)
    texts, _ = clip_texts(cfg, rows, s)
    check_equiv(cfg, s, rows, texts, n=3)
    check_fd(cfg, s, rows, texts)
    check_masks(cfg, s)


def feature_vecs(sparse_dir, cond, clip_id, n_lat=6144):
    """(contrib, z_mass) per feature for one clip: contribution summed over tokens (c),
    and z summed over tokens, from the token-level sparse npz."""
    with np.load(sparse_dir / cond / f"{clip_id}.npz") as d:
        c = np.bincount(d["lat"], weights=d["dfa"].astype(np.float32), minlength=n_lat)
        m = np.bincount(d["lat"], weights=d["z"].astype(np.float32), minlength=n_lat)
    return c, m


def activation_cols(zmass, is_a):
    """Class-mean activation mass on A / B and the min/max ratio (0-1; 0 if inactive) —
    saved for every feature, no threshold set on it (§3 D2)."""
    m_a, m_b = zmass[is_a].mean(0), zmass[~is_a].mean(0)
    hi = np.maximum(m_a, m_b)
    return m_a, m_b, np.where(hi > 0, np.minimum(m_a, m_b) / np.where(hi > 0, hi, 1), 0.0)


def select(cfg):
    """D2: Phase 3's selection on R selection-split clips: discriminative (n_sel 10,
    max_other 1/3) + mass_controls (mean per-clip z sum) + largest (n 10), imported
    unchanged; contributions are NOT flipped for B clips (T_ab is already a-minus-b)."""
    df = pd.read_parquet(cfg["out_dir"] / "per_clip.parquet")
    df = df[(df.condition == "R") & (df.split == "selection")]
    rows, tabs = [], []
    for (a, b), g in df.groupby(["a", "b"]):
        if g.empty:                       # a smoke pair can lack selection-split clips
            continue
        C, M = zip(*(feature_vecs(cfg["out_dir"] / "sparse", "R", c) for c in g.clip_id))
        D, zmass = np.stack(C), np.stack(M)
        is_a = (g.true_class == a).to_numpy()
        sel, tab = discriminative(D, is_a, cfg["n_sel"], cfg["max_other"])
        ctrl, gap = mass_controls(zmass.mean(0), sel, tab.ratio.to_numpy(), cfg["max_other"])
        mx, share = largest(D, cfg["n_sel"])
        m_a, m_b, act_ratio = activation_cols(zmass, is_a)
        tab = tab.assign(a=a, b=b, act_mass_a=m_a, act_mass_b=m_b, act_ratio=act_ratio)
        tabs.append(tab)
        rows.append(tab.loc[sel].assign(target="T_ab", rank=np.arange(len(sel)),
                    control=ctrl, mass=zmass.mean(0)[sel], control_mass=zmass.mean(0)[ctrl],
                    mass_gap=gap, max_feature=mx, max_share=share,
                    n_a=int(is_a.sum()), n_b=int((~is_a).sum())))
    if not rows:
        print("select: no selection-split R clips (smoke subset too small); nothing written")
        return
    sel_df = pd.concat(rows, ignore_index=True)
    cols = ["feature", "evidence", "bias", "mean_a", "mean_b", "ratio", "passed", "target",
            "rank", "control", "mass", "control_mass", "mass_gap", "max_feature", "max_share",
            "n_a", "n_b", "a", "b", "act_mass_a", "act_mass_b", "act_ratio"]
    sel_df[cols].to_csv(cfg["out_dir"] / "mlm_pair_selection.csv", index=False)
    pd.concat(tabs, ignore_index=True).to_parquet(cfg["out_dir"] / "selection_table.parquet")
    print(sel_df.groupby(["a", "b"]).agg(n=("feature", "size"), top=("feature", "first"),
          max_gap=("mass_gap", "max")).to_string())


def category(ratio, max_other):
    """Contribution category from the ratio: <0 sign-reversed, <= max_other silent,
    else same-sign (user-confirmed boundaries, max_other = 1/3)."""
    return "sign_reversed" if ratio < 0 else "silent" if ratio <= max_other else "same_sign"


def load_ap_vec(phase3_dir, cid):
    """(6144,) full-flip AP_T_vtm for one clip, uncached (lru would hold ~2.4k arrays)."""
    with np.load(phase3_dir / "sparse_ap" / f"{cid}.npz") as d:
        return d["ap_T_vtm"].astype(np.float32)


def overlap_row(a, b, mlm_sel, mlm_max, vtm_sel, vtm_max):
    """§3 D3 overlap per pair: counts + feature ids of the four intersections."""
    sets = dict(mlm_sel=set(mlm_sel), mlm_max=set(mlm_max),
                vtm_sel=set(vtm_sel), vtm_max=set(vtm_max))
    row = dict(a=a, b=b, n_mlm_sel=len(mlm_sel), n_vtm_sel=len(vtm_sel))
    for x, y in (("mlm_sel", "vtm_sel"), ("mlm_max", "vtm_max"),
                 ("mlm_sel", "vtm_max"), ("mlm_max", "vtm_sel")):
        both = sorted(sets[x] & sets[y])
        row[f"n_{x}__{y}"] = len(both)
        row[f"features_{x}__{y}"] = ",".join(map(str, both))
    return row


def feature_rows(cfg, a, b, lists, v_sel, tab_all, per):
    """One row per feature in the union of both readouts' selected + max lists (§3 D3,
    updated): VTM evidence/bias/ratio recomputed on this run's R selection clips (the
    same clips as the MLM values); Phase 3's own table values sit alongside as *_p3
    reference columns. Membership flags in_vtm_sel / in_vtm_max stay Phase 3's lists."""
    mlm_sel, mlm_max, vtm_sel, vtm_max = lists
    sub = per[(per.a == a) & (per.b == b) & (per.split == "selection")]
    rclips = sub[sub.condition == "R"]
    is_a = (rclips.true_class == a).to_numpy()
    tab = tab_all[(tab_all.a == a) & (tab_all.b == b)].set_index("feature")
    v_idx = v_sel.set_index("feature")
    ap = np.stack([load_ap_vec(cfg["phase3_dir"], c) for c in rclips.clip_id])
    vtab = discriminative(-0.5 * ap * np.where(is_a, 1.0, -1.0)[:, None], is_a,
                          cfg["n_sel"], cfg["max_other"])[1]
    p3 = pair_rows(dict(cfg, max_clips=0, pairs_env=""))
    p3_sel = set(p3[(p3.a == a) & (p3.b == b) & (p3.split == "selection")].clip_id)
    n_missing = len(p3_sel - set(rclips.clip_id))
    d_ev = max((abs(float(vtab.evidence[f]) - float(v_idx.loc[f, "evidence"]))
                for f in v_idx.index), default=0.0)
    d_ra = max((abs(float(vtab.ratio[f]) - float(v_idx.loc[f, "ratio"]))
                for f in v_idx.index), default=0.0)
    print(f"  {a}_{b}: {n_missing}/{len(p3_sel)} Phase 3 selection clips missing here; "
          f"max |recomputed - phase3| over its {len(v_idx)} table features: "
          f"evidence {d_ev:.4f}, ratio {d_ra:.4f}")
    cc = {}
    for cd, g in sub.groupby("condition", sort=False):
        ia = (g.true_class == a).to_numpy()
        cc[cd] = (ia, np.stack([feature_vecs(cfg["out_dir"] / "sparse", cd, c)[0]
                                for c in g.clip_id]))
    rows = []
    for f in sorted(set(mlm_sel) | set(mlm_max) | set(vtm_sel) | set(vtm_max)):
        t, in_p3 = tab.loc[f], f in v_idx.index
        ra_v = float(vtab.ratio[f])
        row = dict(a=a, b=b, feature=f, in_mlm_sel=f in mlm_sel, in_mlm_max=f in mlm_max,
                   in_vtm_sel=f in vtm_sel, in_vtm_max=f in vtm_max,
                   evidence_mlm=float(t.evidence), bias_mlm=float(t.bias), ratio_mlm=float(t.ratio),
                   category_mlm=category(float(t.ratio), cfg["max_other"]),
                   evidence_vtm=float(vtab.evidence[f]), bias_vtm=float(vtab.bias[f]),
                   ratio_vtm=ra_v, category_vtm=category(ra_v, cfg["max_other"]),
                   in_phase3_table=in_p3,
                   evidence_vtm_p3=float(v_idx.loc[f, "evidence"]) if in_p3 else np.nan,
                   bias_vtm_p3=float(v_idx.loc[f, "bias"]) if in_p3 else np.nan,
                   ratio_vtm_p3=float(v_idx.loc[f, "ratio"]) if in_p3 else np.nan,
                   act_ratio=float(t.act_ratio))
        for cd in ("R", "C", "Rev"):
            row[f"c_mlm_a_{cd}"] = float(cc[cd][1][cc[cd][0], f].mean())
            row[f"c_mlm_b_{cd}"] = float(cc[cd][1][~cc[cd][0], f].mean())
        rows.append(row)
    return rows


def summary_rows(cfg, a, b, m_sel, per):
    """§3 D3 pair x condition summary: accuracy from sign(T_ab) (A clips > 0, B < 0),
    mean T_ab per class, and the mean summed contribution of the MLM selected / max /
    control sets. Splits are kept apart; the selection split never reads g34."""
    sets = dict(sel=m_sel.feature.tolist(), max=m_sel.max_feature.tolist(),
                ctrl=m_sel.control.tolist())
    rows = []
    for cond in ("R", "C", "Rev"):
        for split in ("selection", "g34"):
            g = per[(per.a == a) & (per.b == b) & (per.condition == cond) & (per.split == split)]
            if g.empty:
                continue
            is_a = (g.true_class == a).to_numpy()
            t = g.T_ab.to_numpy()
            C = np.stack([feature_vecs(cfg["out_dir"] / "sparse", cond, c)[0]
                          for c in g.clip_id])
            row = dict(a=a, b=b, condition=cond, split=split, n=len(g),
                       acc=float(((t > 0) == is_a).mean()),
                       mean_T_ab_a=float(t[is_a].mean()), mean_T_ab_b=float(t[~is_a].mean()))
            for nm, fs in sets.items():
                row[f"c_sum_{nm}"] = float(C[:, fs].sum(1).mean())
            rows.append(row)
    return rows


def compare(cfg):
    """D3: MLM (T_ab) vs Phase 3 L9 T_vtm, target T_vtm only, on the extracted pairs ->
    comparison_overlap.csv, comparison_features.csv, summary_by_condition.csv."""
    pc = pd.read_csv(cfg["phase3_dir"] / "pair_selection.csv")
    mlm = pd.read_csv(cfg["out_dir"] / "mlm_pair_selection.csv")
    per = pd.read_parquet(cfg["out_dir"] / "per_clip.parquet")
    tab_all = pd.read_parquet(cfg["out_dir"] / "selection_table.parquet")
    ov, feat, summ = [], [], []
    for (a, b), g in mlm.groupby(["a", "b"]):
        m = g.sort_values("rank")
        v = pc[(pc.target == "T_vtm") & (pc.a == a) & (pc.b == b)].sort_values("rank")
        lists = (m.feature.tolist(), m.max_feature.tolist(),
                 v.feature.tolist(), v.max_feature.tolist())
        ov.append(overlap_row(a, b, *lists))
        feat += feature_rows(cfg, a, b, lists, v, tab_all, per)
        summ += summary_rows(cfg, a, b, m, per)
    pd.DataFrame(ov).to_csv(cfg["out_dir"] / "comparison_overlap.csv", index=False)
    pd.DataFrame(feat).to_csv(cfg["out_dir"] / "comparison_features.csv", index=False)
    pd.DataFrame(summ).to_csv(cfg["out_dir"] / "summary_by_condition.csv", index=False)
    print(f"compare: {len(ov)} pairs, {len(feat)} union feature rows, {len(summ)} summary rows")


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 paths, as the references
    torch.backends.cudnn.allow_tf32 = False
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = dict(CFG)
    if cfg["max_clips"]:                              # smoke never overwrites the real run
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_smoke")
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    print(f"mode={mode} device={cfg['device']} out={cfg['out_dir']} "
          f"max_clips={cfg['max_clips'] or 'all'} pairs={cfg['pairs_env'] or 'all six'}")
    if mode in ("check", "all"):
        run_check(cfg)
    if mode in ("extract", "all"):
        extract(cfg, setup(cfg))
    if mode in ("select", "all"):
        select(cfg)
    if mode in ("compare", "all"):
        compare(cfg)


if __name__ == "__main__":
    main()
