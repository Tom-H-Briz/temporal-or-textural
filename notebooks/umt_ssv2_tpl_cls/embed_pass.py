"""Part B: one VTC embedding pass — SSv2-Template test split videos + 174 template texts.

Repo code paths (no patches): create_dataset ret_eval transform + VidTxtRetEvalDataset,
extract_text_feats / extract_vision_feats / get_sim. VTC only, forward only.
"""
import sys
import json
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
UMT_MM = ROOT / "models" / "unmasked_teacher" / "multi_modality"
sys.path.insert(0, str(UMT_MM))

CONFIG = dict(
    ft_ckpt=ROOT / "models/umt_ckpts/ret_ssv2_tpl_b16_25m.pth",
    anno=ROOT / "models/umt_annos/ssv2_ret_template_val_small.json",
    video_root=ROOT / "data/ssv2_val_set",
    labels=ROOT / "data/ssv2/labels/labels.json",
    out_dir=ROOT / "outputs/umt_ssv2_tpl_cls/raw",
    device="cpu",  # MPS fp32 collapses vision embeddings (CPU/MPS A/B: GT 0.55 vs 0.25 on CPU, flat ~0.34 on MPS)
    num_frames_test=12,
    batch_size_test=16,
    num_workers=0,  # repo transform contains a local lambda -> not picklable across workers
    max_txt_l=25,
)


def stub_decord():
    """Runtime shim: decord has no macOS arm64 wheel; the repo imports it at
    module level in dataset/video_utils.py but never calls it when
    video_reader_type='av'. Injected into sys.modules only — no repo file changed."""
    decord = types.ModuleType("decord")
    bridge = types.ModuleType("decord.bridge")
    bridge.set_bridge = lambda *a, **k: None
    decord.bridge = bridge
    decord.VideoReader = None
    sys.modules["decord"] = decord
    sys.modules["decord.bridge"] = bridge


def ssv2_read_frames_av(video_path, num_frames, sample="rand", fix_start=None, max_num_frames=-1, **kw):
    """Repo read_frames_av, except duration comes from the container when the
    webm stream lacks one (Matroska never sets AVStream.duration; the repo's
    own decord reader doesn't need it). Frame selection, sampling and output
    use the repo's functions unchanged. Local override, no repo file edited."""
    import av
    from dataset.video_utils import get_frame_indices

    reader = av.open(video_path)
    frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    vlen = len(frames)
    stream = reader.streams.video[0]
    duration = (
        stream.duration * float(stream.time_base)
        if stream.duration is not None else reader.duration / 1e6
    )
    fps = vlen / float(duration)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames,
    )
    frames = torch.stack([frames[idx] for idx in frame_indices]).permute(0, 3, 1, 2)
    return frames, frame_indices, duration


def build_model():
    """UMT(is_pretrain=False) per exp/finetuning/ret_ssv2_tpl/b16_25m.py; FT ckpt 0/0 load."""
    from easydict import EasyDict
    from models.umt import UMT
    from models.backbones.bert.tokenization_bert import BertTokenizer
    sys.path.insert(0, str(ROOT / "notebooks/umt_mlm_check"))
    from part_a_structure import model_config_dict

    model_cfg = model_config_dict(is_pretrain=False)
    criterion = model_cfg.pop("criterion")
    cfg = EasyDict(
        model=EasyDict(model_cfg),
        criterion=EasyDict(criterion),
        gradient_checkpointing=False,
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = UMT(config=cfg, tokenizer=tokenizer, is_pretrain=False)
    sd = torch.load(CONFIG["ft_ckpt"], map_location="cpu", weights_only=False)
    msg = model.load_state_dict(sd, strict=False)
    assert not msg.missing_keys and not msg.unexpected_keys, (msg.missing_keys, msg.unexpected_keys)
    model.eval().to(CONFIG["device"])
    return model, tokenizer


def build_eval_config():
    """Repo-shaped flat config for create_dataset('ret_eval', ...) — eval branch only."""
    from easydict import EasyDict

    return EasyDict(
        test_file=dict(val=[str(CONFIG["anno"]), str(CONFIG["video_root"]), "video"]),
        test_types=["val"],
        has_multi_vision_gt=True,
        is_paragraph_retrieval=False,
        inputs=dict(
            image_res=224,
            video_input=EasyDict(
                num_frames=CONFIG["num_frames_test"],
                sample_type="rand",
                num_frames_test=CONFIG["num_frames_test"],
                sample_type_test="middle",
                video_reader_type="av",
                random_aug=False,
            ),
        ),
        model=dict(vision_encoder=dict(name="vit_b16")),
        evaluation=EasyDict(eval_frame_ensemble="concat", eval_offload=True),
    )


def build_loader(cfg):
    """Repo's own create_dataset + create_loader, ret_eval path."""
    from dataset import create_dataset, create_loader

    datasets, names = create_dataset("ret_eval", cfg)
    for d in datasets:  # swap reader only; class/transform/GT maps stay the repo's
        d.video_reader = ssv2_read_frames_av
    loaders = create_loader(
        datasets, [None] * len(datasets),
        batch_size=[CONFIG["batch_size_test"]] * len(datasets),
        num_workers=[CONFIG["num_workers"]] * len(datasets),
        is_trains=[False] * len(datasets),
        collate_fns=[None] * len(datasets),
    )
    return {k: v for k, v in zip(names, loaders)}


def main():
    stub_decord()
    model, tokenizer = build_model()
    cfg = build_eval_config()
    name2loader = build_loader(cfg)
    loader = name2loader["val"]
    dataset = loader.dataset
    print(f"videos={len(dataset.image)} captions={len(dataset.text)}")

    from tasks.retrieval_utils import extract_text_feats, extract_vision_feats
    from models.criterions import get_sim

    device = torch.device(CONFIG["device"])
    with torch.no_grad():
        text_feats, _ = extract_text_feats(dataset.text, CONFIG["max_txt_l"], tokenizer, model, device)
        image_feats, pooled = extract_vision_feats(loader, model, device, cfg)
        vision_proj = model.vision_proj(pooled.to(device))          # (N, T, 512)
        text_proj = model.text_proj(text_feats[:, 0].to(device))    # (174, 512)
        _, sim_t2v = get_sim(vision_proj, text_proj, temp=1.0)      # (174, N) cosine

    return dataset, vision_proj.cpu(), text_proj.cpu(), sim_t2v.cpu(), model.temp.item()


def save_outputs(dataset, vision_proj, text_proj, sim_t2v, temp):
    """Raw saves before any metric: embeddings, S, meta, GT mappings."""
    import pandas as pd

    out = CONFIG["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    frame_unit = torch.nn.functional.normalize(vision_proj, dim=-1)  # per-frame cosine units
    video_emb = frame_unit.mean(1)                                    # (N, 512), reproduces S
    np.save(out / "video_emb_frames.npy", vision_proj.numpy())
    np.save(out / "video_emb.npy", video_emb.numpy())
    np.save(out / "text_emb.npy", text_proj.numpy())
    np.save(out / "S.npy", sim_t2v.numpy())
    labels = json.load(open(CONFIG["labels"]))
    from dataset.utils import pre_text
    strip = lambda s: pre_text(s).replace("[", "").replace("]", "")
    cap2id = {strip(k): int(v) for k, v in labels.items()}
    meta = pd.DataFrame(dict(
        video=dataset.image,
        gt_template_id=[cap2id[strip(dataset.text[dataset.img2txt[i]])] for i in range(len(dataset.image))],
        gt_template=[dataset.text[dataset.img2txt[i]] for i in range(len(dataset.image))],
    ))
    meta.to_csv(out / "meta.csv", index_label="video_index")
    maps = dict(
        txt2img={str(k): v for k, v in dataset.txt2img.items()},
        img2txt={str(k): v for k, v in dataset.img2txt.items()},
        temp=temp, texts=dataset.text, videos=dataset.image,
    )
    json.dump(maps, open(out / "gt_maps.json", "w"))
    print(f"saved {out}: S {sim_t2v.shape}, temp={temp:.4f}")


if __name__ == "__main__":
    save_outputs(*main())
