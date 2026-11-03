"""Do a layer's chosen object / pair features have twins at the other layers? (CPU, local, 03/11/26)
Decoder directions of the L5 / L7 / L9 UMT SAEs share the residual stream basis, so for every
chosen feature (umt_{TASK}_contrast_L{L}/features.csv) find its nearest decoder direction (cosine)
in each other layer's SAE, and whether that match was chosen there for the same unit (object, or
pair + home class). Scale: the same best-match cosine for 1000 random latents.
Usage: TASK=object uv run python src/stage3_analysis/umt_layer_direction_match.py   (TASK=pair)
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src")]
CFG = dict(task=os.environ.get("TASK", "object"), layers=(5, 7, 9), sae_k=64, job_label="7ep", n_random=1000, seed=0)
CFG["out"] = ROOT / f"outputs/analysis/umt_{CFG['task']}_contrast/direction_match.csv"


def dictionaries(cfg):
    """{layer: (n_lat, 768) unit-norm decoder directions} from each layer's trained SAE."""
    from ToT_utils import load_sae_eval, resolve_sae_checkpoint
    out = {}
    for L in cfg["layers"]:
        r = resolve_sae_checkpoint("umt", L, dataset_name="ssv2", sae_k=cfg["sae_k"], job_label=cfg["job_label"])
        D = load_sae_eval(r["sae_path"], input_dim=768, tokens_per_clip=2352, device="cpu", sae_k=cfg["sae_k"]).get_dictionary()
        out[L] = torch.nn.functional.normalize(D.detach().float(), dim=1)
    return out


def chosen(cfg):
    """All layers' chosen features: layer, latent, unit (object, or pair_home)."""
    rows = []
    for L in cfg["layers"]:
        f = pd.read_csv(ROOT / f"outputs/analysis/umt_{cfg['task']}_contrast_L{L}/features.csv")
        unit = f.object + ("_" + f.home.astype(str) if "home" in f else "")
        rows += [dict(layer=L, latent=int(x), unit=u) for x, u in zip(f.latent, unit)]
    return pd.DataFrame(rows)


def main(cfg):
    D, F = dictionaries(cfg), chosen(cfg)
    rng, rows = np.random.default_rng(cfg["seed"]), []
    for r in F.itertuples():
        for L in cfg["layers"]:
            if L == r.layer:
                continue
            cos = D[L] @ D[r.layer][r.latent]
            j = int(cos.argmax())
            rand = (D[L] @ D[r.layer][rng.choice(len(D[r.layer]), cfg["n_random"], replace=False)].T).max(0).values
            twin = F[(F.layer == L) & (F.latent == j)]
            rows.append(dict(layer=r.layer, latent=r.latent, unit=r.unit, to_layer=L, match=j, cos=float(cos[j]),
                             random_p95=float(np.percentile(rand, 95)), match_chosen_for=",".join(twin.unit) or "-",
                             same_unit=r.unit in set(twin.unit)))
    out = pd.DataFrame(rows)
    out.to_csv(cfg["out"], index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main(CFG)
