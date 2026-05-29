"""
Smoke test for train_sae.py — verifies the full pipeline on a tiny subset.

Reuses all functions from train_sae directly. Checks shapes, sparsity,
loss finiteness, and metric sanity before committing to the full run.

Target runtime: ~2 minutes on GH200.
Usage: uv run python notebooks/train_sae_smoketest.py

WandB: runs online if WANDB_API_KEY is set (verifies connectivity), disabled otherwise.
"""

import os
import sys
import traceback
from pathlib import Path

import torch
import wandb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from train_sae import (
    CFG,
    build_loaders,
    build_split,
    setup_sae,
    setup_videomae,
    train_epoch,
    validate,
)
from sae.losses import top_k_auxiliary_loss

OUTPUT_DIR = ROOT / "outputs" / "sae"
OUTPUT_FILE = OUTPUT_DIR / "train_sae_smoketest.txt"

SMOKE_CFG = {
    **CFG,
    "train_clips": 100,
    "epochs": 1,
    "batch_size": 4,
    "num_workers": 2,
    "checkpoint": str(OUTPUT_DIR / "sae_layer_7_smoke.pt"),
    "wandb_run": "sae_layer7_smoketest",
}


def main() -> bool:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pass = True

    with open(OUTPUT_FILE, "w") as out_file:

        def log(msg: str = "") -> None:
            print(msg)
            out_file.write(msg + "\n")
            out_file.flush()

        log(f"Device: {SMOKE_CFG['device']}")
        log(f"Model:  {SMOKE_CFG['model_id']}")
        log(f"Layer:  {SMOKE_CFG['layer']}   nb_concepts={SMOKE_CFG['nb_concepts']}   top_k={SMOKE_CFG['top_k']:,}")

        wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
        log(f"WandB:  {wandb_mode}")
        wandb.init(
            project=SMOKE_CFG["wandb_project"],
            name=SMOKE_CFG["wandb_run"],
            config=SMOKE_CFG,
            mode=wandb_mode,
        )

        try:
            # --- setup ---
            log("\nLoading model and data split...")
            videomae, processor, hook_storage = setup_videomae(SMOKE_CFG)
            train_paths, val_paths = build_split(SMOKE_CFG)

            if not train_paths:
                log("[FAIL] No training clips found on disk")
                return False

            train_loader, val_loader = build_loaders(train_paths, val_paths, processor, SMOKE_CFG)
            sae, optimizer = setup_sae(SMOKE_CFG)
            log(f"  {len(train_paths)} train clips  /  {len(val_paths)} val clips")

            # --- shape checks: one forward pass before training ---
            log("\nShape checks...")
            device = SMOKE_CFG["device"]

            sample = next(iter(train_loader)).to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                    videomae(pixel_values=sample)

            activations = hook_storage["activations"].detach()
            expected_act = (SMOKE_CFG["batch_size"], 1568, SMOKE_CFG["input_dim"])
            act_ok = tuple(activations.shape) == expected_act
            log(f"  [{'PASS' if act_ok else 'FAIL'}] hook activations: {tuple(activations.shape)}  (expected {expected_act})")
            all_pass = all_pass and act_ok

            tokens = activations[0].float()
            sae.train()
            pre_codes, codes, x_hat = sae(tokens)

            codes_shape_ok = tuple(codes.shape) == (1568, SMOKE_CFG["nb_concepts"])
            log(f"  [{'PASS' if codes_shape_ok else 'FAIL'}] codes shape:       {tuple(codes.shape)}")
            all_pass = all_pass and codes_shape_ok

            xhat_shape_ok = tuple(x_hat.shape) == (1568, SMOKE_CFG["input_dim"])
            log(f"  [{'PASS' if xhat_shape_ok else 'FAIL'}] x_hat shape:        {tuple(x_hat.shape)}")
            all_pass = all_pass and xhat_shape_ok

            nnz = int(codes.bool().sum().item())
            expected_nnz = SMOKE_CFG["top_k"]
            sparsity_ok = nnz == expected_nnz
            log(f"  [{'PASS' if sparsity_ok else 'FAIL'}] codes sparsity:     nnz={nnz}  (expected {expected_nnz})")
            all_pass = all_pass and sparsity_ok

            loss = top_k_auxiliary_loss(
                tokens, x_hat, pre_codes, codes, sae.get_dictionary(),
                penalty=SMOKE_CFG["aux_loss_coeff"],
            )
            loss_ok = torch.isfinite(loss)
            log(f"  [{'PASS' if loss_ok else 'FAIL'}] loss finite:        {loss.item():.4f}")
            all_pass = all_pass and bool(loss_ok)

            # --- training epoch ---
            log("\nRunning 1 training epoch...")
            avg_loss, _ = train_epoch(sae, train_loader, videomae, hook_storage, optimizer, SMOKE_CFG, 0, 0)
            epoch_ok = torch.isfinite(torch.tensor(avg_loss))
            log(f"  [{'PASS' if epoch_ok else 'FAIL'}] epoch loss finite:  {avg_loss:.4f}")
            all_pass = all_pass and bool(epoch_ok)

            # --- validation ---
            log("\nRunning validation...")
            if val_paths:
                metrics = validate(sae, val_loader, videomae, hook_storage, SMOKE_CFG)
                metrics.pop("_feature_counts")

                r2_ok = torch.isfinite(torch.tensor(metrics["val/r2"]))
                l0_ok = 50 <= metrics["val/l0"] <= 80
                dead_ok = metrics["val/dead_features"] < SMOKE_CFG["nb_concepts"]

                log(f"  [{'PASS' if r2_ok else 'FAIL'}]  R²={metrics['val/r2']:.4f}")
                log(f"  [{'PASS' if l0_ok else 'FAIL'}]  L0={metrics['val/l0']:.1f}  (expect ~64)")
                log(f"  [{'PASS' if dead_ok else 'FAIL'}]  dead features={metrics['val/dead_features']}  (expect < {SMOKE_CFG['nb_concepts']})")
                all_pass = all_pass and bool(r2_ok) and l0_ok and dead_ok
            else:
                log("  [SKIP] no val clips available")

            # --- checkpoint ---
            torch.save(sae.state_dict(), SMOKE_CFG["checkpoint"])
            ckpt_ok = Path(SMOKE_CFG["checkpoint"]).exists()
            log(f"\n  [{'PASS' if ckpt_ok else 'FAIL'}] checkpoint written: {SMOKE_CFG['checkpoint']}")
            all_pass = all_pass and ckpt_ok

        except Exception:
            log("\n[CRASH] Unhandled exception:")
            log(traceback.format_exc())
            all_pass = False

        log(f"\nResult: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    wandb.finish()
    print(f"\nOutput: {OUTPUT_FILE}")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
