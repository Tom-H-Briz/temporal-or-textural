"""
Smoke test for train_sae.py — verifies the full pipeline on a tiny subset.

Scope: model load, hook shape (tier-1 asserts), SAE forward, loss finiteness,
tier-3 VideoMAE regression guard.

dim_mean is not available pre-training, so a zero tensor is used here.
This is a connectivity and shape test, not a representation quality test.

Target runtime: ~2 minutes on GH200.
Usage: uv run python notebooks/train_sae_smoketest.py
       MODEL_NAME=timesformer uv run python notebooks/train_sae_smoketest.py

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
    SAE_CONFIG,
    build_loaders,
    build_loss_fn,
    build_split,
    setup_model,
    setup_sae,
    train_epoch,
    validate,
)
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY

OUTPUT_DIR  = ROOT / "outputs" / "sae"
OUTPUT_FILE = OUTPUT_DIR / "train_sae_smoketest.txt"

SMOKE_CFG = {
    **CFG,
    "train_clips": 100,
    "epochs":      1,
    "batch_size":  4,
    "num_workers": 2,
    "checkpoint":  str(OUTPUT_DIR / "sae_smoke.pt"),
}


def main() -> bool:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pass = True

    with open(OUTPUT_FILE, "w") as out_file:

        def log(msg: str = "") -> None:
            print(msg)
            out_file.write(msg + "\n")
            out_file.flush()

        model_cfg        = MODEL_REGISTRY[SMOKE_CFG["model_name"]]
        num_patch_tokens = SMOKE_CFG["num_patch_tokens"]
        hidden_dim       = SMOKE_CFG["hidden_dim"]
        cls_offset       = model_cfg["cls_offset"]
        device           = SMOKE_CFG["device"]

        log(f"Device:    {device}")
        checkpoint = CHECKPOINT_REGISTRY[(SMOKE_CFG["model_name"], "ssv2")]
        log(f"Model:     {SMOKE_CFG['model_name']} ({checkpoint})")
        log(f"Layer:     {SMOKE_CFG['layer']}")
        log(f"nb_concepts={SMOKE_CFG['nb_concepts']}  top_k={SMOKE_CFG['top_k']:,}")
        log(f"cls_offset={cls_offset}  num_patch_tokens={num_patch_tokens}")

        # Zero dim_mean — connectivity test only, not representation quality.
        dim_mean = torch.zeros(hidden_dim, device=device)

        wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
        log(f"WandB:     {wandb_mode}")
        wandb.init(
            project=SMOKE_CFG["wandb_project"],
            name=f"smoketest_{SMOKE_CFG['model_name']}",
            config={**SMOKE_CFG, "sae": SAE_CONFIG},
            mode=wandb_mode,
        )

        try:
            # --- model load + hook ---
            log("\nLoading model and data split...")
            model, processor, hook_storage = setup_model(SMOKE_CFG)
            train_paths, val_paths = build_split(SMOKE_CFG)

            if not train_paths:
                log("[FAIL] No training clips found on disk")
                return False

            train_loader, val_loader = build_loaders(train_paths, val_paths, processor, SMOKE_CFG)
            log(f"  {len(train_paths)} train clips  /  {len(val_paths)} val clips")

            # --- shape checks: one forward pass, triggers tier-1 asserts in _hook ---
            log("\nShape checks (tier-1 asserts fire inside hook)...")
            sample = next(iter(train_loader)).to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                    model(pixel_values=sample)

            activations  = hook_storage["activations"].detach()
            expected_act = (SMOKE_CFG["batch_size"], num_patch_tokens, hidden_dim)
            act_ok = tuple(activations.shape) == expected_act
            log(f"  [{'PASS' if act_ok else 'FAIL'}] hook activations: {tuple(activations.shape)}  (expected {expected_act})")
            all_pass = all_pass and act_ok

            # --- SAE forward ---
            log("\nSAE forward pass...")
            sae, optimizer = setup_sae(SMOKE_CFG)
            sae.train()
            tokens = activations[0].float()
            pre_codes, codes, x_hat = sae(tokens)

            codes_ok = tuple(codes.shape) == (num_patch_tokens, SMOKE_CFG["nb_concepts"])
            log(f"  [{'PASS' if codes_ok else 'FAIL'}] codes shape:  {tuple(codes.shape)}")
            all_pass = all_pass and codes_ok

            xhat_ok = tuple(x_hat.shape) == (num_patch_tokens, hidden_dim)
            log(f"  [{'PASS' if xhat_ok else 'FAIL'}] x_hat shape:  {tuple(x_hat.shape)}")
            all_pass = all_pass and xhat_ok

            nnz         = int(codes.bool().sum().item())
            sparsity_ok = nnz == SMOKE_CFG["top_k"]
            log(f"  [{'PASS' if sparsity_ok else 'FAIL'}] codes sparsity: nnz={nnz}  (expected {SMOKE_CFG['top_k']})")
            all_pass = all_pass and sparsity_ok

            loss_fn = build_loss_fn(SMOKE_CFG)
            loss    = loss_fn(tokens, x_hat, pre_codes, codes, sae.get_dictionary())
            loss_ok = torch.isfinite(loss)
            log(f"  [{'PASS' if loss_ok else 'FAIL'}] loss finite:    {loss.item():.4f}")
            all_pass = all_pass and bool(loss_ok)

            # --- training epoch (zero dim_mean — pipeline connectivity only) ---
            log("\nRunning 1 training epoch (zero dim_mean)...")
            avg_loss, _ = train_epoch(
                sae, train_loader, model, hook_storage, optimizer,
                SMOKE_CFG, 0, 0, dim_mean, loss_fn,
            )
            epoch_ok = torch.isfinite(torch.tensor(avg_loss))
            log(f"  [{'PASS' if epoch_ok else 'FAIL'}] epoch loss finite: {avg_loss:.4f}")
            all_pass = all_pass and bool(epoch_ok)

            # --- validation ---
            log("\nRunning validation...")
            if val_paths:
                metrics = validate(sae, val_loader, model, hook_storage, SMOKE_CFG, dim_mean)
                metrics.pop("_feature_counts")

                r2_ok   = torch.isfinite(torch.tensor(metrics["val/r2"]))
                dead_ok = metrics["val/dead_features"] < SMOKE_CFG["nb_concepts"]

                log(f"  [{'PASS' if r2_ok else 'FAIL'}]  R²={metrics['val/r2']:.4f}")
                log(f"  [{'PASS' if dead_ok else 'FAIL'}]  dead features={metrics['val/dead_features']}  (expect < {SMOKE_CFG['nb_concepts']})")
                all_pass = all_pass and bool(r2_ok) and dead_ok
            else:
                log("  [SKIP] no val clips available")

            # --- checkpoint ---
            torch.save(sae.state_dict(), SMOKE_CFG["checkpoint"])
            ckpt_ok = Path(SMOKE_CFG["checkpoint"]).exists()
            log(f"\n  [{'PASS' if ckpt_ok else 'FAIL'}] checkpoint written: {SMOKE_CFG['checkpoint']}")
            all_pass = all_pass and ckpt_ok

            # --- tier-3 regression guard: VideoMAE path unchanged by port ---
            log("\nTier-3 regression guard (VideoMAE)...")
            vmae_cfg = MODEL_REGISTRY["videomae"]
            offset_ok = vmae_cfg["cls_offset"] == 0
            log(f"  [{'PASS' if offset_ok else 'FAIL'}] videomae cls_offset == 0 (slice is identity)")
            all_pass = all_pass and offset_ok

            if SMOKE_CFG["model_name"] == "videomae":
                # Activations already captured above — confirm 1568 patch tokens, nothing dropped.
                patch_ok = activations.shape[1] == vmae_cfg["num_patch_tokens"]
                log(f"  [{'PASS' if patch_ok else 'FAIL'}] VideoMAE hook: {activations.shape[1]} patch tokens (expected {vmae_cfg['num_patch_tokens']})")
                all_pass = all_pass and patch_ok
            else:
                log(f"  [SKIP] running {SMOKE_CFG['model_name']} — re-run with MODEL_NAME=videomae to exercise VideoMAE path")

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
