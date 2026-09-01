"""
Precompute and cache model weights + NF4 residuals.

The expensive operations:

    1. Load FP16 model
    2. Load NF4 model
    3. Dequantize NF4 weights
    4. Compute FP16 - NF4 residuals

are performed once and cached layer-by-layer.

Later experiments can load the cached tensors directly.

Usage:
    python3 -m scripts.cache_model
"""

import os
from pathlib import Path

import torch

from src.model.loader import (
    load_model_pair,
    extract_residuals,
)


# ==============================================================
# CONFIGURATION
# ==============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B"
FAMILY = "llama"

CACHE_ROOT = Path("cache/models")


# ==============================================================
# DEVICE
# ==============================================================

# The models run on MPS.
#
# CPU is used only inside extract_residuals() for the
# BitsAndBytes NF4 dequantization operation.

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 70)
    print("NES MODEL PRECOMPUTATION")
    print("=" * 70)

    print(f"Model : {MODEL_ID}")
    print(f"Family: {FAMILY}")
    print(f"Device: {DEVICE}")
    print(f"Cache : {CACHE_ROOT}")
    print("=" * 70)

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Load models
    # ----------------------------------------------------------

    nf4_model, fp16_model, _ = load_model_pair(
        MODEL_ID,
        device=DEVICE,
    )

    # ----------------------------------------------------------
    # Compute + cache residuals
    # ----------------------------------------------------------

    print("\nComputing residuals...")

    residuals, fp16_weights, quantized_weights = extract_residuals(
        nf4_model=nf4_model,
        fp16_model=fp16_model,
        family=FAMILY,
        model_id=MODEL_ID,
        cache_root=str(CACHE_ROOT),
        force_recompute=False,
    )

    # ----------------------------------------------------------
    # Basic verification
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRECOMPUTATION VERIFICATION")
    print("=" * 70)

    assert len(residuals) == len(fp16_weights)
    assert len(residuals) == len(quantized_weights)

    for layer_id in residuals:

        residual = residuals[layer_id]
        fp16_w = fp16_weights[layer_id]
        quantized_w = quantized_weights[layer_id]

        assert residual.numel() == fp16_w.numel(), (
            f"Layer {layer_id}: residual/FP16 size mismatch"
        )

        assert residual.numel() == quantized_w.numel(), (
            f"Layer {layer_id}: residual/NF4 size mismatch"
        )

    print(
        f"Layers processed : {len(residuals)}"
    )

    print("Tensor verification: PASSED")

    # ----------------------------------------------------------
    # Release models
    # ----------------------------------------------------------

    del nf4_model
    del fp16_model

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ----------------------------------------------------------
    # Complete
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 70)

    print(
        "All layer tensors have been processed through "
        "the persistent ModelTensorCache."
    )

    print(
        f"Cache location: {CACHE_ROOT}"
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()