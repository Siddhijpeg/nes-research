"""
Precompute and cache model weights + NF4 residuals.

The expensive operations:
    1. Load FP16 model
    2. Load NF4 model
    3. Dequantize NF4 weights
    4. Compute residuals

are performed once and stored on disk.

Later experiments can load the cached tensors directly.

Usage:
    python3 -m scripts.cache_model
"""

import os
import json
from pathlib import Path

import torch

from src.model.loader import load_model_pair, extract_residuals


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.1-8B"
FAMILY = "llama"

CACHE_ROOT = Path("cache/models")
MODEL_CACHE = CACHE_ROOT / "llama-3.1-8b"

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    print("=" * 70)
    print("NES MODEL PRECOMPUTATION")
    print("=" * 70)

    print(f"Model : {MODEL_ID}")
    print(f"Family: {FAMILY}")
    print(f"Device: {DEVICE}")
    print(f"Cache : {MODEL_CACHE}")
    print("=" * 70)

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Prevent accidental overwriting
    # --------------------------------------------------------------

    metadata_file = MODEL_CACHE / "metadata.json"

    if metadata_file.exists():
        print("\nCache already exists.")
        print(f"Location: {MODEL_CACHE}")
        print("Delete the cache manually if you want to regenerate it.")
        return

    # --------------------------------------------------------------
    # Load models
    # --------------------------------------------------------------

    nf4_model, fp16_model, _ = load_model_pair(
        MODEL_ID,
        device=DEVICE
    )

    # --------------------------------------------------------------
    # Compute residuals + reference weights
    # --------------------------------------------------------------

    print("\nComputing residuals...")

    residuals, fp16_weights, quantized_weights = extract_residuals(
        nf4_model,
        fp16_model,
        FAMILY
    )

    # --------------------------------------------------------------
    # Save tensors
    # --------------------------------------------------------------

    print("\nSaving cached tensors...")

    torch.save(
        residuals,
        MODEL_CACHE / "residuals.pt"
    )

    torch.save(
        fp16_weights,
        MODEL_CACHE / "fp16_weights.pt"
    )

    torch.save(
        quantized_weights,
        MODEL_CACHE / "quantized_weights.pt"
    )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    metadata = {
        "model_id": MODEL_ID,
        "family": FAMILY,
        "num_layers": len(residuals),
        "device_used_for_precomputation": str(DEVICE),
        "dtype_fp16_reference": "float32",
        "dtype_quantized_dequantized": "float32",
        "description": (
            "FP16 reference weights, dequantized NF4 weights, "
            "and FP16-NF4 residuals."
        ),
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------

    del nf4_model
    del fp16_model

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --------------------------------------------------------------
    # Verification
    # --------------------------------------------------------------

    print("\nVerifying cache...")

    cached_residuals = torch.load(
        MODEL_CACHE / "residuals.pt",
        map_location="cpu",
        weights_only=True
    )

    cached_fp16 = torch.load(
        MODEL_CACHE / "fp16_weights.pt",
        map_location="cpu",
        weights_only=True
    )

    cached_quantized = torch.load(
        MODEL_CACHE / "quantized_weights.pt",
        map_location="cpu",
        weights_only=True
    )

    assert len(cached_residuals) == len(cached_fp16)
    assert len(cached_residuals) == len(cached_quantized)

    for layer_id in cached_residuals:

        assert (
            cached_residuals[layer_id].numel()
            == cached_fp16[layer_id].numel()
        )

        assert (
            cached_residuals[layer_id].numel()
            == cached_quantized[layer_id].numel()
        )

    print("Cache verification: PASSED")

    print("\n" + "=" * 70)
    print("CACHE CREATED SUCCESSFULLY")
    print("=" * 70)

    print(f"Layers cached: {len(cached_residuals)}")
    print(f"Location     : {MODEL_CACHE}")


if __name__ == "__main__":
    main()