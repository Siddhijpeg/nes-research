"""
Experiment 5A — Tensor-Level Fidelity Validation

Measures distortion introduced by NES embedding using residual tensors.

This is a fast proxy experiment; real model perplexity is evaluated
separately in Experiment 5B.
"""

import os
import torch

from src.model.model_loader import load_model_pair, extract_residuals
from src.evaluation.fidelity_validator import FidelityValidator
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.core.types import EmbeddingConfig


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
MODELS = [
    ("meta-llama/Llama-3.1-8B", "llama"),
    ("mistralai/Mistral-7B-v0.3", "mistral"),
    ("google/gemma-2-9b", "gemma"),
    ("Qwen/Qwen2.5-7B", "qwen"),
    ("Qwen/Qwen2.5-3B", "qwen"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama"),
    ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
]


# ------------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------------
PAYLOAD_BITS = 50_000
MESSAGE = "A" * 6_000

validator = FidelityValidator(
    max_ppl_degradation=0.02
)


# ------------------------------------------------------------------
# Run Experiment 5A
# ------------------------------------------------------------------
for model_id, family in MODELS:

    print("\n" + "=" * 70)
    print(f"MODEL: {model_id}")
    print("=" * 70)

    try:

        # ----------------------------------------------------------
        # 1. Load NF4 + FP16 model pair
        # ----------------------------------------------------------
        # Model loading / forward computation uses the selected device.
        nf4_model, fp16_model, _ = load_model_pair(
            model_id,
            device=DEVICE
        )

        # ----------------------------------------------------------
        # 2. Detect actual architecture depth
        # ----------------------------------------------------------
        actual_layers = len(nf4_model.model.layers)

        print(f"Detected layers: {actual_layers}")

        # ----------------------------------------------------------
        # 3. Extract real residuals
        #
        # Returns:
        #   residuals
        #   FP16 weights
        #   quantized weights
        # ----------------------------------------------------------
        residuals, fp16_weights, quantized_weights = extract_residuals(
            nf4_model=nf4_model,
            fp16_model=fp16_model,
            family=family,
            model_id=model_id,
        )

        total_capacity = sum(
            tensor.numel()
            for tensor in residuals.values()
        )

        print(f"Residual tensors: {len(residuals)}")
        print(f"Total residual capacity: {total_capacity:,} values")

        # ----------------------------------------------------------
        # 4. Configure NES
        # ----------------------------------------------------------
        config = EmbeddingConfig(
            total_payload_bits=PAYLOAD_BITS,
            model_family=family,
            num_hidden_layers=actual_layers,
        )

        # ----------------------------------------------------------
        # 5. Embed 50,000-bit payload
        # ----------------------------------------------------------
        embedder = IntelligentEmbedder(config)

        embed_result = embedder.embed(
            MESSAGE,
            residuals,
            fp16_weights=fp16_weights,
            quantized_weights=quantized_weights,
        )

        # ----------------------------------------------------------
        # 6. Tensor-level fidelity validation
        # ----------------------------------------------------------
        result = validator.validate_tensors(
            residuals,
            embed_result.embedded_residuals,
        )

        # ----------------------------------------------------------
        # 7. Report
        # ----------------------------------------------------------
        print("\n" + result.report())

        print(f"  Payload bits       : {PAYLOAD_BITS:,}")
        print(f"  Carrier tensors    : {len(embed_result.embedded_residuals)}")

    except Exception as exc:

        print(f"\nERROR: {type(exc).__name__}: {exc}")

    finally:

        # ----------------------------------------------------------
        # 8. Release model memory before next model
        # ----------------------------------------------------------
        try:
            del nf4_model
            del fp16_model
        except NameError:
            pass

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print("\nMemory cleared.")