# scripts/exp3_clean_ber.py
from src.model.loader import load_model_pair, extract_residuals
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.extraction.decrypt_pipeline import DecryptPipeline
from src.core.types import EmbeddingConfig

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


MODELS = [
    # ("meta-llama/Llama-2-7b", "llama", 32),
    ("meta-llama/Llama-3.1-8B", "llama", 32),
    # ("mistralai/Mistral-7B-v0.3", "mistral", 32),
    # ("google/gemma-2-9b", "gemma", 42),
    # ("google/gemma-2-2b", "gemma", 26),
    # ("Qwen/Qwen2.5-7B", "qwen", 28),
    # ("Qwen/Qwen2.5-3B", "qwen", 36),
    # ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", 22),
]


MESSAGE = "NES multi-model steganography validation."


for model_id, family, n_layers in MODELS:

    print("\n" + "=" * 70)
    print(f"Model: {model_id}")
    print(f"Family: {family}")
    print(f"Expected layers: {n_layers}")
    print("=" * 70)

    try:

        nf4, fp16, _ = load_model_pair(
            model_id,
            device=DEVICE
        )

        actual_layers = len(
            nf4.model.layers
        )

        print(f"Detected layers: {actual_layers}")

        if actual_layers != n_layers:
            print(
                f"WARNING: expected {n_layers}, "
                f"detected {actual_layers}"
            )

        residuals, fp16_weights, quantized_weights = extract_residuals(
            nf4,
            fp16,
            family
        )

        config = EmbeddingConfig(
            total_payload_bits=2_000,
            model_family=family,
            num_hidden_layers=actual_layers
        )

        embed_result = IntelligentEmbedder(
            config
        ).embed(
            MESSAGE,
            residuals,
            fp16_weights=fp16_weights,
            quantized_weights=quantized_weights
        )

        recovered, s = DecryptPipeline(
            key=embed_result.key
        ).run(
            embed_result.embedded_residuals,
            embed_result.carrier_indices
        )

        ok = recovered == MESSAGE

        print(f"Recovered: {recovered}")
        print(f"BER: {0.0 if ok else 1.0:.3f}")
        print(f"PASS: {ok}")

    except Exception as e:

        print(f"FAILED: {model_id}")
        print(f"{type(e).__name__}: {e}")