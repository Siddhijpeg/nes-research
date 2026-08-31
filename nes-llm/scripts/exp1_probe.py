from src.model.loader import load_model_pair, extract_residuals
from src.carrier_intelligence.qaci_pipeline import QACIPipeline

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    # The residual extraction probe currently hits unsupported MPS matmul/dequantize
    # paths (e.g. aten::dequantize.self on MPS). Falling back to CPU avoids the
    # runtime shape mismatch and keeps the experiment runnable on Apple Silicon.
    print("MPS detected; forcing CPU for this residual probe because the quantized residual path is not stable on MPS.")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


MODELS = [
    # ("meta-llama/Llama-2-7b", "llama", 32),
    # ("meta-llama/Llama-3.1-8B", "llama", 32),
    # ("mistralai/Mistral-7B-v0.3", "mistral", 32)
    # ("google/gemma-2-9b", "gemma", 42)
    ("google/gemma-2-2b", "gemma", 26)
    # ("Qwen/Qwen2.5-7B", "qwen", 28)
    # ("Qwen/Qwen2.5-3B", "qwen", 36)
    # ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", 22)
]


for MODEL_ID, FAMILY, EXPECTED_LAYERS in MODELS:

    print("\n" + "=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Family: {FAMILY}")
    print(f"Expected layers: {EXPECTED_LAYERS}")
    print("=" * 70)

    try:
        nf4, fp16, tok = load_model_pair(
            MODEL_ID,
            device=DEVICE
        )

        n_layers = len(
            nf4.model.layers
        )

        print(f"Layers detected: {n_layers}")

        if n_layers != EXPECTED_LAYERS:
            print(
                f"WARNING: expected {EXPECTED_LAYERS}, "
                f"but detected {n_layers}"
            )

        residuals = extract_residuals(
            nf4,
            fp16,
            FAMILY
        )

        print(f"Residual layers found: {len(residuals)}")

        for lid, r in sorted(residuals.items())[:5]:
            print(
                f" L{lid:02d} "
                f"shape={tuple(r.shape)} "
                f"mean={r.abs().mean().item():.6f}"
            )

        pipeline = QACIPipeline(
            total_layers=n_layers
        )

        result = pipeline.select(
            residuals,
            total_payload_bits=10_000
        )

        print(
            f"QACI selected {result.total_selected} "
            f"carriers across {n_layers} layers"
        )

    except Exception as e:
        print(f"FAILED: {MODEL_ID}")
        print(f"{type(e).__name__}: {e}")