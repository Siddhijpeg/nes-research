import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# scripts/exp2_residual_fingerprint.py

from src.carrier_intelligence.layer_profiler import LayerProfiler
from src.model.loader import load_model_pair,extract_residuals
from src.model.registry import get_num_layers
import json
import torch


DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


MODELS = [
    # ("meta-llama/Llama-2-7b", "llama", 32),
    # ("meta-llama/Llama-3.1-8B", "llama", 32)
    # ("mistralai/Mistral-7B-v0.3", "mistral", 32)
    # ("google/gemma-2-9b", "gemma", 42),
    # ("google/gemma-2-2b", "gemma", 26)
    # ("Qwen/Qwen2.5-7B", "qwen", 28)
    # ("Qwen/Qwen2.5-3B", "qwen", 36)
    # ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", 22)
]


for MODEL_ID, FAMILY, EXPECTED_LAYERS in MODELS:

    print("\n" + "=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Family: {FAMILY}")
    print("=" * 70)

    try:
        nf4, fp16, _ = load_model_pair(
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

        profiler = LayerProfiler()

        profiles = [
            profiler.profile(
                residuals[i],
                i,
                total_layers=n_layers
            )
            for i in sorted(residuals)
        ]

        stats = {
            "model": MODEL_ID,
            "family": FAMILY,
            "num_layers": n_layers,
            "mean_quality": (
                sum(
                    p["quality_score"]
                    for p in profiles
                ) / len(profiles)
            ),
            "mean_mag_mean": (
                sum(
                    p["mag_mean"]
                    for p in profiles
                ) / len(profiles)
            ),
            "mean_entropy": (
                sum(
                    p["entropy"]
                    for p in profiles
                ) / len(profiles)
            ),
        }

        print(json.dumps(stats, indent=2))

        output_file = (
            f"residual_profile_{FAMILY}"
            f"_{MODEL_ID.split('/')[-1]}.json"
        )

        with open(output_file, "w") as f:
            json.dump(
                {
                    **stats,
                    "per_layer": profiles
                },
                f,
                indent=2
            )

        print(f"Saved: {output_file}")

    except Exception as e:
        print(f"FAILED: {MODEL_ID}")
        print(f"{type(e).__name__}: {e}")