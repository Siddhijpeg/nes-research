# scripts/exp2_residual_fingerprint.py

from src.carrier_intelligence.layer_profiler import LayerProfiler
import json
from src.model.loader import load_model_pair,extract_residuals
from src.model.registry import get_num_layers
for MODEL_ID, FAMILY in [
    ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'llama'),
    ('mistralai/Mistral-7B-v0.3', 'mistral'),
    ('google/gemma-2-9b', 'gemma'),
    ('Qwen/Qwen2.5-7B', 'qwen'),
]:
    nf4, fp16, _ = load_model_pair(MODEL_ID)

    n_layers = get_num_layers(nf4)

    residuals = extract_residuals(nf4, fp16, FAMILY)

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
        'model': MODEL_ID,
        'num_layers': n_layers,
        'mean_quality': sum(
            p['quality_score'] for p in profiles
        ) / len(profiles),
        'mean_mag_mean': sum(
            p['mag_mean'] for p in profiles
        ) / len(profiles),
        'mean_entropy': sum(
            p['entropy'] for p in profiles
        ) / len(profiles),
    }

    print(json.dumps(stats, indent=2))

    with open(f'residual_profile_{FAMILY}.json', 'w') as f:
        json.dump(
            {
                **stats,
                'per_layer': profiles
            },
            f,
            indent=2
        )