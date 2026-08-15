# scripts/exp3_clean_ber.py
from src.model.loader import load_model_pair, extract_residuals
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.extraction.decrypt_pipeline import DecryptPipeline
from src.core.types import EmbeddingConfig

MODELS = [
    ('meta-llama/Llama-2-7b', 'llama', 32),
    ('meta-llama/Llama-3.1-8B', 'llama3', 32),
    ('mistralai/Mistral-7B-v0.3', 'mistral', 32),
    ('google/gemma-2-9b', 'gemma', 42),
    ('google/gemma-2-2b', 'gemma', 26),
    ('Qwen/Qwen2.5-7B', 'qwen', 28),
    ('Qwen/Qwen2.5-3B', 'qwen', 36),
    ('microsoft/Phi-3-mini-4k-instruct', 'phi3', 32),
    ('microsoft/Phi-3.5-mini-instruct', 'phi3', 32),
    ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'llama', 22),
]

MESSAGE = 'NES multi-model steganography validation.'

for model_id, family, n_layers in MODELS:
    nf4, fp16, _ = load_model_pair(model_id)

    residuals = extract_residuals(nf4, fp16, family)

    config = EmbeddingConfig(
        total_payload_bits=2_000,
        model_family=family,
        num_hidden_layers=n_layers
    )

    embed_result = IntelligentEmbedder(config).embed(
        MESSAGE,
        residuals
    )

    recovered, s = DecryptPipeline(
        key=embed_result.key
    ).run(
        embed_result.embedded_residuals,
        embed_result.carrier_indices
    )

    ok = recovered == MESSAGE