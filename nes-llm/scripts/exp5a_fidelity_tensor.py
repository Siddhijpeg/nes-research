from src.model.model_loader import load_model_pair, extract_residuals
from src.evaluation.fidelity_validator import FidelityValidator
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.core.types import EmbeddingConfig


MODELS = [
    ("meta-llama/Llama-3.1-8B", "llama", 32),
    ("mistralai/Mistral-7B-v0.3", "mistral", 32),
    ("google/gemma-2-9b", "gemma", 42),
    ("Qwen/Qwen2.5-7B", "qwen", 28),
    ("Qwen/Qwen2.5-3B", "qwen", 36),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", 22),
    ("microsoft/Phi-3-mini-4k-instruct", "phi3", 32),
]


for model_id, family, n_layers in MODELS:

    print("\n" + "=" * 70)
    print(f"Model: {model_id}")
    print("=" * 70)

    nf4, fp16, _ = load_model_pair(model_id)

    residuals = extract_residuals(
        nf4,
        fp16,
        family
    )

    config = EmbeddingConfig(
        total_payload_bits=50_000,
        model_family=family,
        num_hidden_layers=n_layers
    )

    embed_result = IntelligentEmbedder(config).embed(
        "A" * 6000,
        residuals
    )

    validator = FidelityValidator(
        max_ppl_degradation=0.02
    )

    result = validator.validate_tensors(
        residuals,
        embed_result.embedded_residuals
    )

    print(result.report())