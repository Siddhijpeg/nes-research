from datasets import load_dataset

from src.model.model_loader import load_model_pair, extract_residuals
from src.evaluation.fidelity_validator import FidelityValidator
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.core.types import EmbeddingConfig
from src.model.model_loader import apply_residuals_to_model


MODELS = [
    ("meta-llama/Llama-3.1-8B", "llama", 32),
    ("mistralai/Mistral-7B-v0.3", "mistral", 32),
    ("google/gemma-2-9b", "gemma", 42),
    ("Qwen/Qwen2.5-7B", "qwen", 28),
    ("Qwen/Qwen2.5-3B", "qwen", 36),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", 22),
    ("microsoft/Phi-3-mini-4k-instruct", "phi3", 32),
]


dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="validation"
)

texts = [
    t
    for t in dataset["text"]
    if len(t.strip()) > 50
][:200]


for model_id, family, n_layers in MODELS:

    print("\n" + "=" * 70)
    print(f"Model: {model_id}")
    print("=" * 70)

    nf4, fp16, tok = load_model_pair(model_id)

    residuals = extract_residuals(
        nf4,
        fp16,
        family
    )

    validator = FidelityValidator(
        max_ppl_degradation=0.02
    )

    ppl_base = validator.validate_perplexity(
        nf4,
        tok,
        texts
    )

    embed_result = IntelligentEmbedder(
        EmbeddingConfig(
            total_payload_bits=50_000,
            model_family=family,
            num_hidden_layers=n_layers
        )
    ).embed(
        "A" * 6000,
        residuals
    )

    apply_residuals_to_model(
        nf4,
        fp16,
        embed_result.embedded_residuals,
        family
    )

    ppl_embed = validator.validate_perplexity(
        nf4,
        tok,
        texts
    )

    result = validator.compare_perplexity(
        ppl_base,
        ppl_embed
    )

    print(
        f"{model_id} "
        f"{ppl_base:.3f} -> "
        f"{ppl_embed:.3f} "
        f"D={result.ppl_degradation * 100:.3f}% "
        f"{result.status}"
    )