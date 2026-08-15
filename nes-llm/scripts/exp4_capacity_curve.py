from src.core.types import EmbeddingConfig
from src.extraction.decrypt_pipeline import DecryptPipeline
from src.model.loader import extract_residuals
from src.embedding.intelligent_embedder import IntelligentEmbedder
from scripts.exp3_clean_ber import MODELS
PAYLOAD_SIZES = [1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
for model_id, family, n_layers in MODELS:
    residuals = extract_residuals(...)
    total_capacity = sum(r.numel() for r in residuals.values())
    print(f'{model_id}: capacity = {total_capacity:,} bits')
    for n_bits in PAYLOAD_SIZES:
        if n_bits > total_capacity * 0.9:
            print(f' {n_bits:>9,} bits — SKIPPED (exceeds 90% capacity)')
        continue
config = EmbeddingConfig(total_payload_bits=n_bits, model_family=family,
num_hidden_layers=n_layers)
result = IntelligentEmbedder(config).embed('A'*(n_bits//8), residuals)
rec, st = DecryptPipeline(key=result.key).run(
result.embedded_residuals, result.carrier_indices)
ber = 0.0 if (st['success'] and len(rec) > 0) else 1.0
print(f' {n_bits:>9,} bits — BER={ber:.4f}')