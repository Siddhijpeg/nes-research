from src.model.loader import load_model_pair, extract_residuals
from src.model.registry import get_num_layers
from src.carrier_intelligence.qaci_pipeline import QACIPipeline

MODEL_ID = 'mistralai/Mistral-7B-v0.3'
FAMILY = 'mistral'

nf4, fp16, tok = load_model_pair(MODEL_ID)

n_layers = get_num_layers(nf4)

print(f'Layers detected: {n_layers}')

residuals = extract_residuals(nf4, fp16, FAMILY)

for lid, r in sorted(residuals.items())[:5]:
    print(f' L{lid:02d} shape={r.shape} mean={r.abs().mean():.5f}')

pipeline = QACIPipeline(total_layers=n_layers)

result = pipeline.select(
    residuals,
    total_payload_bits=10_000
)

print(
    f'QACI selected {result.total_selected} '
    f'carriers across {n_layers} layers'
)