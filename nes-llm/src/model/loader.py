from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
NF4_CONFIG = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type='nf4',
bnb_4bit_use_double_quant=True,
bnb_4bit_compute_dtype=torch.float16,
)
def load_model_pair(model_id: str, device: str = 'mps:0'):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    nf4_model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=NF4_CONFIG,
    device_map=device, trust_remote_code=True,
    )
    fp16_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16,
    device_map='cpu', trust_remote_code=True,
    )
    return nf4_model, fp16_model, tokenizer
def extract_residuals(nf4_model, fp16_model, family: str) -> dict:
    from src.model.registry import get_layer_module, get_num_layers
    n = get_num_layers(nf4_model)
    residuals = {}
    for i in range(n):
        nf4_mlp = get_layer_module(nf4_model, family, i, 'mlp')
        fp16_mlp = get_layer_module(fp16_model, family, i, 'mlp')
        nf4_w = nf4_mlp.down_proj.weight
        fp16_w = fp16_mlp.down_proj.weight.to(nf4_w.device)
        dq = nf4_w.dequantize().float() if hasattr(nf4_w,'dequantize') else nf4_w.float()
        residuals[i] = (fp16_w.float() - dq).flatten()
    return residuals