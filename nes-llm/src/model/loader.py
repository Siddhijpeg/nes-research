import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


NF4_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model_pair(model_id: str, device=None):

    if device is None:
        device = get_device()

    print(f"Loading model: {model_id}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    # NF4 model
    nf4_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=NF4_CONFIG,
        device_map={"": str(device)},
        trust_remote_code=True
    )

    # FP16 model
    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": str(device)},
        trust_remote_code=True
    )

    return nf4_model, fp16_model, tokenizer


def extract_residuals(nf4_model, fp16_model, family: str):

    from src.model.registry import (
        get_layer_module,
        get_num_layers
    )

    n = get_num_layers(nf4_model)

    residuals = {}

    for i in range(n):

        nf4_mlp = get_layer_module(
            nf4_model,
            family,
            i,
            "mlp"
        )

        fp16_mlp = get_layer_module(
            fp16_model,
            family,
            i,
            "mlp"
        )

        nf4_w = nf4_mlp.down_proj.weight
        fp16_w = fp16_mlp.down_proj.weight

        # Move FP16 weights to the same device
        # as the NF4 weights.
        fp16_w = fp16_w.to(nf4_w.device)

        if hasattr(nf4_w, "dequantize"):
            dq = nf4_w.dequantize().float()
        else:
            dq = nf4_w.float()

        residuals[i] = (
            fp16_w.float() - dq
        ).flatten()

    return residuals