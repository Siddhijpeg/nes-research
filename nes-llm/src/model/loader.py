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

    nf4_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=NF4_CONFIG,
        device_map={"": str(device)},
        trust_remote_code=True
    )

    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": str(device)},
        trust_remote_code=True
    )

    return nf4_model, fp16_model, tokenizer


def extract_residuals(nf4_model, fp16_model, family: str):
    from src.model.registry import get_layer_module, get_num_layers

    n = get_num_layers(nf4_model)

    residuals = {}
    fp16_weights = {}
    quantized_weights = {}

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

        # FP16 reference weight
        fp16_w = fp16_mlp.down_proj.weight.detach().float()

        # NF4 quantized weight
        nf4_w = nf4_mlp.down_proj.weight

        # Dequantize NF4 weight
        dq = (
            nf4_w.dequantize().float()
            if hasattr(nf4_w, "dequantize")
            else nf4_w.float()
        )

<<<<<<< Updated upstream
        if hasattr(nf4_w, "quant_state"):
            try:
                import bitsandbytes.functional as bnb_func
                dq = bnb_func.dequantize_4bit(
                    nf4_w,
                    nf4_w.quant_state,
                ).float()
            except Exception:
                dq = nf4_w.dequantize().float()
        elif hasattr(nf4_w, "dequantize"):
            dq = nf4_w.dequantize().float()
        else:
            dq = nf4_w.float()

        if dq.shape != fp16_w.shape:
            if dq.numel() == fp16_w.numel():
                dq = dq.reshape(fp16_w.shape)
            else:
                raise RuntimeError(
                    f"Layer {i}: NF4 has {dq.numel()} elements, "
                    f"but FP16 has {fp16_w.numel()} elements."
                )

        residuals[i] = (
            fp16_w.float() - dq
        ).flatten()
=======
        # Make sure shapes match
        if dq.numel() != fp16_w.numel():
            raise RuntimeError(
                f"Layer {i}: shape mismatch: "
                f"FP16={fp16_w.shape}, NF4={dq.shape}"
            )

        dq = dq.reshape(fp16_w.shape)
>>>>>>> Stashed changes

        # Quantization residual
        residual = fp16_w - dq

        # Store everything
        residuals[i] = residual.flatten()

        fp16_weights[i] = fp16_w.flatten()

        quantized_weights[i] = dq.flatten()

    return residuals, fp16_weights, quantized_weights