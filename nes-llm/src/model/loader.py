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
    # MPS is preferred on Apple Silicon Macs.
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

    # Load NF4 quantized model.
    nf4_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=NF4_CONFIG,
        device_map={"": str(device)},
        trust_remote_code=True
    )

    # Load FP16 reference model.
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

        # ---------------------------------------------------------
        # FP16 reference weight
        # ---------------------------------------------------------
        fp16_w = (
            fp16_mlp.down_proj.weight
            .detach()
            .float()
        )

        # ---------------------------------------------------------
        # NF4 quantized weight
        # ---------------------------------------------------------
        nf4_w = nf4_mlp.down_proj.weight

        # ---------------------------------------------------------
        # DIAGNOSTIC INFORMATION
        #
        # We are checking the actual BitsAndBytes representation
        # before attempting to construct the residual.
        #
        # Only print detailed information for the first layer
        # to avoid flooding the terminal.
        # ---------------------------------------------------------
        if i == 0:

            print("\n" + "=" * 60)
            print("NF4 DEQUANTIZATION DIAGNOSTICS")
            print("=" * 60)

            print("FP16 shape:")
            print(fp16_w.shape)

            print("\nNF4 tensor shape:")
            print(nf4_w.shape)

            print("\nNF4 tensor type:")
            print(type(nf4_w))

            print("\nNF4 tensor dtype:")
            print(nf4_w.dtype)

            print("\nNF4 tensor device:")
            print(nf4_w.device)

            print("\nHas dequantize():")
            print(hasattr(nf4_w, "dequantize"))

            print("\nHas quant_state:")
            print(hasattr(nf4_w, "quant_state"))

            if hasattr(nf4_w, "quant_state"):
                print("\nQuantization state:")
                print(nf4_w.quant_state)

            print("=" * 60)

        # ---------------------------------------------------------
        # Try BitsAndBytes dequantization
        # ---------------------------------------------------------
        if hasattr(nf4_w, "dequantize"):

            dq = nf4_w.dequantize()

        else:

            dq = nf4_w

        dq = dq.float()

        # ---------------------------------------------------------
        # DIAGNOSTIC: inspect reconstructed tensor
        # ---------------------------------------------------------
        if i == 0:

            print("\nDequantized tensor shape:")
            print(dq.shape)

            print("Dequantized tensor dtype:")
            print(dq.dtype)

            print("Dequantized tensor device:")
            print(dq.device)

            print("FP16 number of elements:")
            print(fp16_w.numel())

            print("NF4/dequantized number of elements:")
            print(dq.numel())

            print("=" * 60)

        # ---------------------------------------------------------
        # Shape validation
        #
        # IMPORTANT:
        # Do NOT reshape here unless the number of elements matches.
        # ---------------------------------------------------------
        if dq.numel() != fp16_w.numel():

            raise RuntimeError(
                f"Layer {i}: shape mismatch after NF4 dequantization: "
                f"FP16={fp16_w.shape} ({fp16_w.numel()} elements), "
                f"NF4={dq.shape} ({dq.numel()} elements)"
            )

        # ---------------------------------------------------------
        # Restore original FP16 weight shape.
        # ---------------------------------------------------------
        dq = dq.reshape(fp16_w.shape)

        # ---------------------------------------------------------
        # Quantization residual
        #
        # r = W_FP16 - W_NF4
        # ---------------------------------------------------------
        residual = fp16_w - dq

        # ---------------------------------------------------------
        # Store flattened representations.
        # ---------------------------------------------------------
        residuals[i] = residual.flatten()

        fp16_weights[i] = fp16_w.flatten()

        quantized_weights[i] = dq.flatten()

    return residuals, fp16_weights, quantized_weights