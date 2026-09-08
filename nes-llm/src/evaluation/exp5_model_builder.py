import copy
import torch
import bitsandbytes.functional as bnb

from src.model.model_registry import get_layer_module


def _move_quant_state_to_cpu(qstate):
    """
    Make a CPU copy of BitsAndBytes QuantState.

    We copy it instead of modifying the original NF4 model's
    QuantState, so the NF4 model remains usable.

    Time: O(size of quantization metadata)
    Space: O(size of quantization metadata)
    """
    state = copy.deepcopy(qstate)

    for attr, value in vars(state).items():
        if torch.is_tensor(value):
            setattr(state, attr, value.cpu())

        elif hasattr(value, "__dict__"):
            for sub_attr, sub_value in vars(value).items():
                if torch.is_tensor(sub_value):
                    setattr(value, sub_attr, sub_value.cpu())

    return state


def _dequantize_weight_cpu(nf4_weight):
    """
    Dequantize one NF4 weight completely on CPU.

    The packed NF4 tensor itself is small, while the dequantized
    FP16/FP32 tensor has the original weight shape.

    Time: O(number of weights)
    Space: O(number of weights)
    """
    packed_weight = nf4_weight.data.detach().cpu()
    qstate = _move_quant_state_to_cpu(nf4_weight.quant_state)

    dequantized = bnb.dequantize_4bit(
        packed_weight,
        quant_state=qstate
    )

    return dequantized.float()


@torch.no_grad()
def build_embedded_eval_model(
    nf4_model,
    fp16_model,
    embedded_residuals,
    family
):
    """
    Build the Exp5 evaluation model.

    For every affected layer:

        W_embedded = W_NF4 + R_embedded

    The resulting weight is placed into the FP16 model.

    IMPORTANT:
    - NF4 model is NOT modified.
    - Shared model_loader.py is NOT modified.
    - This function is specific to Exp5 evaluation.

    Time: O(total number of weights in affected layers)
    Space: O(size of one dequantized layer + embedded weight)
    """

    num_layers = len(nf4_model.model.layers)

    print("\nBuilding Exp5 embedded evaluation model...")

    for layer_id in range(num_layers):

        if layer_id not in embedded_residuals:
            continue

        # ---------------------------------------------------------
        # Get corresponding down_proj modules
        # Time: O(1)
        # ---------------------------------------------------------
        nf4_layer = get_layer_module(
        nf4_model,
        family,
        layer_id,
        "mlp"
    )

        fp16_layer = get_layer_module(
        fp16_model,
        family,
        layer_id,
        "mlp"
    )

        nf4_weight = nf4_layer.weight
        residual = embedded_residuals[layer_id]

        # ---------------------------------------------------------
        # Dequantize NF4 weight on CPU
        # Time: O(number of weights in layer)
        # Space: O(number of weights in layer)
        # ---------------------------------------------------------
        nf4_fp32 = _dequantize_weight_cpu(nf4_weight)

        # ---------------------------------------------------------
        # Move residual to CPU and reconstruct embedded weight
        # Time: O(number of weights in layer)
        # Space: O(number of weights in layer)
        # ---------------------------------------------------------
        residual_cpu = residual.detach().cpu().float()

        if residual_cpu.numel() != nf4_fp32.numel():
            raise ValueError(
                f"Layer {layer_id}: residual size mismatch. "
                f"Residual={residual_cpu.numel()}, "
                f"NF4 dequantized={nf4_fp32.numel()}"
            )

        residual_cpu = residual_cpu.reshape(nf4_fp32.shape)

        embedded_weight_cpu = nf4_fp32 + residual_cpu

        # ---------------------------------------------------------
        # Put resulting FP16 weight into FP16 model
        # Time: O(number of weights in layer)
        # ---------------------------------------------------------
        embedded_weight = embedded_weight_cpu.to(
            device=fp16_layer.weight.device,
            dtype=fp16_layer.weight.dtype
        )

        fp16_layer.weight = torch.nn.Parameter(
            embedded_weight,
            requires_grad=False
        )

        print(f"  Layer {layer_id}: embedded")

        del nf4_fp32
        del residual_cpu
        del embedded_weight_cpu
        del embedded_weight

    print("Exp5 embedded evaluation model ready.")

    return fp16_model