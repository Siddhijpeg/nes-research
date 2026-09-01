import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from bitsandbytes.functional import dequantize_4bit

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


def extract_residuals(
    nf4_model,
    fp16_model,
    family: str,
    model_id: str,
    cache_root: str = "cache/models",
    force_recompute: bool = False,
):
    from src.model.registry import (
        get_layer_module,
        get_num_layers,
    )

    from src.model.cache_manager import (
        ModelTensorCache,
    )

    n = get_num_layers(nf4_model)

    cache = ModelTensorCache(
        model_id=model_id,
        cache_root=cache_root,
        quantization_type="nf4",
        use_double_quant=True,
        compute_dtype="float16",
    )

    residuals = {}
    fp16_weights = {}
    quantized_weights = {}

    print("\n" + "=" * 60)
    print("RESIDUAL PREPROCESSING")
    print("=" * 60)

    for i in range(n):

        # --------------------------------------------------------------
        # CACHE HIT
        # --------------------------------------------------------------

        if (
            not force_recompute
            and cache.validate_layer(i)
        ):
            print(
                f"Layer {i:02d}: "
                f"loading from cache"
            )

            residual, fp16_w, dq = cache.load_layer(
                i,
                device=next(
                    fp16_model.parameters()
                ).device,
            )

            residuals[i] = residual
            fp16_weights[i] = fp16_w
            quantized_weights[i] = dq

            continue

        # --------------------------------------------------------------
        # CACHE MISS
        # --------------------------------------------------------------

        print(
            f"Layer {i:02d}: "
            f"computing residual"
        )

        nf4_mlp = get_layer_module(
            nf4_model,
            family,
            i,
            "mlp",
        )

        fp16_mlp = get_layer_module(
            fp16_model,
            family,
            i,
            "mlp",
        )

        # --------------------------------------------------------------
        # FP16 reference
        # --------------------------------------------------------------

        fp16_w = (
            fp16_mlp
            .down_proj
            .weight
            .detach()
            .float()
        )

        # --------------------------------------------------------------
        # NF4 parameter
        # --------------------------------------------------------------

        nf4_w = nf4_mlp.down_proj.weight

        # --------------------------------------------------------------
        # IMPORTANT:
        # BitsAndBytes NF4 dequantization is not reliably supported
        # directly on Apple's MPS backend.
        #
        # Therefore:
        #
        #       MPS NF4 parameter
        #              ↓
        #           CPU
        #              ↓
        #       dequantize()
        #              ↓
        #            MPS
        #
        # Only this operation uses CPU.
        # --------------------------------------------------------------

        nf4_cpu = nf4_w.detach().to("cpu")

        dq_cpu = (
            nf4_cpu
            .dequantize()
            .float()
        )

        # --------------------------------------------------------------
        # Validate shape BEFORE moving the full tensor to MPS.
        # --------------------------------------------------------------

        if dq_cpu.numel() != fp16_w.numel():
            raise RuntimeError(
                f"Layer {i}: shape mismatch after "
                f"NF4 dequantization: "
                f"FP16={fp16_w.shape} "
                f"({fp16_w.numel()} elements), "
                f"NF4={dq_cpu.shape} "
                f"({dq_cpu.numel()} elements)"
            )

        dq_cpu = dq_cpu.reshape(
            fp16_w.shape
        )

        # --------------------------------------------------------------
        # Move dequantized NF4 tensor to same device as FP16.
        # --------------------------------------------------------------

        dq = dq_cpu.to(
            fp16_w.device
        )

        # --------------------------------------------------------------
        # Quantization residual
        # --------------------------------------------------------------

        residual = fp16_w - dq

        # --------------------------------------------------------------
        # Store runtime tensors
        # --------------------------------------------------------------

        residuals[i] = residual.flatten()

        fp16_weights[i] = fp16_w.flatten()

        quantized_weights[i] = dq.flatten()

        # --------------------------------------------------------------
        # Persist CPU copies to disk.
        # --------------------------------------------------------------

        cache.save_layer(
            layer_id=i,
            residual=residual.flatten(),
            fp16_weight=fp16_w.flatten(),
            nf4_dequantized=dq.flatten(),
        )

        print(
            f"Layer {i:02d}: "
            f"cached successfully"
        )

    print("\nCache preprocessing complete.")

    return (
        residuals,
        fp16_weights,
        quantized_weights,
    )