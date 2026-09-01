import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ==============================================================
# NF4 CONFIGURATION
# ==============================================================

NF4_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


# ==============================================================
# DEVICE
# ==============================================================

def get_device():
    """
    Prefer Apple Silicon MPS when available.

    The model itself runs on MPS.
    NF4 dequantization is temporarily performed on CPU because
    BitsAndBytes dequantization is not fully supported on MPS.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ==============================================================
# MODEL LOADING
# ==============================================================

def load_model_pair(model_id: str, device=None):

    if device is None:
        device = get_device()

    print(f"Loading model: {model_id}")
    print(f"Device: {device}")

    # ----------------------------------------------------------
    # Tokenizer
    # ----------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    # ----------------------------------------------------------
    # NF4 model
    # ----------------------------------------------------------

    nf4_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=NF4_CONFIG,
        device_map={"": str(device)},
        trust_remote_code=True,
    )

    # ----------------------------------------------------------
    # FP16 reference model
    # ----------------------------------------------------------

    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map={"": str(device)},
        trust_remote_code=True,
    )

    return nf4_model, fp16_model, tokenizer


# ==============================================================
# RESIDUAL EXTRACTION + CACHING
# ==============================================================

def extract_residuals(
    nf4_model,
    fp16_model,
    family: str,
    model_id: str,
    cache_root: str = "cache/models",
    force_recompute: bool = False,
):
    """
    Extract FP16-vs-NF4 residuals for every transformer layer.

    Residual:

        R = W_FP16 - W_NF4_dequantized

    Three tensors are maintained for each layer:

        residuals[i]
        fp16_weights[i]
        quantized_weights[i]

    Expensive NF4 dequantization is cached to disk so later
    experiments can reuse the precomputed tensors.

    Important:
        - Model loading happens on MPS.
        - NF4 dequantization happens on CPU.
        - The resulting tensor is moved back to MPS.
        - Cache files are stored on CPU.
    """

    from src.model.registry import (
        get_layer_module,
        get_num_layers,
    )

    from src.model.cache_manager import (
        ModelTensorCache,
    )

    # ----------------------------------------------------------
    # Number of transformer layers
    # ----------------------------------------------------------

    n = get_num_layers(nf4_model)

    # ----------------------------------------------------------
    # Initialize cache manager
    # ----------------------------------------------------------

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

    # ==========================================================
    # PROCESS EACH LAYER
    # ==========================================================

    for i in range(n):

        # ------------------------------------------------------
        # CACHE HIT
        # ------------------------------------------------------

        if (
            not force_recompute
            and cache.validate_layer(i)
        ):

            print(
                f"Layer {i:02d}: "
                f"loading from cache"
            )

            # Cache is stored on CPU.
            # Move tensors to the FP16 model's device for runtime.
            runtime_device = next(
                fp16_model.parameters()
            ).device

            residual, fp16_w, dq = cache.load_layer(
                i,
                device=runtime_device,
            )

            residuals[i] = residual
            fp16_weights[i] = fp16_w
            quantized_weights[i] = dq

            continue

        # ------------------------------------------------------
        # CACHE MISS
        # ------------------------------------------------------

        print(
            f"Layer {i:02d}: "
            f"computing residual"
        )

        # ------------------------------------------------------
        # Locate corresponding MLP modules
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # FP16 reference weight
        # ------------------------------------------------------

        fp16_w = (
            fp16_mlp
            .down_proj
            .weight
            .detach()
            .float()
        )

        # ------------------------------------------------------
        # NF4 quantized parameter
        # ------------------------------------------------------

        nf4_w = nf4_mlp.down_proj.weight

        # ======================================================
        # NF4 DEQUANTIZATION
        # ======================================================

        """
        Params4bit contains compressed NF4 values.

        On MPS:

            NF4 Params4bit
                    |
                    v
                  CPU
                    |
                    v
              dequantize()
                    |
                    v
             FP32 tensor
                    |
                    v
                  MPS
        """

        nf4_cpu = nf4_w.detach().to("cpu")

        dq_cpu = (
            nf4_cpu
            .dequantize()
            .float()
        )

        # ------------------------------------------------------
        # Shape validation
        # ------------------------------------------------------

        if dq_cpu.numel() != fp16_w.numel():

            raise RuntimeError(
                f"Layer {i}: shape mismatch after "
                f"NF4 dequantization: "
                f"FP16={fp16_w.shape} "
                f"({fp16_w.numel()} elements), "
                f"NF4={dq_cpu.shape} "
                f"({dq_cpu.numel()} elements)"
            )

        # ------------------------------------------------------
        # Restore original matrix shape
        # ------------------------------------------------------

        dq_cpu = dq_cpu.reshape(
            fp16_w.shape
        )

        # ------------------------------------------------------
        # Move dequantized NF4 weight to model device
        # ------------------------------------------------------

        dq = dq_cpu.to(
            fp16_w.device
        )

        # ======================================================
        # QUANTIZATION RESIDUAL
        # ======================================================

        residual = fp16_w - dq

        # ------------------------------------------------------
        # Runtime representation
        # ------------------------------------------------------

        residuals[i] = residual.flatten()

        fp16_weights[i] = fp16_w.flatten()

        quantized_weights[i] = dq.flatten()

        # ======================================================
        # SAVE CACHE
        # ======================================================

        """
        Save CPU copies.

        This is important because the cache should not depend
        on MPS tensors or GPU/accelerator memory.
        """

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

    # ==========================================================
    # COMPLETE
    # ==========================================================

    print("\nCache preprocessing complete.")

    return (
        residuals,
        fp16_weights,
        quantized_weights,
    )