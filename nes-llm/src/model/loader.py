import copy

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

    The models run on MPS.

    NF4 dequantization is performed on CPU because the
    BitsAndBytes 4-bit dequantization path is not reliable
    for MPS tensors.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ==============================================================
# MODEL LOADING
# ==============================================================

def load_model_pair(model_id: str, device=None):
    """
    Load:

        1. NF4 quantized model
        2. FP16 reference model
        3. Tokenizer

    The models themselves are placed on the requested runtime
    device, normally MPS on Apple Silicon.
    """

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
# QUANTIZATION STATE → CPU
# ==============================================================

def _quant_state_to_cpu(qstate):
    """
    Create a CPU-safe copy of a BitsAndBytes QuantState.

    Moving the packed NF4 parameter to CPU does NOT necessarily
    move every tensor contained inside its QuantState.

    This matters especially when double quantization is enabled.

    Tensor attributes commonly used by BitsAndBytes include:

        - absmax
        - code
        - offset
        - quant_map
        - nested_absmax
        - nested_quant_map

    Different BitsAndBytes versions may represent nested
    quantization state differently, so both `nested` and
    `state2` are handled.
    """

    if qstate is None:
        return None

    # ----------------------------------------------------------
    # Make a shallow copy so the original model's QuantState
    # is never modified.
    # ----------------------------------------------------------

    state_cpu = copy.copy(qstate)

    # ----------------------------------------------------------
    # Move tensor attributes to CPU.
    # ----------------------------------------------------------

    tensor_attributes = (
        "absmax",
        "code",
        "offset",
        "quant_map",
        "nested_absmax",
        "nested_quant_map",
    )

    for attr in tensor_attributes:

        value = getattr(
            state_cpu,
            attr,
            None,
        )

        if torch.is_tensor(value):

            setattr(
                state_cpu,
                attr,
                value.detach().to("cpu"),
            )

    # ----------------------------------------------------------
    # Handle nested QuantState.
    #
    # BitsAndBytes versions can use different attribute names.
    # ----------------------------------------------------------

    nested = getattr(
        state_cpu,
        "nested",
        None,
    )

    if nested is not None:

        state_cpu.nested = _quant_state_to_cpu(
            nested
        )

    state2 = getattr(
        state_cpu,
        "state2",
        None,
    )

    if state2 is not None:

        state_cpu.state2 = _quant_state_to_cpu(
            state2
        )

    return state_cpu


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

    Three tensors are maintained for every layer:

        residuals[i]
        fp16_weights[i]
        quantized_weights[i]

    Expensive NF4 dequantization is cached to disk so later
    experiments can reuse the precomputed tensors.

    Architecture:

        Model execution
            ↓
           MPS
            ↓
        NF4 parameter
            ↓
           CPU
            ↓
        NF4 dequantization
            ↓
        FP32 NF4 weight
            ↓
        FP16 reference → CPU
            ↓
        residual = FP16 - NF4
            ↓
        CPU cache

    Important:

        - Models remain on MPS.
        - NF4 dequantization happens on CPU.
        - QuantState is also moved to CPU.
        - Residual calculation happens on CPU.
        - Cache files contain CPU tensors only.
        - We do NOT move the huge dequantized matrix back
          to MPS during preprocessing.
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

    n = get_num_layers(
        nf4_model
    )

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

    # ----------------------------------------------------------
    # Runtime dictionaries
    #
    # NOTE:
    # These currently contain the complete model's tensors.
    # The persistent cache itself remains layer-by-layer.
    # ----------------------------------------------------------

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

        # ======================================================
        # CACHE HIT
        # ======================================================

        if (
            not force_recompute
            and cache.validate_layer(i)
        ):

            print(
                f"Layer {i:02d}: "
                f"loading from cache"
            )

            # --------------------------------------------------
            # Cache is stored on CPU.
            #
            # Load to the runtime device because the current
            # experiment pipeline expects runtime tensors there.
            # --------------------------------------------------

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

        # ======================================================
        # CACHE MISS
        # ======================================================

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

        # ======================================================
        # FP16 REFERENCE WEIGHT
        # ======================================================

        # The model is on MPS, but preprocessing is performed
        # on CPU to avoid large MPS allocations.

        fp16_w_cpu = (
            fp16_mlp
            .down_proj
            .weight
            .detach()
            .float()
            .to("cpu")
        )

        # ======================================================
        # NF4 QUANTIZED PARAMETER
        # ======================================================

        nf4_w = (
            nf4_mlp
            .down_proj
            .weight
        )

        # ======================================================
        # NF4 DEQUANTIZATION
        # ======================================================

        # ------------------------------------------------------
        # Get the BitsAndBytes quantization state BEFORE moving
        # the parameter data.
        # ------------------------------------------------------

        qstate = getattr(
            nf4_w,
            "quant_state",
            None,
        )

        if qstate is None:

            raise RuntimeError(
                f"Layer {i}: NF4 parameter does not contain "
                f"a BitsAndBytes quant_state."
            )

        # ------------------------------------------------------
        # Move the entire QuantState to CPU.
        #
        # This is the critical fix for the previous:
        #
        #     Expected all tensors to be on the same device
        #
        # error.
        # ------------------------------------------------------

        qstate_cpu = _quant_state_to_cpu(
            qstate
        )

        # ------------------------------------------------------
        # Get packed NF4 data.
        # ------------------------------------------------------

        qweight = getattr(
            nf4_w,
            "data",
            nf4_w,
        )

        qweight_cpu = (
            qweight
            .detach()
            .to("cpu")
        )

        # ------------------------------------------------------
        # Functional BitsAndBytes dequantization.
        #
        # Both:
        #
        #     qweight_cpu
        #     qstate_cpu
        #
        # are now on CPU.
        # ------------------------------------------------------

        import bitsandbytes.functional as bnb_func

        dq_cpu = bnb_func.dequantize_4bit(
            qweight_cpu,
            qstate_cpu,
        ).float()

        # ======================================================
        # SHAPE VALIDATION
        # ======================================================

        expected_shape = getattr(
            qstate,
            "shape",
            None,
        )

        if (
            expected_shape is not None
            and dq_cpu.shape != expected_shape
        ):

            try:

                dq_cpu = dq_cpu.reshape(
                    expected_shape
                )

            except RuntimeError:
                # If the number of elements is checked below,
                # we will raise a more informative error there.
                pass

        # ------------------------------------------------------
        # Final element-count validation
        # ------------------------------------------------------

        if dq_cpu.numel() != fp16_w_cpu.numel():

            raise RuntimeError(
                f"Layer {i}: shape mismatch after "
                f"NF4 dequantization: "
                f"FP16={fp16_w_cpu.shape} "
                f"({fp16_w_cpu.numel()} elements), "
                f"NF4={dq_cpu.shape} "
                f"({dq_cpu.numel()} elements)"
            )

        # ------------------------------------------------------
        # Restore original FP16 matrix shape.
        # ------------------------------------------------------

        dq_cpu = dq_cpu.reshape(
            fp16_w_cpu.shape
        )

        # ======================================================
        # QUANTIZATION RESIDUAL
        # ======================================================

        # Both tensors are CPU tensors.
        #
        # R = W_FP16 - W_NF4

        residual_cpu = (
            fp16_w_cpu
            - dq_cpu
        )

        # ======================================================
        # RUNTIME REPRESENTATION
        # ======================================================

        # The persistent cache is CPU-based.
        #
        # For the current API, keep cache-miss results on CPU
        # as well. This prevents a huge per-layer MPS allocation
        # during preprocessing.

        residuals[i] = (
            residual_cpu.flatten()
        )

        fp16_weights[i] = (
            fp16_w_cpu.flatten()
        )

        quantized_weights[i] = (
            dq_cpu.flatten()
        )

        # ======================================================
        # SAVE CACHE
        # ======================================================

        cache.save_layer(
            layer_id=i,
            residual=residual_cpu.flatten(),
            fp16_weight=fp16_w_cpu.flatten(),
            nf4_dequantized=dq_cpu.flatten(),
        )

        print(
            f"Layer {i:02d}: "
            f"cached successfully"
        )

        # ------------------------------------------------------
        # Explicitly release temporary references.
        #
        # Important for large 7B/8B models on unified memory.
        # ------------------------------------------------------

        del (
            nf4_mlp,
            fp16_mlp,
            nf4_w,
            qweight,
            qweight_cpu,
            qstate,
            qstate_cpu,
            dq_cpu,
            fp16_w_cpu,
            residual_cpu,
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