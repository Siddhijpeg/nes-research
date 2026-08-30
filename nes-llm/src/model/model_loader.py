"""
Model loader for NES experiments.

Supports:
    - NF4 model loading
    - FP16 reference model loading
    - Residual extraction
    - Applying embedded residuals

Designed for Apple Silicon (MPS).
"""

from typing import Optional, Tuple
import os

# Enable CPU fallback for PyTorch operations that are not supported by MPS.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


# ============================================================
# DEVICE
# Time Complexity: O(1)
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# NF4 CONFIGURATION
# Time Complexity: O(1)
# ============================================================

NF4_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


# ============================================================
# MODEL LOADER CLASS
# ============================================================

class ModelLoader:
    """
    Loads a HuggingFace causal LM in NF4 quantization.

    Supports models such as:
        - meta-llama/Llama-3.1-8B
        - mistralai/Mistral-7B-v0.3
        - Qwen/Qwen2.5-7B
        - TinyLlama/TinyLlama-1.1B-Chat-v1.0
    """

    def __init__(
        self,
        device: str = "mps",
        use_double_quant: bool = True,
        compute_dtype: str = "float16",
    ):
        self.device = device
        self.use_double_quant = use_double_quant
        self.compute_dtype = compute_dtype

    def load(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Tuple:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=self.use_double_quant,
            bnb_4bit_compute_dtype=getattr(
                torch,
                self.compute_dtype
            ),
        )

        print(
            f"[ModelLoader] Loading {model_id} in NF4..."
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": self.device},
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=True,
        )

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(
            f"[ModelLoader] Loaded. "
            f"Layers: {model.config.num_hidden_layers}"
        )

        return model, tokenizer

    def get_layer_names(self, model) -> list:
        """
        Return all quantized linear layer names.

        Time Complexity: O(N)
        where N = number of modules in the model.
        """

        names = []

        for name, module in model.named_modules():

            if (
                hasattr(module, "weight")
                and hasattr(module.weight, "quant_type")
            ):
                names.append(name)

        return names


# ============================================================
# LOAD NF4 + FP16 MODEL PAIR
# ============================================================

def load_model_pair(
    model_id: str,
    device=None,
    token: Optional[str] = None,
):
    """
    Load NF4 and FP16 copies of the same model.

    NF4 model:
        Used as the carrier model.

    FP16 model:
        Used as the reference model for residual calculation.
    """

    if device is None:
        device = DEVICE

    if isinstance(device, torch.device):
        device = str(device)

    print(f"\n[load_model_pair] Model: {model_id}")
    print(f"[load_model_pair] Device: {device}")

    # --------------------------------------------------------
    # Tokenizer
    # Time Complexity: O(model tokenizer size)
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # NF4 MODEL
    # Time Complexity: O(model size)
    # --------------------------------------------------------

    print("[load_model_pair] Loading NF4 model...")

    nf4_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=NF4_CONFIG,
        device_map={"": device},
        trust_remote_code=True,
        token=token,
    )

    nf4_model.eval()

    # --------------------------------------------------------
    # FP16 MODEL
    # Time Complexity: O(model size)
    # --------------------------------------------------------

    print("[load_model_pair] Loading FP16 model...")

    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
        token=token,
    )

    fp16_model.eval()

    print(
        f"[load_model_pair] "
        f"NF4 + FP16 loaded successfully."
    )

    return nf4_model, fp16_model, tokenizer


# ============================================================
# EXTRACT RESIDUALS
# ============================================================

def extract_residuals(
    nf4_model,
    fp16_model,
    family: str
) -> dict:
    """
    Calculate:

        residual = FP16 weight - dequantized NF4 weight

    for every layer's down_proj.

    Returns:
        {layer_index: flattened residual tensor}
    """

    from src.model.registry import (
        get_layer_module,
        get_num_layers
    )

    n = get_num_layers(nf4_model)

    residuals = {}

    # --------------------------------------------------------
    # Iterate through transformer layers
    # Time Complexity: O(L * W)
    # L = number of layers
    # W = number of weights per down_proj
    # --------------------------------------------------------

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

        # Move FP16 weight to the same device
        # Time Complexity: O(W)
        fp16_w = fp16_w.to(nf4_w.device)

        # Dequantize NF4 weight
        # Time Complexity: O(W)
        if hasattr(nf4_w, "dequantize"):
            dq = nf4_w.dequantize().float()
        else:
            dq = nf4_w.float()

        # Check number of elements before reshaping
        if dq.numel() != fp16_w.numel():
            raise RuntimeError(
                f"Weight size mismatch at layer {i}: "
                f"NF4={dq.numel()}, FP16={fp16_w.numel()}"
            )

        # Restore original weight shape
        # Time Complexity: O(W)
        dq = dq.reshape(fp16_w.shape)

        # Calculate residual
        # Time Complexity: O(W)
        residuals[i] = (
            fp16_w.float() - dq
        ).flatten()

    return residuals


# ============================================================
# APPLY EMBEDDED RESIDUALS
# ============================================================

def apply_residuals_to_model(
    nf4_model,
    fp16_model,
    embedded_residuals: dict,
    family: str
):
    """
    Apply the modified residuals to the NF4 model.

    Formula:

        modified_weight =
            original_dequantized_NF4_weight
            + embedded_residual

    The resulting weight is stored as a regular floating-point
    parameter so that the modified model can be evaluated.

    Note:
        After modification, the affected down_proj weights are
        no longer represented as BitsAndBytes 4-bit parameters.
    """

    from src.model.registry import (
        get_layer_module,
        get_num_layers
    )

    n = get_num_layers(nf4_model)

    # --------------------------------------------------------
    # Iterate through all layers
    # Time Complexity: O(L * W)
    # --------------------------------------------------------

    for i in range(n):

        if i not in embedded_residuals:
            continue

        nf4_mlp = get_layer_module(
            nf4_model,
            family,
            i,
            "mlp"
        )

        nf4_w = nf4_mlp.down_proj.weight

        # ----------------------------------------------------
        # Dequantize original NF4 weight
        # Time Complexity: O(W)
        # ----------------------------------------------------

        if hasattr(nf4_w, "dequantize"):
            original_weight = (
                nf4_w.dequantize().float()
            )
        else:
            original_weight = nf4_w.float()

        # ----------------------------------------------------
        # Reshape residual to original weight shape
        # Time Complexity: O(W)
        # ----------------------------------------------------

        residual = embedded_residuals[i]

        residual = residual.reshape(
            original_weight.shape
        )

        residual = residual.to(
            original_weight.device
        ).float()

        # ----------------------------------------------------
        # Apply residual
        # Time Complexity: O(W)
        # ----------------------------------------------------

        modified_weight = (
            original_weight + residual
        )

        # ----------------------------------------------------
        # Replace parameter
        # Time Complexity: O(W)
        # ----------------------------------------------------

        nf4_mlp.down_proj.weight = torch.nn.Parameter(
            modified_weight.to(torch.float16),
            requires_grad=False
        )

    nf4_model.eval()

    return nf4_model