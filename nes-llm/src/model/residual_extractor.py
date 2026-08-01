"""
Residual extractor — computes FP16 - NF4 residuals from real model weights.

For each target layer:
    residual = dequantize(nf4_weight) - fp16_weight_reference

These residuals are what we embed bits into.
"""

from typing import Dict, List, Optional, Tuple
import torch


# Target modules for Llama-style models (most capacity, least sensitive)
LLAMA_TARGET_MODULES = [
    "down_proj",
    "gate_proj",
    "up_proj",
]

CONSERVATIVE_TARGET_MODULES = [
    "down_proj",
]


class ResidualExtractor:
    """
    Extracts per-layer quantization residuals from a BitsAndBytes NF4 model.

    For each layer × module:
        1. Dequantize the NF4 weight  → fp16_dequant
        2. residual = fp16_dequant    (NF4 models don't store original FP16)

    In pure NF4 mode the "residual" IS the dequantized weight — sign embedding
    modifies the dequantized weight values which survive requantization because
    we select high-magnitude positions.

    Usage:
        extractor = ResidualExtractor(target_modules=LLAMA_TARGET_MODULES)
        residuals, fp16_weights = extractor.extract(model)
        # residuals:    {layer_id: tensor}
        # fp16_weights: {layer_id: tensor}  (same as dequantized)
    """

    def __init__(
        self,
        target_modules: List[str] = None,
        layer_range:    Optional[Tuple[int, int]] = None,
    ):
        self.target_modules = target_modules or LLAMA_TARGET_MODULES
        self.layer_range    = layer_range   # e.g. (8, 24) for middle layers only

    def extract(
        self,
        model,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, str]]:
        """
        Extract residuals from all target layers.

        Returns:
            residuals:    {layer_id: dequantized_weight_tensor}
            fp16_weights: {layer_id: same tensor (alias)}
            module_names: {layer_id: module_name_string}
        """
        residuals    = {}
        fp16_weights = {}
        module_names = {}
        layer_id     = 0

        num_layers = model.config.num_hidden_layers
        lo, hi     = self.layer_range if self.layer_range else (0, num_layers)

        for layer_idx in range(num_layers):
            if not (lo <= layer_idx < hi):
                continue

            layer = model.model.layers[layer_idx]

            for mod_name in self.target_modules:
                # Navigate to the module (mlp.down_proj etc.)
                module = self._get_submodule(layer, f"mlp.{mod_name}")
                if module is None:
                    continue

                try:
                    dequant = self._dequantize(module)
                    residuals[layer_id]    = dequant
                    fp16_weights[layer_id] = dequant
                    module_names[layer_id] = f"layer{layer_idx}.{mod_name}"
                    layer_id += 1
                except Exception as e:
                    print(f"[ResidualExtractor] Skipping layer{layer_idx}.{mod_name}: {e}")

        print(f"[ResidualExtractor] Extracted {layer_id} residual tensors "
              f"({sum(t.numel() for t in residuals.values()):,} total params)")
        return residuals, fp16_weights, module_names

    def _get_submodule(self, parent, path: str):
        """Traverse dotted path from parent module."""
        parts = path.split(".")
        mod   = parent
        for part in parts:
            mod = getattr(mod, part, None)
            if mod is None:
                return None
        return mod

    def _dequantize(self, module) -> torch.Tensor:
        """
        Dequantize a BitsAndBytes NF4 linear layer weight to float32.
        """
        import bitsandbytes as bnb

        if isinstance(module, bnb.nn.Linear4bit):
            # BitsAndBytes provides dequantize_4bit
            weight = bnb.functional.dequantize_4bit(
                module.weight.data,
                module.weight.quant_state,
            ).to(torch.float32)
            return weight
        elif hasattr(module, "weight"):
            return module.weight.data.float()
        else:
            raise ValueError(f"Cannot dequantize module of type {type(module)}")