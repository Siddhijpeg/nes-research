"""
Weight patcher — writes modified residuals back into model layers.

After IntelligentEmbedder modifies residuals, this patches the
changes back into the actual model parameters so they survive
when the model is saved and reloaded.
"""

from typing import Dict, List
import torch


class WeightPatcher:
    """
    Patches embedded residual tensors back into a BitsAndBytes NF4 model.

    Process per layer:
        1. Take the embedded (modified) dequantized weight.
        2. Requantize it to NF4 using BitsAndBytes.
        3. Replace the model's stored NF4 weight + quant_state.

    Usage:
        patcher = WeightPatcher()
        patcher.patch(model, embedded_residuals, module_names)
    """

    def patch(
        self,
        model,
        embedded_residuals: Dict[int, torch.Tensor],
        module_names:       Dict[int, str],
    ) -> int:
        """
        Patch embedded residuals into model in-place.

        Args:
            model:              The loaded NF4 model.
            embedded_residuals: {layer_id: modified_weight_tensor}
            module_names:       {layer_id: "layer{i}.{module}"}

        Returns:
            Number of layers successfully patched.
        """
        import bitsandbytes as bnb

        patched = 0
        for layer_id, weight_tensor in embedded_residuals.items():
            mod_name = module_names.get(layer_id, "")
            if not mod_name:
                continue

            module = self._resolve_module(model, mod_name)
            if module is None:
                print(f"[WeightPatcher] Could not find module: {mod_name}")
                continue

            try:
                self._requantize_and_patch(module, weight_tensor)
                patched += 1
            except Exception as e:
                print(f"[WeightPatcher] Failed to patch {mod_name}: {e}")

        print(f"[WeightPatcher] Patched {patched}/{len(embedded_residuals)} layers")
        return patched

    def _resolve_module(self, model, mod_name: str):
        """
        Resolve module from name like 'layer5.down_proj'.
        """
        try:
            parts     = mod_name.split(".")   # ['layer5', 'down_proj']
            layer_idx = int(parts[0].replace("layer", ""))
            sub_name  = parts[1]
            return getattr(model.model.layers[layer_idx].mlp, sub_name, None)
        except Exception:
            return None

    def _requantize_and_patch(self, module, weight_fp32: torch.Tensor):
        """
        Requantize weight_fp32 to NF4 and replace module's stored weight.
        """
        import bitsandbytes as bnb

        device = module.weight.device
        w      = weight_fp32.to(device)

        # Quantize using BitsAndBytes
        quant_weight, quant_state = bnb.functional.quantize_4bit(
            w,
            quant_type="nf4",
            compress_statistics=True,
        )

        # Replace in-place
        module.weight.data        = quant_weight
        module.weight.quant_state = quant_state