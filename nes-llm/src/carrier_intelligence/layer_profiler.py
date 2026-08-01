"""
Layer profiler for QACI (Quality-Aware Carrier Intelligence).

Analyzes per-layer residual tensors to produce a quality profile used
by the CarrierScheduler to allocate payload bits across layers.
"""

import math
import torch


class LayerProfiler:
    """
    Computes a quality profile for a single transformer layer.

    For each layer it computes:
      - magnitude statistics (mean, std, max)
      - Shannon entropy of the residual distribution
      - a raw quality_score in [0, 1]
      - a position_bias factor (middle layers favoured)
      - adjusted_quality = quality_score * position_bias

    The adjusted_quality is what CarrierScheduler uses for allocation.
    """

    def __init__(self, num_bins: int = 64, eps: float = 1e-8):
        self.num_bins = num_bins
        self.eps = eps

    def _entropy(self, tensor: torch.Tensor) -> float:
        """Shannon entropy of flattened tensor, normalised to [0,1]."""
        flat = tensor.flatten().float()
        mn, mx = flat.min().item(), flat.max().item()
        if abs(mx - mn) < self.eps:
            return 0.0
        counts = torch.histc(flat, bins=self.num_bins, min=mn, max=mx)
        probs = counts / (counts.sum() + self.eps)
        log_probs = torch.log(probs + self.eps)
        entropy = -(probs * log_probs).sum().item()
        max_entropy = math.log(self.num_bins)
        return min(entropy / (max_entropy + self.eps), 1.0)

    def _position_bias(self, layer_id: int, total_layers: int) -> float:
        """
        Returns a bias factor based on layer position.
          - layers 0 – 3   : 0.65  (very early, sensitive)
          - layers 4 – 7   : 0.80  (early)
          - layers 8 – 23  : 1.00  (middle — full capacity)
          - layers 24 – 27 : 0.80  (late)
          - layers 28 – 31 : 0.65  (very late)
        """
        quarter = total_layers // 4
        if layer_id < quarter // 2:
            return 0.65
        elif layer_id < quarter:
            return 0.80
        elif layer_id < total_layers - quarter:
            return 1.00
        elif layer_id < total_layers - quarter // 2:
            return 0.80
        else:
            return 0.65

    def profile(
        self,
        residual: torch.Tensor,
        layer_id: int = 0,
        module_name: str = "unknown",
        total_layers: int = 32,
    ) -> dict:
        """
        Profile a single layer's residual tensor.

        Returns dict with keys:
          layer_id, module_name, num_params,
          mag_mean, mag_std, mag_max, entropy,
          quality_score, position_bias, adjusted_quality
        """
        flat = residual.float().flatten()
        abs_flat = flat.abs()

        mag_mean = abs_flat.mean().item()
        mag_std  = abs_flat.std().item()
        mag_max  = abs_flat.max().item()
        entropy  = self._entropy(flat)

        # 60% magnitude (soft-sigmoid normalised) + 40% entropy
        mag_norm = float(torch.sigmoid(torch.tensor(mag_mean * 20.0)).item())
        quality_score = max(0.0, min(1.0, 0.60 * mag_norm + 0.40 * entropy))

        position_bias    = self._position_bias(layer_id, total_layers)
        adjusted_quality = quality_score * position_bias

        return {
            "layer_id":         layer_id,
            "module_name":      module_name,
            "num_params":       flat.numel(),
            "mag_mean":         mag_mean,
            "mag_std":          mag_std,
            "mag_max":          mag_max,
            "entropy":          entropy,
            "quality_score":    quality_score,
            "position_bias":    position_bias,
            "adjusted_quality": adjusted_quality,
        }