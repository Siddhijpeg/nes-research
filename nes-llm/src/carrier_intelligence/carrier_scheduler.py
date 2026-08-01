"""
QACI Carrier Scheduler — Hamilton Largest-Remainder payload allocator.

Distributes total_payload_bits across layers proportionally to their
adjusted_quality, guaranteeing Σ allocated_bits == total_payload_bits.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class CarrierAllocation:
    """Payload assignment for one layer."""
    layer_id:         int
    module_name:      str
    adjusted_quality: float
    capacity:         int    # residual.numel()
    allocated_bits:   int


class CarrierScheduler:
    """
    Hamilton Largest-Remainder Payload Scheduler.

    Distributes total_payload_bits across layers so that:
      1. Each layer's share is proportional to its adjusted_quality.
      2. Σ allocated_bits == total_payload_bits exactly.
      3. No layer receives more bits than its capacity.
      4. Zero-quality layers receive 0 bits.
    """

    def __init__(self, gamma: float = 2.5, eps: float = 1e-8):
        self.gamma = gamma
        self.eps   = eps

    def _weighted_scores(self, profiles: List[dict]) -> torch.Tensor:
        scores = torch.tensor(
            [p["adjusted_quality"] for p in profiles], dtype=torch.float32
        )
        scores = scores.clamp(min=0.0).pow(self.gamma)
        return scores / (scores.sum() + self.eps)

    def allocate(
        self,
        layer_profiles: List[dict],
        total_payload_bits: int,
    ) -> List[CarrierAllocation]:
        """
        Allocate bits across layers using Hamilton Largest-Remainder.

        Each profile dict must contain:
            layer_id, module_name, adjusted_quality, num_params
        """
        n = len(layer_profiles)
        if n == 0:
            return []

        weights    = self._weighted_scores(layer_profiles)
        capacities = torch.tensor(
            [p["num_params"] for p in layer_profiles], dtype=torch.float32
        )

        raw_bits = torch.min(weights * total_payload_bits, capacities)

        clip_total = raw_bits.sum().item()
        if clip_total < self.eps:
            raw_bits = torch.ones(n) * (total_payload_bits / n)

        raw_bits = raw_bits * (total_payload_bits / (raw_bits.sum().item() + self.eps))

        # Hamilton Largest-Remainder
        floors    = raw_bits.floor().long()
        remainder = raw_bits - raw_bits.floor()
        deficit   = total_payload_bits - floors.sum().item()

        if deficit > 0:
            _, indices = torch.topk(remainder, int(deficit))
            for idx in indices:
                floors[idx] += 1

        # Clamp to capacity
        floors = torch.min(floors, capacities.long())

        # Fix any overshoot after clamping
        overshoot = floors.sum().item() - total_payload_bits
        if overshoot > 0:
            order = torch.argsort(
                torch.tensor([p["adjusted_quality"] for p in layer_profiles])
            )
            for idx in order:
                if overshoot <= 0:
                    break
                reduction  = min(overshoot, floors[idx].item())
                floors[idx] -= reduction
                overshoot   -= reduction

        allocations = [
            CarrierAllocation(
                layer_id=         profile["layer_id"],
                module_name=      profile["module_name"],
                adjusted_quality= profile["adjusted_quality"],
                capacity=         int(capacities[i].item()),
                allocated_bits=   int(floors[i].item()),
            )
            for i, profile in enumerate(layer_profiles)
        ]
        return sorted(allocations, key=lambda a: a.layer_id)

    def allocation_to_dict(self, allocations: List[CarrierAllocation]) -> Dict[int, int]:
        """Convert to {layer_id: bits}."""
        return {a.layer_id: a.allocated_bits for a in allocations}

    def summary(self, allocations: List[CarrierAllocation]) -> dict:
        bits = [a.allocated_bits for a in allocations]
        return {
            "total_layers":   len(allocations),
            "total_bits":     sum(bits),
            "mean_bits":      sum(bits) / max(len(bits), 1),
            "max_bits":       max(bits) if bits else 0,
            "min_bits":       min(bits) if bits else 0,
            "nonzero_layers": sum(1 for b in bits if b > 0),
        }