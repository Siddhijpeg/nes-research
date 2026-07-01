import torch

from src.carrier_intelligence.selector import (
    CarrierSelector,
)


class QCAEResidualEmbedder:
    """
    Quantization-Constrained Adaptive Embedding (QCAE)

    Uses the Carrier Intelligence Engine to
    determine optimal embedding locations.
    """

    def __init__(self):

        self.selector = CarrierSelector()

    def embed_bits(
        self,
        residual_tensor,
        fp16_weight,
        quantized_weight,
        bits,
    ):

        residual = residual_tensor.clone()

        flat = residual.flatten()

        positions = self.selector.select(
            residual,
            fp16_weight,
            quantized_weight,
            len(bits),
        )

        for bit, pos in zip(bits, positions):

            value = max(
                abs(flat[pos].item()),
                1e-6,
            )

            if bit == 1:

                flat[pos] = value

            else:

                flat[pos] = -value

        return {
            "tensor": residual,
            "positions": positions,
        }
    
    def extract_bits(
        self,
        embedded_tensor,
        positions,
    ):

        flat = embedded_tensor.flatten()

        bits = []

        for pos in positions:

            bits.append(
                1 if flat[pos] >= 0 else 0
            )

        return bits