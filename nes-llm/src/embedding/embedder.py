from src.embedding.strategies.strategy_base import (
    EmbeddingStrategy,
)

from src.carrier_selection.selector_base import (
    CarrierSelector,
)


class ResidualEmbedder:
    """
    Main embedding pipeline.

    The embedder itself contains no embedding logic.
    It simply orchestrates:

        Residual
            ↓
        Carrier Selector
            ↓
        Embedding Strategy
            ↓
        Embedded Residual

    This makes every embedding algorithm interchangeable.
    """

    def __init__(
        self,
        strategy: EmbeddingStrategy,
        selector: CarrierSelector,
    ):

        self.strategy = strategy
        self.selector = selector

    def embed(
        self,
        residual_tensor,
        bits,
        secret_key,
    ):

        positions = self.selector.select(
            residual_tensor=residual_tensor,
            num_bits=len(bits),
            secret_key=secret_key,
        )

        embedded = self.strategy.embed(
            residual_tensor=residual_tensor,
            bits=bits,
            positions=positions,
        )

        return embedded

    def extract(
        self,
        embedded_tensor,
        num_bits,
        secret_key,
    ):

        positions = self.selector.select(
            residual_tensor=embedded_tensor,
            num_bits=num_bits,
            secret_key=secret_key,
        )

        bits = self.strategy.extract(
            embedded_tensor=embedded_tensor,
            positions=positions,
        )

        return bits