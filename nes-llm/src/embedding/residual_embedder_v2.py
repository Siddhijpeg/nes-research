from src.embedding.strategies.sign_strategy import (
    SignStrategy,
)

from src.embedding.strategies.nf4_quantization_strategy import (
    NF4QuantizationStrategy,
)


class ResidualEmbedder:
    """
    Generic residual embedder supporting
    multiple embedding strategies.
    """

    def __init__(
        self,
        strategy=None,
    ):

        if strategy is None:
            strategy = SignStrategy()

        self.strategy = strategy

    def embed_bits(
        self,
        residual_tensor,
        reference_tensor,
        bits,
        positions=None,
    ):

        if positions is None:
            positions = list(range(len(bits)))

        if len(bits) > residual_tensor.numel():
            raise ValueError(
                "Payload exceeds carrier capacity."
            )

        #
        # V2 Strategy
        #
        if isinstance(
            self.strategy,
            NF4QuantizationStrategy,
        ):

            return self.strategy.embed(
                residual_tensor,
                reference_tensor,
                bits,
                positions,
            )

        #
        # V1 Strategy
        #
        return self.strategy.embed(
            residual_tensor,
            bits,
            positions,
        )

    def extract_bits(
        self,
        embedded_tensor,
        reference_tensor,
        num_bits,
        positions=None,
    ):

        if positions is None:
            positions = list(range(num_bits))

        if isinstance(
            self.strategy,
            NF4QuantizationStrategy,
        ):

            return self.strategy.extract(
                embedded_tensor,
                reference_tensor,
                positions,
            )

        return self.strategy.extract(
            embedded_tensor,
            positions,
        )