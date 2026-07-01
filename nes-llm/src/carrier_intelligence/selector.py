import torch

from src.carrier_intelligence.feature_extractor import (
    CarrierFeatureExtractor,
)

from src.carrier_intelligence.objective import (
    QCAEObjective,
)


class CarrierSelector:
    """
    QCAE Carrier Intelligence Engine.

    Pipeline

        Extract Features

                ↓

        Compute Objective

                ↓

        Rank

                ↓

        Select Top-K
    """

    def __init__(self):

        self.extractor = (
            CarrierFeatureExtractor()
        )

        self.objective = (
            QCAEObjective()
        )

    def select(
        self,
        residual,
        fp16_weight,
        quantized_weight,
        payload_bits,
    ):

        feature_matrix = (
            self.extractor.extract(
                residual,
                fp16_weight,
                quantized_weight,
            )
        )

        scores = (
            self.objective.score(
                feature_matrix
            )
        )

        _, indices = torch.topk(
            scores,
            payload_bits,
            largest=True,
        )

        return indices.cpu().tolist()