import torch

from src.carrier_selection.carrier_features import (
    CarrierFeatures,
)


class AdaptiveScorer:
    """
    Adaptive Carrier Scoring (ACS).

    Computes a carrier quality score from one or
    more residual characteristics.

    The current version supports only magnitude.

    Later versions progressively include:

        • Entropy
        • Local variance
        • Kurtosis
        • Layer priors
        • Quantization stability
    """

    def __init__(
        self,
        magnitude_weight=1.0,
        entropy_weight=0.0,
        variance_weight=0.0,
        kurtosis_weight=0.0,
        layer_weight=0.0,
    ):

        self.magnitude_weight = magnitude_weight
        self.entropy_weight = entropy_weight
        self.variance_weight = variance_weight
        self.kurtosis_weight = kurtosis_weight
        self.layer_weight = layer_weight

    @staticmethod
    def normalize(scores):

        scores = scores.float()

        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            return torch.zeros_like(scores)

        return (scores - minimum) / (
            maximum - minimum
        )

    def score(
        self,
        residual_tensor,
    ):
        """
        Compute the Adaptive Carrier Score.
        """

        score = torch.zeros_like(
            residual_tensor,
            dtype=torch.float32,
        )

        # ----------------------------
        # Magnitude
        # ----------------------------

        if self.magnitude_weight > 0:

            magnitude = self.normalize(
                CarrierFeatures.magnitude(
                    residual_tensor
                )
            )

            score += (
                self.magnitude_weight
                * magnitude
            )

        # ----------------------------
        # Entropy (future)
        # ----------------------------

        if self.entropy_weight > 0:

            entropy = self.normalize(
                CarrierFeatures.entropy(
                    residual_tensor
                )
            )

            score += (
                self.entropy_weight
                * entropy
            )

        # ----------------------------
        # Variance (future)
        # ----------------------------

        if self.variance_weight > 0:

            variance = self.normalize(
                CarrierFeatures.variance(
                    residual_tensor
                )
            )

            score += (
                self.variance_weight
                * variance
            )

        # ----------------------------
        # Kurtosis (future)
        # ----------------------------

        if self.kurtosis_weight > 0:

            kurtosis = self.normalize(
                CarrierFeatures.kurtosis(
                    residual_tensor
                )
            )

            score += (
                self.kurtosis_weight
                * kurtosis
            )

        # ----------------------------
        # Layer Prior (future)
        # ----------------------------

        if self.layer_weight > 0:

            layer = self.normalize(
                CarrierFeatures.layer_prior(
                    residual_tensor
                )
            )

            score += (
                self.layer_weight
                * layer
            )

        return score