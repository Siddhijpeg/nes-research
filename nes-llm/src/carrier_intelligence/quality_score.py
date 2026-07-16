import torch


class QualityScore:
    """
    Computes an overall carrier quality score
    from the complete carrier feature matrix.

    Feature Matrix Columns

        0 Magnitude
        1 Variance
        2 Std

        3 Quantization Error
        4 Stability
        5 Robustness

        6 Cost
        7 Z-score
        8 Distortion
    """

    MAGNITUDE = 0
    VARIANCE = 1
    STD = 2

    QUANTIZATION_ERROR = 3
    STABILITY = 4
    ROBUSTNESS = 5

    COST = 6
    ZSCORE = 7
    DISTORTION = 8

    def __init__(
        self,
        eps=1e-8,
    ):

        self.eps = eps

    def normalize(
        self,
        x,
    ):

        return (

            x - x.min()

        ) / (

            x.max()

            - x.min()

            + self.eps

        )

    ##############################################################
    # Composite Feature Scores
    ##############################################################

    def information_score(
        self,
        features,
    ):

        magnitude = features[:, self.MAGNITUDE]

        variance = features[:, self.VARIANCE]

        std = features[:, self.STD]

        return (

            0.50 * magnitude

            +

            0.25 * variance

            +

            0.25 * std

        )

    def robustness_score(
        self,
        features,
    ):

        stability = features[:, self.STABILITY]

        robustness = features[:, self.ROBUSTNESS]

        quantization = 1.0 - features[:, self.QUANTIZATION_ERROR]

        return (

            0.40 * robustness

            +

            0.40 * stability

            +

            0.20 * quantization

        )

    def distortion_score(
        self,
        features,
    ):

        distortion = features[:, self.DISTORTION]

        cost = features[:, self.COST]

        zscore = features[:, self.ZSCORE].abs()

        return (

            0.50 * distortion

            +

            0.30 * cost

            +

            0.20 * zscore

        )

    ##############################################################
    # Overall Quality
    ##############################################################

    def compute(
        self,
        feature_matrix,
    ):

        information = self.normalize(

            self.information_score(
                feature_matrix
            )

        )

        robustness = self.normalize(

            self.robustness_score(
                feature_matrix
            )

        )

        distortion = self.normalize(

            self.distortion_score(
                feature_matrix
            )

        )

        quality = (

            robustness

            * information

        ) / (

            1.0

            + distortion

        )

        return quality

    def statistics(
        self,
        quality,
    ):

        return {

            "mean":
            quality.mean().item(),

            "std":
            quality.std().item(),

            "min":
            quality.min().item(),

            "max":
            quality.max().item(),

        }


def main():

    features = torch.rand(
        10000,
        9,
    )

    scorer = QualityScore()

    quality = scorer.compute(
        features
    )

    print()

    print(
        scorer.statistics(
            quality
        )
    )


if __name__ == "__main__":
    main()