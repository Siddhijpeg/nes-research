import torch


class AdaptiveMarginController:
    """
    Adaptive Margin Controller (AMC)

    Converts carrier quality scores into
    adaptive embedding margins.

    Higher quality carriers receive larger
    margins, improving robustness while
    low-quality carriers receive smaller
    perturbations to reduce distortion.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        eps: float = 1e-8,
    ):

        self.alpha = alpha
        self.eps = eps

    def normalize(
        self,
        quality_scores: torch.Tensor,
    ) -> torch.Tensor:

        minimum = quality_scores.min()

        maximum = quality_scores.max()

        return (
            quality_scores - minimum
        ) / (
            maximum - minimum + self.eps
        )

    def compute(
        self,
        residual: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> torch.Tensor:

        quality = self.normalize(
            quality_scores
        )

        global_scale = (
            residual.std()
            * self.alpha
        )

        margins = (
            global_scale
            * quality
        )

        return margins

    def statistics(
        self,
        margins: torch.Tensor,
    ):

        return {

            "mean":
            margins.mean().item(),

            "std":
            margins.std().item(),

            "min":
            margins.min().item(),

            "max":
            margins.max().item(),

        }


def main():

    residual = torch.randn(
        1024
    )

    quality = torch.rand(
        1024
    )

    controller = (
        AdaptiveMarginController()
    )

    margins = controller.compute(
        residual,
        quality,
    )

    print()

    print(
        controller.statistics(
            margins
        )
    )


if __name__ == "__main__":
    main()