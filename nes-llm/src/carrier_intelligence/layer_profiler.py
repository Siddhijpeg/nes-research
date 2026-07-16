import torch


class LayerProfiler:
    """
    Computes layer-level statistics from
    carrier quality scores.

    Each transformer layer receives a
    single profile describing its overall
    embedding suitability.
    """

    def __init__(self):

        pass

    def profile(
        self,
        quality_scores: torch.Tensor,
    ):

        quality = quality_scores.float()

        return {

            "mean_quality":
            quality.mean().item(),

            "std_quality":
            quality.std().item(),

            "max_quality":
            quality.max().item(),

            "min_quality":
            quality.min().item(),

            "median_quality":
            quality.median().item(),

            "top10_mean":
            torch.topk(
                quality.flatten(),
                max(
                    1,
                    int(
                        0.10 *
                        quality.numel()
                    )
                ),
            ).values.mean().item(),

            "carrier_count":
            quality.numel(),

        }