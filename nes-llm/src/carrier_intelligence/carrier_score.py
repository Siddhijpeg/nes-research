import torch


class CarrierScore:
    """
    Computes the overall carrier score used
    for payload allocation.

    Score

        S = wQ·Q + wC·C + wM·M
    """

    def __init__(

        self,

        quality_weight=0.50,

        confidence_weight=0.30,

        margin_weight=0.20,

        eps=1e-8,

    ):

        self.wq = quality_weight

        self.wc = confidence_weight

        self.wm = margin_weight

        self.eps = eps

    ##########################################################

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

    ##########################################################

    def compute(

        self,

        quality,

        confidence,

        margin,

    ):

        quality = self.normalize(
            quality
        )

        confidence = self.normalize(
            confidence
        )

        margin = self.normalize(
            margin
        )

        score = (

            self.wq * quality

            +

            self.wc * confidence

            +

            self.wm * margin

        )

        return score

    ##########################################################

    def rank(

        self,

        score,

    ):

        return torch.argsort(

            score,

            descending=True,

        )

    ##########################################################

    def top_k(

        self,

        score,

        k,

    ):

        ranking = self.rank(
            score
        )

        return ranking[:k]

    ##########################################################

    def statistics(

        self,

        score,

    ):

        return {

            "mean":
            score.mean().item(),

            "std":
            score.std().item(),

            "min":
            score.min().item(),

            "max":
            score.max().item(),

        }


def main():

    quality = torch.rand(50000)

    confidence = torch.rand(50000)

    margin = torch.rand(50000)

    scorer = CarrierScore()

    score = scorer.compute(

        quality,

        confidence,

        margin,

    )

    print()

    print(

        scorer.statistics(

            score

        )

    )


if __name__ == "__main__":
    main()