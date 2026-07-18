import torch


class FeatureNormalizer:

    """
    Column-wise min-max normalization of the
    carrier feature matrix.
    """

    def __init__(self, eps=1e-8):

        self.eps = eps

    def normalize(self, features):

        features = features.float()

        minimum = features.min(dim=0).values

        maximum = features.max(dim=0).values

        return (features - minimum) / (
            maximum - minimum + self.eps
        )