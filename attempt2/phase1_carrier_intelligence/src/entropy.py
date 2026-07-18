"""
entropy.py

Advanced entropy analysis for Carrier Intelligence.

This module computes multiple entropy measures for every trainable
tensor in a quantized Large Language Model.

Implemented Metrics
-------------------
1. Shannon Entropy
2. Rényi Entropy
3. Tsallis Entropy
4. Local Entropy
5. Residual Entropy
6. Entropy Stability
7. Normalized Entropy
8. Entropy Percentile
Project: Neural-Entropic Steganography (NES v2)
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from dataclasses import asdict

from pathlib import Path

from typing import List

import numpy as np
import pandas as pd
import torch

from model_loader import ModelLoader


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


@dataclass
class EntropyMetrics:

    tensor_id: int

    layer_name: str

    shannon_entropy: float

    renyi_entropy: float

    tsallis_entropy: float

    local_entropy_mean: float

    local_entropy_std: float

    local_entropy_min: float

    local_entropy_max: float

    residual_entropy: float

    entropy_stability: float

    normalized_entropy: float

    entropy_percentile: float


class EntropyAnalyzer:

    def __init__(

        self,

        model,

        bins: int = 256,

        alpha: float = 2.0,

        q: float = 2.0,

        window_size: int = 512

    ):

        self.model = model

        self.bins = bins

        self.alpha = alpha

        self.q = q

        self.window_size = window_size

    ####################################################################
    # Probability Distribution
    ####################################################################

    def probability_distribution(self, tensor):

        x = tensor.detach().float().cpu().flatten()

        minimum = float(torch.min(x))

        maximum = float(torch.max(x))

        if minimum == maximum:

            return np.array([1.0])

        histogram = torch.histc(

            x,

            bins=self.bins,

            min=minimum,

            max=maximum

        )

        histogram = histogram.numpy()

        probability = histogram / histogram.sum()

        probability = probability[probability > 0]

        return probability

    ####################################################################
    # Shannon Entropy
    ####################################################################

    def shannon_entropy(self, probability):

        return float(

            -np.sum(

                probability *

                np.log2(probability)

            )

        )

    ####################################################################
    # Rényi Entropy
    ####################################################################

    def renyi_entropy(self, probability):

        alpha = self.alpha

        if alpha == 1:

            return self.shannon_entropy(probability)

        return float(

            (1 / (1 - alpha))

            *

            np.log2(

                np.sum(

                    probability ** alpha

                )

            )

        )

    ####################################################################
    # Tsallis Entropy
    ####################################################################

    def tsallis_entropy(self, probability):

        q = self.q

        if q == 1:

            return self.shannon_entropy(probability)

        return float(

            (

                1

                -

                np.sum(

                    probability ** q

                )

            )

            /

            (

                q - 1

            )

        )
    
        ####################################################################
    # Sliding Window Generator
    ####################################################################

    def sliding_windows(self, tensor):

        """
        Split a flattened tensor into fixed-size windows.
        """

        x = tensor.detach().float().cpu().flatten()

        windows = []

        for start in range(0, len(x), self.window_size):

            window = x[start:start + self.window_size]

            if len(window) > 1:

                windows.append(window)

        return windows

    ####################################################################
    # Local Shannon Entropy
    ####################################################################

    def local_entropy(self, tensor):

        """
        Compute Shannon entropy for every local window.
        """

        windows = self.sliding_windows(tensor)

        entropy_values = []

        for window in windows:

            minimum = float(torch.min(window))
            maximum = float(torch.max(window))

            if minimum == maximum:

                entropy_values.append(0.0)

                continue

            histogram = torch.histc(

                window,

                bins=self.bins,

                min=minimum,

                max=maximum

            )

            probability = histogram.numpy()

            probability = probability / probability.sum()

            probability = probability[probability > 0]

            entropy = self.shannon_entropy(probability)

            entropy_values.append(entropy)

        return entropy_values

    ####################################################################
    # Local Entropy Statistics
    ####################################################################

    def local_entropy_statistics(self, tensor):

        """
        Aggregate local entropy measurements.
        """

        entropy_values = self.local_entropy(tensor)

        if len(entropy_values) == 0:

            return {

                "mean": 0.0,

                "std": 0.0,

                "min": 0.0,

                "max": 0.0

            }

        entropy_values = np.asarray(entropy_values)

        return {

            "mean": float(np.mean(entropy_values)),

            "std": float(np.std(entropy_values)),

            "min": float(np.min(entropy_values)),

            "max": float(np.max(entropy_values))

        }

    ####################################################################
    # Normalized Entropy
    ####################################################################

    def normalize_entropy(self, entropy):

        """
        Normalize entropy into [0,1].
        """

        maximum_entropy = np.log2(self.bins)

        if maximum_entropy == 0:

            return 0.0

        return entropy / maximum_entropy
    
        ####################################################################
    # Residual Entropy
    ####################################################################

    def residual_entropy(self, tensor):

        """
        Computes entropy of first-order residuals.
        """

        x = tensor.detach().float().cpu().flatten()

        if len(x) < 2:
            return 0.0

        residual = x[1:] - x[:-1]

        probability = self.probability_distribution(residual)

        return self.shannon_entropy(probability)

    ####################################################################
    # Entropy Stability
    ####################################################################

    def entropy_stability(self, tensor):

        """
        Measures robustness of entropy under a small perturbation.
        """

        x = tensor.detach().float().cpu()

        probability_original = self.probability_distribution(x)

        entropy_original = self.shannon_entropy(
            probability_original
        )

        noise = torch.randn_like(x) * 1e-6

        probability_noisy = self.probability_distribution(
            x + noise
        )

        entropy_noisy = self.shannon_entropy(
            probability_noisy
        )

        return abs(
            entropy_original -
            entropy_noisy
        )

    ####################################################################
    # Complete Analysis
    ####################################################################

    def analyze(self):

        logging.info("Computing advanced entropy metrics...")

        results = []

        for idx, (name, param) in enumerate(
                self.model.named_parameters()):

            probability = self.probability_distribution(
                param
            )

            shannon = self.shannon_entropy(
                probability
            )

            renyi = self.renyi_entropy(
                probability
            )

            tsallis = self.tsallis_entropy(
                probability
            )

            local = self.local_entropy_statistics(
                param
            )

            residual = self.residual_entropy(
                param
            )

            stability = self.entropy_stability(
                param
            )

            results.append(

                EntropyMetrics(

                    tensor_id=idx,

                    layer_name=name,

                    shannon_entropy=shannon,

                    renyi_entropy=renyi,

                    tsallis_entropy=tsallis,

                    local_entropy_mean=local["mean"],

                    local_entropy_std=local["std"],

                    local_entropy_min=local["min"],

                    local_entropy_max=local["max"],

                    residual_entropy=residual,

                    entropy_stability=stability,

                    normalized_entropy=self.normalize_entropy(
                        shannon
                    ),

                    entropy_percentile=0.0

                )

            )

        entropy_values = np.array([
            r.shannon_entropy
            for r in results
        ])

        for r in results:

            r.entropy_percentile = float(

                (
                    entropy_values <=
                    r.shannon_entropy
                ).mean()

            )

        logging.info(
            "Computed entropy metrics for %d tensors.",
            len(results)
        )

        return results

    ####################################################################
    # Export
    ####################################################################

    @staticmethod
    def export(
        results,
        output_dir="../outputs"
    ):

        output = Path(output_dir)

        output.mkdir(
            parents=True,
            exist_ok=True
        )

        dataframe = pd.DataFrame(

            [asdict(r) for r in results]

        )

        dataframe.to_csv(

            output / "entropy_scores.csv",

            index=False

        )

        dataframe.to_json(

            output / "entropy_scores.json",

            orient="records",

            indent=4

        )

        logging.info(
            "Entropy results exported successfully."
        )


####################################################################
# Main
####################################################################

if __name__ == "__main__":

    loader = ModelLoader()

    _, model = loader.load()

    analyzer = EntropyAnalyzer(model)

    results = analyzer.analyze()

    analyzer.export(results)

    print()

    print("=" * 80)
    print("Advanced Entropy Analysis Complete")
    print("=" * 80)
    print()

    for layer in results[:5]:

        print(layer)