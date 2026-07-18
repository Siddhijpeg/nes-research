"""
statistics.py

Computes higher-order statistical features for each tensor.

Project: Neural-Entropic Steganography (NES v2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


@dataclass
class StatisticsMetrics:

    tensor_id: int

    layer_name: str

    variance: float

    skewness: float

    kurtosis: float

    coefficient_of_variation: float

    dynamic_range: float


class StatisticsAnalyzer:

    def __init__(self, model):

        self.model = model

    ####################################################################
    # Coefficient of Variation
    ####################################################################

    @staticmethod
    def coefficient_of_variation(mean, std):

        if abs(mean) < 1e-12:

            return 0.0

        return std / abs(mean)

    ####################################################################
    # Analyze Single Tensor
    ####################################################################

    def analyze_tensor(self, tensor):

        x = tensor.detach().float().cpu().flatten()

        mean = torch.mean(x)

        std = torch.std(x)

        variance = torch.var(x)

        centered = x - mean

        if std > 0:

            skewness = torch.mean(
                (centered / std) ** 3
            ).item()

            kurtosis = torch.mean(
                (centered / std) ** 4
            ).item()

        else:

            skewness = 0.0

            kurtosis = 0.0

        dynamic_range = (

            torch.max(x) -

            torch.min(x)

        ).item()

        cv = self.coefficient_of_variation(

            mean.item(),

            std.item()

        )

        return {

            "variance": variance.item(),

            "skewness": skewness,

            "kurtosis": kurtosis,

            "coefficient_of_variation": cv,

            "dynamic_range": dynamic_range

        }

    ####################################################################
    # Analyze Model
    ####################################################################

    def analyze(self):

        logging.info(
            "Computing higher-order statistics..."
        )

        results = []

        for idx, (name, param) in enumerate(

                self.model.named_parameters()):

            stats = self.analyze_tensor(param)

            results.append(

                StatisticsMetrics(

                    tensor_id=idx,

                    layer_name=name,

                    variance=stats["variance"],

                    skewness=stats["skewness"],

                    kurtosis=stats["kurtosis"],

                    coefficient_of_variation=stats[
                        "coefficient_of_variation"
                    ],

                    dynamic_range=stats[
                        "dynamic_range"
                    ]

                )

            )

        logging.info(
            "Computed statistics for %d tensors.",
            len(results)
        )

        return results

    ####################################################################
    # Export
    ####################################################################

    @staticmethod
    def export(results, output_dir="../outputs"):

        output = Path(output_dir)

        output.mkdir(

            parents=True,

            exist_ok=True

        )

        dataframe = pd.DataFrame(

            [asdict(r) for r in results]

        )

        dataframe.to_csv(

            output / "statistics.csv",

            index=False

        )

        dataframe.to_json(

            output / "statistics.json",

            orient="records",

            indent=4

        )

        logging.info(
            "Statistics exported successfully."
        )


####################################################################
# Main
####################################################################

if __name__ == "__main__":

    from model_loader import ModelLoader

    loader = ModelLoader()

    _, model = loader.load()

    analyzer = StatisticsAnalyzer(model)

    results = analyzer.analyze()

    analyzer.export(results)

    print()

    print("=" * 80)

    print("Higher-Order Statistics Complete")

    print("=" * 80)

    print()

    for item in results[:5]:

        print(item)