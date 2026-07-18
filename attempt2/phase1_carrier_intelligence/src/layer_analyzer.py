"""
layer_analyzer.py

Generates a statistical fingerprint for every trainable tensor
in the model.
Project: Neural-Entropic Steganography (NES v2)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pandas as pd
import torch

from model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


@dataclass
class LayerStatistics:

    tensor_id: int

    layer_name: str

    shape: tuple

    mean: float

    std: float

    minimum: float

    maximum: float

    dynamic_range: float

    l1_norm: float

    l2_norm: float

    zero_ratio: float

    non_zero_ratio: float

    histogram: list


class LayerAnalyzer:

    def __init__(self,
                 model,
                 histogram_bins: int = 100):

        self.model = model
        self.histogram_bins = histogram_bins

    def analyze(self) -> List[LayerStatistics]:

        logging.info("Analyzing model tensors...")

        results = []

        for idx, (name, param) in enumerate(
                self.model.named_parameters()):

            tensor = param.detach().float().cpu()

            flat = tensor.flatten()

            histogram = torch.histc(
                flat,
                bins=self.histogram_bins,
                min=float(flat.min()),
                max=float(flat.max())
            ).tolist()

            stats = LayerStatistics(

                tensor_id=idx,

                layer_name=name,

                shape=tuple(tensor.shape),

                mean=float(flat.mean()),

                std=float(flat.std()),

                minimum=float(flat.min()),

                maximum=float(flat.max()),

                dynamic_range=float(
                    flat.max() - flat.min()
                ),

                l1_norm=float(
                    torch.norm(flat, p=1)
                ),

                l2_norm=float(
                    torch.norm(flat, p=2)
                ),

                zero_ratio=float(
                    (flat == 0).float().mean()
                ),

                non_zero_ratio=float(
                    (flat != 0).float().mean()
                ),

                histogram=histogram

            )

            results.append(stats)

        logging.info(
            "Finished analyzing %d tensors.",
            len(results)
        )

        return results

    @staticmethod
    def export(results: List[LayerStatistics],
               output_dir="../outputs"):

        output = Path(output_dir)

        output.mkdir(
            parents=True,
            exist_ok=True
        )

        dataframe = pd.DataFrame(

            [asdict(r) for r in results]

        )

        dataframe.to_csv(

            output / "layer_statistics.csv",

            index=False

        )

        with open(

                output / "layer_statistics.json",

                "w"

        ) as f:

            json.dump(

                [asdict(r) for r in results],

                f,

                indent=4

            )

        logging.info(
            "Statistics exported successfully."
        )


if __name__ == "__main__":

    loader = ModelLoader()

    _, model = loader.load()

    analyzer = LayerAnalyzer(model)

    results = analyzer.analyze()

    analyzer.export(results)

    print()

    print("=" * 80)

    print("Layer Statistics Generated")

    print("=" * 80)

    print()

    for layer in results[:5]:

        print(layer)