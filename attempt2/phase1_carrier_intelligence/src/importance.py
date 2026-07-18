"""
importance.py

Computes layer-wise importance scores for every trainable tensor
in the quantized LLM.
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
class ImportanceMetrics:

    tensor_id: int

    layer_name: str

    mean_absolute_weight: float

    weight_variance: float

    l2_norm: float

    max_absolute_weight: float

    sparsity: float

    importance_score: float


class ImportanceAnalyzer:

    def __init__(self, model):

        self.model = model

    def compute_metrics(self, tensor):

        x = tensor.detach().float().cpu().flatten()

        abs_mean = torch.mean(torch.abs(x)).item()

        variance = torch.var(x).item()

        l2_norm = torch.norm(x, p=2).item()

        max_abs = torch.max(torch.abs(x)).item()

        sparsity = (x == 0).float().mean().item()

        importance_score = (
            abs_mean *
            variance *
            l2_norm *
            (1 - sparsity)
        )

        return (
            abs_mean,
            variance,
            l2_norm,
            max_abs,
            sparsity,
            importance_score
        )

    def analyze(self) -> List[ImportanceMetrics]:

        logging.info("Computing importance metrics...")

        results = []

        for idx, (name, param) in enumerate(
                self.model.named_parameters()):

            (
                abs_mean,
                variance,
                l2_norm,
                max_abs,
                sparsity,
                score
            ) = self.compute_metrics(param)

            results.append(

                ImportanceMetrics(

                    tensor_id=idx,

                    layer_name=name,

                    mean_absolute_weight=abs_mean,

                    weight_variance=variance,

                    l2_norm=l2_norm,

                    max_absolute_weight=max_abs,

                    sparsity=sparsity,

                    importance_score=score

                )

            )

        logging.info(
            "Importance analysis completed for %d tensors.",
            len(results)
        )

        return results

    @staticmethod
    def export(results,
               output_dir="../outputs"):

        output_path = Path(output_dir)

        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        dataframe = pd.DataFrame(

            [asdict(r) for r in results]

        )

        dataframe.to_csv(

            output_path / "importance_scores.csv",

            index=False

        )

        with open(

                output_path / "importance_scores.json",

                "w"

        ) as f:

            json.dump(

                [asdict(r) for r in results],

                f,

                indent=4

            )

        logging.info(
            "Importance metrics exported successfully."
        )


if __name__ == "__main__":

    loader = ModelLoader()

    _, model = loader.load()

    analyzer = ImportanceAnalyzer(model)

    results = analyzer.analyze()

    analyzer.export(results)

    print()

    print("=" * 80)
    print("Importance Analysis Complete")
    print("=" * 80)
    print()

    for layer in results[:5]:
        print(layer)