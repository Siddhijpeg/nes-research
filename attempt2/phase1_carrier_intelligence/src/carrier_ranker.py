"""
carrier_ranker.py

Ranks carrier tensors using entropy, importance, and statistical
features computed during Phase 1.

Author: Mehar Kapoor
Project: Neural-Entropic Steganography (NES v2)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


class CarrierRanker:

    def __init__(
        self,
        entropy_file="../outputs/entropy_scores.csv",
        importance_file="../outputs/importance_scores.csv",
        statistics_file="../outputs/layer_statistics.csv",
    ):

        self.entropy_file = Path(entropy_file)
        self.importance_file = Path(importance_file)
        self.statistics_file = Path(statistics_file)

    @staticmethod
    def normalize(series):

        denominator = series.max() - series.min()

        if denominator == 0:
            return series

        return (series - series.min()) / denominator

    def rank(self):

        logging.info("Loading analysis files...")

        entropy = pd.read_csv(self.entropy_file)

        importance = pd.read_csv(self.importance_file)

        statistics = pd.read_csv(self.statistics_file)

        dataframe = entropy.merge(
            importance,
            on=["tensor_id", "layer_name"]
        )

        dataframe = dataframe.merge(
            statistics,
            on=["tensor_id", "layer_name"]
        )

        dataframe["entropy_score"] = self.normalize(
            dataframe["normalized_entropy"]
        )

        dataframe["importance_score_norm"] = self.normalize(
            dataframe["importance_score"]
        )

        dataframe["dynamic_range_norm"] = self.normalize(
            dataframe["dynamic_range"]
        )

        dataframe["final_score"] = (

            dataframe["entropy_score"]

            +

            dataframe["importance_score_norm"]

            +

            dataframe["dynamic_range_norm"]

        ) / 3.0

        dataframe = dataframe.sort_values(
            by="final_score",
            ascending=False
        )

        dataframe["rank"] = range(
            1,
            len(dataframe) + 1
        )

        return dataframe

    @staticmethod
    def export(
        dataframe,
        output_dir="../outputs"
    ):

        output = Path(output_dir)

        output.mkdir(
            parents=True,
            exist_ok=True
        )

        dataframe.to_csv(

            output / "carrier_ranking.csv",

            index=False

        )

        logging.info(
            "Carrier ranking exported successfully."
        )


if __name__ == "__main__":

    ranker = CarrierRanker()

    ranking = ranker.rank()

    ranker.export(ranking)

    print()

    print("=" * 80)
    print("Top 10 Carrier Candidates")
    print("=" * 80)

    print()

    print(

        ranking[
            [
                "rank",
                "layer_name",
                "final_score"
            ]
        ].head(10)

    )