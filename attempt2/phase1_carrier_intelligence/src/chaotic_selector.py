"""
chaotic_selector.py

Generates carrier indices using a Logistic Chaotic Map.
Project: Neural-Entropic Steganography (NES v2)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
from model_loader import ModelLoader
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


@dataclass
class ChaoticCarrier:

    sequence_id: int

    chaotic_value: float

    normalized_index: int


class ChaoticSelector:

    def __init__(
        self,
        model,
        sequence_length: int = 5000,
        r: float = 3.99,
        x0: float = 0.61803398875,
    ):

        self.model = model
        self.sequence_length = sequence_length
        self.r = r
        self.x = x0

    ####################################################################
    # Logistic Map
    ####################################################################

    def logistic_map(self):

        self.x = self.r * self.x * (1 - self.x)

        return self.x

    ####################################################################
    # Generate Chaotic Sequence
    ####################################################################

    def select(self):

        logging.info("Generating chaotic carrier sequence...")

        total_parameters = sum(
            p.numel()
            for p in self.model.parameters()
        )

        carriers = []

        for i in range(self.sequence_length):

            value = self.logistic_map()

            index = int(value * total_parameters)

            carriers.append(

                ChaoticCarrier(

                    sequence_id=i,

                    chaotic_value=value,

                    normalized_index=index

                )

            )

        logging.info(
            "Generated %d chaotic carrier candidates.",
            len(carriers)
        )

        return carriers

    ####################################################################
    # Export
    ####################################################################

    @staticmethod
    def export(
        carriers,
        output_dir="../outputs"
    ):

        output_path = Path(output_dir)

        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        dataframe = pd.DataFrame(

            [asdict(c) for c in carriers]

        )

        dataframe.to_csv(

            output_path / "chaotic_carriers.csv",

            index=False

        )

        with open(

            output_path / "chaotic_carriers.json",

            "w"

        ) as f:

            json.dump(

                [asdict(c) for c in carriers],

                f,

                indent=4

            )

        logging.info(
            "Chaotic carrier sequence exported successfully."
        )


if __name__ == "__main__":

    loader = ModelLoader()

    _, model = loader.load()

    selector = ChaoticSelector(model)

    carriers = selector.select()

    selector.export(carriers)

    print()

    print("=" * 80)
    print("Chaotic Carrier Generation Complete")
    print("=" * 80)

    print()

    for carrier in carriers[:10]:
        print(carrier)