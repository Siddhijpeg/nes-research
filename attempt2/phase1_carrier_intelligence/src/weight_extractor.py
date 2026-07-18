"""
weight_extractor.py

Extracts and catalogs metadata for every trainable tensor in a
quantized Large Language Model.

The generated metadata serves as the foundation for all subsequent
Carrier Intelligence analyses including statistical profiling,
entropy estimation, importance computation, and carrier ranking.
Project: Neural-Entropic Steganography (NES v2)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import pandas as pd

from model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


@dataclass
class LayerMetadata:
    """Stores metadata for a single trainable tensor."""

    tensor_id: int
    layer_name: str
    layer_type: str

    shape: tuple

    dtype: str

    num_parameters: int

    parameter_percentage: float

    requires_grad: bool

    quantized: bool


class WeightExtractor:
    """
    Extracts metadata for every trainable tensor in the model.
    """

    def __init__(self, model):

        self.model = model

        self.total_parameters = sum(
            p.numel()
            for p in model.parameters()
        )

    @staticmethod
    def identify_layer_type(layer_name: str) -> str:

        name = layer_name.lower()

        mapping = {

            "embed": "Embedding",

            "self_attn": "Attention",

            "mlp": "MLP",

            "norm": "Normalization",

            "lm_head": "LM Head"

        }

        for key, value in mapping.items():

            if key in name:

                return value

        return "Other"

    @staticmethod
    def is_quantized(dtype: str) -> bool:

        keywords = [

            "int8",

            "int4",

            "uint8",

            "nf4",

            "fp4"

        ]

        return any(k in dtype.lower() for k in keywords)

    def extract(self) -> List[LayerMetadata]:

        logging.info("Extracting layer metadata...")

        metadata = []

        for idx, (name, param) in enumerate(
                self.model.named_parameters()):

            num_parameters = param.numel()

            metadata.append(

                LayerMetadata(

                    tensor_id=idx,

                    layer_name=name,

                    layer_type=self.identify_layer_type(name),

                    shape=tuple(param.shape),

                    dtype=str(param.dtype),

                    num_parameters=num_parameters,

                    parameter_percentage=(
                        num_parameters /
                        self.total_parameters
                    ) * 100,

                    requires_grad=param.requires_grad,

                    quantized=self.is_quantized(
                        str(param.dtype)
                    )

                )

            )

        logging.info(
            "Successfully extracted metadata for %d tensors.",
            len(metadata)
        )

        return metadata

    @staticmethod
    def export(metadata: List[LayerMetadata],
               output_dir: str = "../outputs") -> None:

        output_path = Path(output_dir)

        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        dataframe = pd.DataFrame(

            [asdict(layer) for layer in metadata]

        )

        dataframe.to_csv(

            output_path / "layer_metadata.csv",

            index=False

        )

        with open(

                output_path / "layer_metadata.json",

                "w"

        ) as f:

            json.dump(

                [asdict(layer) for layer in metadata],

                f,

                indent=4

            )

        logging.info(
            "Metadata exported to %s",
            output_path.resolve()
        )


if __name__ == "__main__":

    loader = ModelLoader()

    _, model = loader.load()

    extractor = WeightExtractor(model)

    metadata = extractor.extract()

    extractor.export(metadata)

    print()

    print("=" * 80)

    print(f"Total Tensors : {len(metadata)}")

    print("=" * 80)

    print()

    for layer in metadata[:5]:

        print(layer)