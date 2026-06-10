import torch

from src.embedding.payload_encoder import PayloadEncoder
from src.embedding.keyed_residual_embedder import (
    KeyedResidualEmbedder
)

message = "HELLO NES"

bits = PayloadEncoder.text_to_bits(
    message
)

residual = torch.randn(
    100000
)

embedded = (
    KeyedResidualEmbedder.embed_bits(
        residual,
        bits,
        "mehar123"
    )
)

correct_bits = (
    KeyedResidualEmbedder.extract_bits(
        embedded,
        "mehar123",
        len(bits)
    )
)

wrong_bits = (
    KeyedResidualEmbedder.extract_bits(
        embedded,
        "wrongkey",
        len(bits)
    )
)

print("\nCorrect Key:")

print(
    PayloadEncoder.bits_to_text(
        correct_bits
    )
)

print("\nWrong Key:")

print(
    PayloadEncoder.bits_to_text(
        wrong_bits
    )
)