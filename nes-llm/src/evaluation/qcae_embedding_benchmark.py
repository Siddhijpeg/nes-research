import json
import time
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from bitsandbytes.functional import (
    dequantize_4bit,
)

from src.embedding.payload_encoder import (
    PayloadEncoder,
)

from src.embedding.residual_embedder import (
    ResidualEmbedder,
)

from src.embedding.residual_embedder_qcae import (
    QCAEResidualEmbedder,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build_residual():

    print("Loading FP16...")

    fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="cpu",
    )

    print("Loading NF4...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    nf4 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="cpu",
    )

    fp16_layer = (
        fp16.model.layers[0]
        .mlp.down_proj
    )

    nf4_layer = (
        nf4.model.layers[0]
        .mlp.down_proj
    )

    fp16_weight = (
        fp16_layer.weight
        .detach()
        .float()
        .cpu()
    )

    nf4_weight = (
        dequantize_4bit(
            nf4_layer.weight.data,
            quant_state=nf4_layer.weight.quant_state,
        )
        .detach()
        .float()
        .cpu()
    )

    residual = fp16_weight - nf4_weight

    return (
        residual,
        fp16_weight,
        nf4_weight,
    )


def bit_error_rate(
    original,
    recovered,
):

    errors = sum(
        x != y
        for x, y in zip(
            original,
            recovered,
        )
    )

    return errors / len(original)


def evaluate_v1(
    residual,
    bits,
):

    start = time.time()

    embedded = (
        ResidualEmbedder.embed_bits(
            residual,
            bits,
        )
    )

    recovered = (
        ResidualEmbedder.extract_bits(
            embedded,
            len(bits),
        )
    )

    runtime = (
        time.time()
        - start
    )

    return {

        "ber":
        bit_error_rate(
            bits,
            recovered,
        ),

        "accuracy":
        1
        -
        bit_error_rate(
            bits,
            recovered,
        ),

        "runtime":
        runtime,

        "capacity":
        residual.numel(),

        "payload":
        len(bits),

        "utilization":
        len(bits)
        /
        residual.numel(),
    }


def evaluate_qcae(
    residual,
    fp16_weight,
    nf4_weight,
    bits,
):

    embedder = (
        QCAEResidualEmbedder()
    )

    start = time.time()

    embedded = (
        embedder.embed_bits(
            residual,
            fp16_weight,
            nf4_weight,
            bits,
        )
    )

    recovered = (
        embedder.extract_bits(
            embedded,
            fp16_weight,
            nf4_weight,
            len(bits),
        )
    )

    runtime = (
        time.time()
        - start
    )

    return {

        "ber":
        bit_error_rate(
            bits,
            recovered,
        ),

        "accuracy":
        1
        -
        bit_error_rate(
            bits,
            recovered,
        ),

        "runtime":
        runtime,

        "capacity":
        residual.numel(),

        "payload":
        len(bits),

        "utilization":
        len(bits)
        /
        residual.numel(),
    }


def main():

    (
        residual,
        fp16_weight,
        nf4_weight,
    ) = build_residual()

    message = (
        "Quantization Constrained Adaptive Embedding "
        "for Neural Network Steganography"
    )

    bits = (
        PayloadEncoder.text_to_bits(
            message
        )
    )

    print()

    print(
        "Running V1..."
    )

    v1 = evaluate_v1(
        residual,
        bits,
    )

    print(
        "Running QCAE..."
    )

    qcae = evaluate_qcae(
        residual,
        fp16_weight,
        nf4_weight,
        bits,
    )

    results = {

        "V1":
        v1,

        "QCAE":
        qcae,

    }

    print()

    print(
        json.dumps(
            results,
            indent=4,
        )
    )

    with open(
        "qcae_embedding_benchmark.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print()

    print(
        "Saved qcae_embedding_benchmark.json"
    )


if __name__ == "__main__":
    main()