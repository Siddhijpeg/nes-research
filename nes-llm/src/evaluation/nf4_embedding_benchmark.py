import json
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from bitsandbytes.functional import (
    dequantize_4bit,
)

from src.embedding.payload_encoder import PayloadEncoder

from src.embedding.residual_embedder_v2 import (
    ResidualEmbedder,
)

from src.embedding.strategies.sign_strategy import (
    SignStrategy,
)

from src.embedding.strategies.nf4_quantization_strategy import (
    NF4QuantizationStrategy,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build_residual():

    print("Loading FP16...")

    fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
    ).cpu()

    print("Loading NF4...")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    nf4 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map="cpu",
    )

    fp16_layer = (
        fp16.model.layers[0]
        .self_attn.q_proj
    )

    nf4_layer = (
        nf4.model.layers[0]
        .self_attn.q_proj
    )

    fp16_weight = (
        fp16_layer.weight.float().cpu()
    )

    nf4_weight = (
        dequantize_4bit(
            nf4_layer.weight.data,
            quant_state=nf4_layer.weight.quant_state,
        )
        .float()
        .cpu()
    )

    return (
        fp16_weight - nf4_weight,
        nf4_weight,
    )


def bit_error_rate(
    target,
    recovered,
):

    errors = sum(
        a != b
        for a, b in zip(
            target,
            recovered,
        )
    )

    return errors / len(target)


def evaluate_strategy(
    strategy,
    name,
    residual,
    reference,
    bits,
):

    embedder = ResidualEmbedder(
        strategy=strategy,
    )

    embedded = embedder.embed_bits(
        residual,
        reference,
        bits,
    )

    recovered = embedder.extract_bits(
        embedded,
        reference,
        len(bits),
    )

    ber = bit_error_rate(
        bits,
        recovered,
    )

    return {

        "strategy": name,

        "ber": ber,

        "accuracy": 1 - ber,

        "payload_bits": len(bits),

        "capacity": residual.numel(),

        "utilization": (
            len(bits)
            / residual.numel()
        ),

    }


def main():

    residual, reference = build_residual()

    message = (
        "NF4 Quantization Aware "
        "Embedding Benchmark"
    )

    bits = PayloadEncoder.text_to_bits(
        message
    )

    results = []

    print("\nRunning Sign Strategy...")

    results.append(

        evaluate_strategy(

            SignStrategy(),

            "Sign",

            residual,

            reference,

            bits,

        )

    )

    print("\nRunning NF4 Strategy...")

    results.append(

        evaluate_strategy(

            NF4QuantizationStrategy(),

            "NF4-QAE",

            residual,

            reference,

            bits,

        )

    )

    print()

    for r in results:

        print(r)

    with open(

        "nf4_embedding_benchmark.json",

        "w",

    ) as f:

        json.dump(

            results,

            f,

            indent=4,

        )

    print(
        "\nSaved nf4_embedding_benchmark.json"
    )


if __name__ == "__main__":

    main()