import json
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from bitsandbytes.functional import (
    quantize_4bit,
    dequantize_4bit,
)

from src.embedding.residual_embedder_v2 import (
    ResidualEmbedder,
)

from src.embedding.strategies.sign_strategy import (
    SignStrategy,
)

from src.embedding.strategies.nf4_quantization_strategy import (
    NF4QuantizationStrategy,
)

from src.embedding.reference_builder import (
    ReferenceBuilder,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build_residual():

    print("Loading FP16...")

    fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
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
        fp16_layer.weight.data
        .float()
        .cpu()
    )

    nf4_weight = (
        dequantize_4bit(
            nf4_layer.weight.data,
            quant_state=nf4_layer.weight.quant_state,
        )
        .float()
        .cpu()
    )

    residual = fp16_weight - nf4_weight

    return residual, nf4_weight


def bit_error_rate(
    original,
    recovered,
):

    errors = sum(
        a != b
        for a, b in zip(
            original,
            recovered,
        )
    )

    return errors / len(original)


def evaluate_strategy(
    strategy,
    strategy_name,
    residual,
    reference,
    payload_sizes,
):

    embedder = ResidualEmbedder(
        strategy=strategy,
    )

    results = {}

    for payload in payload_sizes:

        print(
            f"\n[{strategy_name}] Payload = {payload}"
        )

        bits = torch.randint(
            0,
            2,
            (payload,),
        ).tolist()

        embedded = embedder.embed_bits(
            residual,
            reference,
            bits,
        )

        print("Real NF4 requantization...")

        qweight, qstate = quantize_4bit(
            embedded,
            quant_type="nf4",
        )

        requantized = dequantize_4bit(
            qweight,
            quant_state=qstate,
        )

        recovered = embedder.extract_bits(
            requantized,
            reference,
            payload,
        )

        ber = bit_error_rate(
            bits,
            recovered,
        )

        results[payload] = {
            "ber": ber,
            "accuracy": 1 - ber,
        }

        print(results[payload])

    return results


def main():

    residual, nf4_weight = build_residual()

    reference = ReferenceBuilder.build_from_residual(
        residual,
        nf4_weight,
    )

    payload_sizes = [
        1000,
        10000,
        50000,
        100000,
    ]

    sign_results = evaluate_strategy(
        SignStrategy(),
        "Sign",
        residual,
        reference,
        payload_sizes,
    )

    nf4_results = evaluate_strategy(
        NF4QuantizationStrategy(),
        "NF4-QAE",
        residual,
        reference,
        payload_sizes,
    )

    final_results = {
        "Sign": sign_results,
        "NF4-QAE": nf4_results,
    }

    with open(
        "nf4_requantization_results_v2.json",
        "w",
    ) as f:

        json.dump(
            final_results,
            f,
            indent=4,
        )

    print(
        "\nSaved nf4_requantization_results_v2.json"
    )


if __name__ == "__main__":
    main()