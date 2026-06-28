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

from src.embedding.reference_builder import (
    ReferenceBuilder,
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

    fp16_layer = fp16.model.layers[0].mlp.down_proj
    nf4_layer = nf4.model.layers[0].mlp.down_proj

    fp16_weight = (
        fp16_layer.weight.detach()
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

    return residual, nf4_weight


def bit_error_rate(a, b):

    errors = sum(
        x != y
        for x, y in zip(a, b)
    )

    return errors / len(a)


def nf4_cycle(tensor):

    qweight, qstate = quantize_4bit(
        tensor,
        quant_type="nf4",
    )

    return dequantize_4bit(
        qweight,
        quant_state=qstate,
    )


def evaluate_strategy(
    strategy,
    name,
    residual,
    reference,
    cycles,
):

    payload = 100000

    bits = torch.randint(
        0,
        2,
        (payload,),
    ).tolist()

    embedder = ResidualEmbedder(
        strategy=strategy,
    )

    embedded = embedder.embed_bits(
        residual,
        reference,
        bits,
    )

    results = {}

    current = embedded.clone()

    for cycle in cycles:

        print(
            f"{name} :: Cycle {cycle}"
        )

        for _ in range(cycle):
            current = nf4_cycle(current)

        recovered = embedder.extract_bits(
            current,
            reference,
            payload,
        )

        ber = bit_error_rate(
            bits,
            recovered,
        )

        results[f"{cycle}_cycles"]= {
            "ber": ber,
            "accuracy": 1 - ber,
        }

        current = embedded.clone()

    return results


def main():

    residual, nf4_weight = build_residual()

    reference = (
        ReferenceBuilder
        .build_from_residual(
            residual,
            nf4_weight,
        )
    )

    cycles = [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        1000,
    ]

    sign_results = evaluate_strategy(
        SignStrategy(),
        "Sign",
        residual,
        reference,
        cycles,
    )

    nf4_results = evaluate_strategy(
        NF4QuantizationStrategy(),
        "NF4-QAE",
        residual,
        reference,
        cycles,
    )

    results = {
        "Sign": sign_results,
        "NF4-QAE": nf4_results,
    }

    with open(
        "multi_cycle_requantization_v2.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        "\nSaved multi_cycle_requantization_v2.json"
    )


if __name__ == "__main__":
    main()