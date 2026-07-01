import json
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from bitsandbytes.functional import (
    dequantize_4bit,
)

from src.embedding.residual_embedder import (
    ResidualEmbedder,
)

from src.embedding.residual_embedder_qcae import (
    QCAEResidualEmbedder,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build_weights():

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
        fp16_layer.weight.detach().float().cpu()
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

    return residual, fp16_weight, nf4_weight


def bit_error_rate(a, b):

    errors = sum(
        x != y
        for x, y in zip(a, b)
    )

    return errors / len(a)


def evaluate_v1(
    residual,
    bits,
    sigma,
):

    embedded = (
        ResidualEmbedder.embed_bits(
            residual,
            bits,
        )
    )

    noisy = embedded + (
        torch.randn_like(embedded)
        * sigma
    )

    recovered = (
        ResidualEmbedder.extract_bits(
            noisy,
            len(bits),
        )
    )

    ber = bit_error_rate(
        bits,
        recovered,
    )

    return {
        "ber": ber,
        "accuracy": 1 - ber,
    }


def evaluate_qcae(
    residual,
    fp16_weight,
    nf4_weight,
    bits,
    sigma,
):

    embedder = (
        QCAEResidualEmbedder()
    )

    embedding = embedder.embed_bits(
        residual,
        fp16_weight,
        nf4_weight,
        bits,
    )

    embedded = embedding["tensor"]

    positions = embedding["positions"]

    noisy = embedded + (
        torch.randn_like(embedded)
        * sigma
    )

    recovered = embedder.extract_bits(
        noisy,
        positions,
    )

    ber = bit_error_rate(
        bits,
        recovered,
    )

    return {
        "ber": ber,
        "accuracy": 1 - ber,
    }


def main():

    residual, fp16_weight, nf4_weight = (
        build_weights()
    )

    payload = 100000

    bits = torch.randint(
        0,
        2,
        (payload,),
    ).tolist()

    sigmas = [
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
    ]

    results = {
        "V1": {},
        "QCAE": {},
    }

    for sigma in sigmas:

        print(f"\nNoise σ = {sigma}")

        results["V1"][str(sigma)] = (
            evaluate_v1(
                residual,
                bits,
                sigma,
            )
        )

        results["QCAE"][str(sigma)] = (
            evaluate_qcae(
                residual,
                fp16_weight,
                nf4_weight,
                bits,
                sigma,
            )
        )

        print(
            "V1:",
            results["V1"][str(sigma)]
        )

        print(
            "QCAE:",
            results["QCAE"][str(sigma)]
        )

    with open(
        "qcae_noise_robustness.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        "\nSaved qcae_noise_robustness.json"
    )


if __name__ == "__main__":
    main()