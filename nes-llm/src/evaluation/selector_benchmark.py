import json
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from bitsandbytes.functional import (
    dequantize_4bit,
)

from src.embedding.embedder import (
    ResidualEmbedder,
)

from src.embedding.strategies.sign_strategy import (
    SignStrategy,
)

from src.carrier_selection.selectors.random_selector import (
    RandomSelector,
)

from src.carrier_selection.selectors.magnitude_selector import (
    MagnitudeSelector,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build_residual(fp16_layer, nf4_layer):

    fp16_weight = (
        fp16_layer.weight.data.float().cpu()
    )

    nf4_weight = (
        dequantize_4bit(
            nf4_layer.weight.data,
            quant_state=nf4_layer.weight.quant_state,
        )
        .float()
        .cpu()
    )

    return fp16_weight - nf4_weight


def bit_error_rate(a, b):

    errors = sum(
        x != y
        for x, y in zip(a, b)
    )

    return errors / len(a)


def evaluate_selector(
    selector,
    residual,
    payload,
):

    bits = torch.randint(
        0,
        2,
        (payload,),
    ).tolist()

    embedder = ResidualEmbedder(
        strategy=SignStrategy(),
        selector=selector,
    )

    stego = embedder.embed(
        residual_tensor=residual.clone(),
        bits=bits,
        secret_key="nes_secret",
    )

    recovered = embedder.extract(
        embedded_tensor=stego,
        num_bits=payload,
        secret_key="nes_secret",
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

    fp16_layer = fp16.model.layers[0].mlp.down_proj
    nf4_layer = nf4.model.layers[0].mlp.down_proj

    residual = build_residual(
        fp16_layer,
        nf4_layer,
    )

    payload = 100000

    selectors = {

        "random": RandomSelector(),

        "magnitude": MagnitudeSelector(),

    }

    results = {}

    for name, selector in selectors.items():

        print(f"\n===== {name.upper()} =====")

        results[name] = evaluate_selector(
            selector,
            residual,
            payload,
        )

        print(results[name])

    with open(
        "selector_benchmark.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        "\nSaved selector_benchmark.json"
    )


if __name__ == "__main__":
    main()