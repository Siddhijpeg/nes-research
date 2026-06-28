import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from bitsandbytes.functional import (
    dequantize_4bit,
)

from src.embedding.payload_encoder import PayloadEncoder
from src.embedding.residual_embedder import ResidualEmbedder

from src.embedding.strategies.nf4_quantization_strategy import (
    NF4QuantizationStrategy,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build_residual():

    print("Loading FP16 model...")

    fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
    ).cpu()

    print("Loading NF4 model...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    nf4 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
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

    return fp16_weight - nf4_weight


def main():

    residual = build_residual()

    print("\nResidual Shape:")
    print(residual.shape)

    message = (
        "MY NAME IS NES I AM SO SO SO STUPID "
        "AND I CAN'T EVEN EMBED A MESSAGE PROPERLY"
    )

    bits = PayloadEncoder.text_to_bits(
        message
    )

    strategy = NF4QuantizationStrategy(
        alpha=0.25
    )

    embedder = ResidualEmbedder(
        strategy=strategy
    )

    embedded = embedder.embed_bits(
        residual,
        bits,
    )

    recovered_bits = embedder.extract_bits(
        embedded,
        len(bits),
    )

    recovered_message = (
        PayloadEncoder.bits_to_text(
            recovered_bits
        )
    )

    print("\nOriginal Message:")
    print(message)

    print("\nRecovered Message:")
    print(recovered_message)

    print("\nSuccess:")
    print(
        recovered_message == message
    )

    print("\nCarrier Capacity:")
    print(residual.numel())

    print("\nPayload Bits:")
    print(len(bits))

    print("\nPayload Utilization:")

    print(
        f"{100 * len(bits) / residual.numel():.8f}%"
    )


if __name__ == "__main__":
    main()