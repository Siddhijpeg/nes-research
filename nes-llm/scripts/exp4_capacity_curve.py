import os

# Must be set before importing torch/model code.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

from src.core.types import EmbeddingConfig
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.extraction.decrypt_pipeline import DecryptPipeline
from src.model.loader import load_model_pair, extract_residuals
from scripts.exp3_clean_ber import MODELS


# ============================================================
# CONFIGURATION
# ============================================================

PAYLOAD_SIZES = [
    # 1_000,
    # 10_000,
    # 50_000,
    # 100_000,
    # 250_000,
    # 500_000,
    # 1_000_000,
    5_000_000,
    10_000_000,
    1_000_000_000,
    # 2_000_000_000,
    # 3_000_000_000,
]

MESSAGE_BYTE = b"A"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# BER CALCULATION
# ============================================================

def calculate_ber(original: str, recovered: str) -> float:
    """
    Calculate bit error rate between original and recovered text.

    BER = number of differing bits / total number of bits.

    If lengths differ, missing/extra bytes are counted as errors.

    Time Complexity: O(N)
    Space Complexity: O(N)
    """

    original_bytes = original.encode("utf-8")
    recovered_bytes = recovered.encode("utf-8")

    max_len = max(len(original_bytes), len(recovered_bytes))

    if max_len == 0:
        return 0.0

    error_bits = 0

    for i in range(max_len):
        original_byte = original_bytes[i] if i < len(original_bytes) else 0
        recovered_byte = recovered_bytes[i] if i < len(recovered_bytes) else 0

        error_bits += (original_byte ^ recovered_byte).bit_count()

    total_bits = max_len * 8

    return error_bits / total_bits


# ============================================================
# EXPERIMENT 4
# ============================================================

print("\n" + "=" * 75)
print("EXPERIMENT 4 — CAPACITY CURVE")
print("=" * 75)

print(f"Payload sizes: {PAYLOAD_SIZES}")
print(f"Models: {len(MODELS)}")


for model_id, family, n_layers in MODELS:

    print("\n" + "=" * 75)
    print(f"MODEL: {model_id}")
    print(f"FAMILY: {family}")
    print(f"EXPECTED LAYERS: {n_layers}")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD MODEL PAIR
    # --------------------------------------------------------
    # Model itself is loaded on MPS.
    # Residual preprocessing uses the persistent CPU cache.
    #
    # Time Complexity: model-dependent
    # Space Complexity: model-dependent
    # --------------------------------------------------------

    print("\nLoading model pair...")

    nf4_model, fp16_model, _ = load_model_pair(
        model_id,
        device=DEVICE,
    )

    actual_layers = len(nf4_model.model.layers)

    print(f"Detected layers: {actual_layers}")

    if actual_layers != n_layers:
        print(
            f"WARNING: expected {n_layers} layers, "
            f"but detected {actual_layers}"
        )

    # --------------------------------------------------------
    # EXTRACT / LOAD RESIDUALS
    # --------------------------------------------------------
    # extract_residuals() now returns:
    #
    #   residuals
    #   fp16_weights
    #   quantized_weights
    #
    # Cached tensors are kept on CPU to avoid large MPS
    # allocations.
    #
    # Time Complexity: O(total carrier parameters)
    # Space Complexity: O(total carrier parameters)
    # --------------------------------------------------------

    print("\nLoading residuals...")

    residuals, fp16_weights, quantized_weights = extract_residuals(
        nf4_model=nf4_model,
        fp16_model=fp16_model,
        family=family,
        model_id=model_id,
    )

    # --------------------------------------------------------
    # RAW RESIDUAL CAPACITY
    # --------------------------------------------------------
    # Every residual value represents one potential carrier
    # position in the current sign-based embedding strategy.
    #
    # Time Complexity: O(number of layers)
    # Space Complexity: O(1)
    # --------------------------------------------------------

    total_capacity = sum(
        residual.numel()
        for residual in residuals.values()
    )

    usable_capacity = int(total_capacity * 0.90)

    print(
        f"\nRaw carrier capacity : {total_capacity:,} bits"
    )

    print(
        f"90% test capacity    : {usable_capacity:,} bits"
    )

    # --------------------------------------------------------
    # PAYLOAD EXPERIMENTS
    # --------------------------------------------------------

    results = []

    for n_bits in PAYLOAD_SIZES:

        print("\n" + "-" * 65)
        print(f"Testing payload: {n_bits:,} bits")

        # ----------------------------------------------------
        # CAPACITY CHECK
        # ----------------------------------------------------

        if n_bits > usable_capacity:
            print(
                f"{n_bits:>9,} bits — "
                "SKIPPED (exceeds 90% raw capacity)"
            )

            results.append(
                {
                    "payload_bits": n_bits,
                    "ber": None,
                    "status": "SKIPPED",
                }
            )

            continue

        # ----------------------------------------------------
        # CREATE MESSAGE
        # ----------------------------------------------------
        #
        # The payload size requested by the experiment is
        # expressed in bits. The embedding pipeline itself
        # handles the payload/header/encryption representation.
        #
        # We therefore create a plaintext whose size is close
        # to the requested payload while ensuring at least
        # one byte exists.
        #
        # ----------------------------------------------------

        message_bytes = max(1, n_bits // 8)

        message = (
            MESSAGE_BYTE * message_bytes
        ).decode("ascii")

        # ----------------------------------------------------
        # EMBEDDING CONFIGURATION
        # ----------------------------------------------------
        #
        # Use the actual detected layer count rather than
        # relying on a hardcoded 32.
        #
        # ----------------------------------------------------

        config = EmbeddingConfig(
            total_payload_bits=n_bits,
            model_family=family,
            num_hidden_layers=actual_layers,
        )

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------
        #
        # Pass all three representations so QACI can use
        # the complete feature set:
        #
        #   residuals
        #   FP16 weights
        #   quantized weights
        #
        # Time Complexity: O(number of selected carriers)
        # Space Complexity: O(number of selected carriers)
        # ----------------------------------------------------

        try:

            embed_result = IntelligentEmbedder(config).embed(
                message,
                residuals,
                fp16_weights=fp16_weights,
                quantized_weights=quantized_weights,
            )

        except Exception as exc:

            print(
                f"{n_bits:>9,} bits — "
                f"EMBED FAILED: {exc}"
            )

            results.append(
                {
                    "payload_bits": n_bits,
                    "ber": None,
                    "status": "EMBED_FAILED",
                }
            )

            continue

        # ----------------------------------------------------
        # RECOVERY
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # Always pass the embedded residuals and the exact
        # carrier indices returned by the embedding stage.
        #
        # ----------------------------------------------------

        try:

            recovered, status = DecryptPipeline(
                key=embed_result.key
            ).run(
                embed_result.embedded_residuals,
                embed_result.carrier_indices,
            )

        except Exception as exc:

            print(
                f"{n_bits:>9,} bits — "
                f"RECOVERY FAILED: {exc}"
            )

            results.append(
                {
                    "payload_bits": n_bits,
                    "ber": None,
                    "status": "RECOVERY_FAILED",
                }
            )

            continue

        # ----------------------------------------------------
        # ACTUAL BER
        # ----------------------------------------------------
        #
        # Do NOT simply check whether recovered is non-empty.
        # Compare the original and recovered plaintext bits.
        #
        # Time Complexity: O(message length)
        # Space Complexity: O(message length)
        # ----------------------------------------------------

        if status.get("success", False):

            ber = calculate_ber(
                message,
                recovered,
            )

        else:

            ber = 1.0

        passed = (
            status.get("success", False)
            and ber == 0.0
        )

        print(
            f"{n_bits:>9,} bits — "
            f"BER={ber:.6f} — "
            f"{'PASS' if passed else 'FAIL'}"
        )

        results.append(
            {
                "payload_bits": n_bits,
                "ber": ber,
                "status": "PASS" if passed else "FAIL",
            }
        )

    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    successful_payloads = [
        row["payload_bits"]
        for row in results
        if row["status"] == "PASS"
    ]

    print("\n" + "=" * 75)
    print(f"CAPACITY SUMMARY — {model_id}")
    print("=" * 75)

    for row in results:

        if row["ber"] is None:
            print(
                f"{row['payload_bits']:>9,} bits : "
                f"{row['status']}"
            )

        else:
            print(
                f"{row['payload_bits']:>9,} bits : "
                f"BER={row['ber']:.6f} : "
                f"{row['status']}"
            )

    if successful_payloads:

        max_clean_payload = max(successful_payloads)

        print(
            f"\nMaximum tested payload at BER=0: "
            f"{max_clean_payload:,} bits"
        )

    else:

        print(
            "\nNo tested payload achieved BER=0."
        )

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    del nf4_model
    del fp16_model
    del residuals
    del fp16_weights
    del quantized_weights

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


print("\n" + "=" * 75)
print("EXPERIMENT 4 COMPLETE")
print("=" * 75)