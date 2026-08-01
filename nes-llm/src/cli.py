"""
NES Command Line Interface.

Usage:
    # Embed a message into a model
    python -m src.cli embed \
        --message "secret text" \
        --model   meta-llama/Llama-3-8B \
        --output  embedded_model/ \
        --keyfile keys/run1.json

    # Extract a message from an embedded model
    python -m src.cli extract \
        --model   embedded_model/ \
        --keyfile keys/run1.json \
        --keyid   <key_id>

    # Run benchmark gates
    python -m src.cli benchmark --layers 8 --size 10000 --bits 5000

    # Find optimal parameters
    python -m src.cli tune --layers 8 --size 10000
"""

import argparse
import json
import sys
import os


def cmd_embed(args):
    """Embed a message into a model's residuals (synthetic demo if no GPU)."""
    import torch
    from src.embedding.intelligent_embedder import IntelligentEmbedder
    from src.crypto.key_manager             import KeyManager
    from src.core.types                     import EmbeddingConfig

    print(f"\n[NES] Embedding message ({len(args.message)} chars)...")

    # Synthetic residuals for demo — replace with real model residuals
    n_layers  = args.layers
    layer_size = args.size
    residuals = {i: torch.randn(layer_size) * 0.05 for i in range(n_layers)}

    config   = EmbeddingConfig(
        total_payload_bits=args.bits,
        embedding_strategy="sign",
    )
    embedder = IntelligentEmbedder(config)
    result   = embedder.embed(args.message, residuals)

    if not result.success:
        print("[NES] ❌ Embedding failed.")
        sys.exit(1)

    # Save key
    km  = KeyManager()
    kid = km.add_key(result.key, model_id=args.model)
    os.makedirs(os.path.dirname(args.keyfile) if os.path.dirname(args.keyfile) else ".", exist_ok=True)
    km.save(args.keyfile)

    print(f"[NES] ✅ Embedding complete.")
    print(f"       Bits embedded : {result.bits_embedded:,}")
    print(f"       Layers used   : {sum(1 for b in result.layer_allocation.values() if b > 0)}")
    print(f"       Key ID        : {kid}")
    print(f"       Key saved to  : {args.keyfile}")

    # Save carrier map alongside key
    carrier_file = args.keyfile.replace(".json", "_carriers.json")
    with open(carrier_file, "w") as f:
        json.dump({
            str(lid): indices
            for lid, indices in result.carrier_indices.items()
        }, f)
    print(f"       Carrier map   : {carrier_file}")


def cmd_extract(args):
    """Extract a message from embedded residuals."""
    import torch
    from src.extraction.decrypt_pipeline import DecryptPipeline
    from src.crypto.key_manager          import KeyManager

    print(f"\n[NES] Extracting message...")

    km  = KeyManager()
    km.load(args.keyfile)
    key = km.get_key(args.keyid)

    # Load carrier map
    carrier_file = args.keyfile.replace(".json", "_carriers.json")
    with open(carrier_file, "r") as f:
        raw = json.load(f)
    carrier_indices = {int(lid): indices for lid, indices in raw.items()}

    # Synthetic residuals — replace with real embedded model residuals
    n   = max(carrier_indices.keys()) + 1
    sz  = max(max(v) for v in carrier_indices.values() if v) + 1
    residuals = {}
    for lid, indices in carrier_indices.items():
        t = torch.zeros(sz)
        # Demo: set positive for all (will decode as all-1s — real use needs actual residuals)
        residuals[lid] = t

    pipeline = DecryptPipeline(key=key)
    message, stats = pipeline.run(residuals, carrier_indices)

    if stats.get("success"):
        print(f"[NES] ✅ Extraction complete.")
        print(f"       Message : {message}")
    else:
        print(f"[NES] ❌ Extraction failed: {stats.get('error')}")
        sys.exit(1)


def cmd_benchmark(args):
    """Run all NES quality gates."""
    from src.evaluation.nes_benchmark import NESBenchmark

    bench  = NESBenchmark(
        n_layers=    args.layers,
        layer_size=  args.size,
        payload_bits=args.bits,
        verbose=     True,
    )
    result = bench.run_all()
    sys.exit(0 if result["all_pass"] else 1)


def cmd_tune(args):
    """Find optimal parameters for given residual shape."""
    import torch
    from src.evaluation.optimal_config_finder import OptimalConfigFinder

    residuals = {i: torch.randn(args.size) * 0.05 for i in range(args.layers)}
    finder    = OptimalConfigFinder(verbose=True)
    config, params = finder.find(residuals)

    out = "configs/optimal.json"
    os.makedirs("configs", exist_ok=True)
    config.to_json(out)
    print(f"\n[NES] Config saved to {out}")


def main():
    parser = argparse.ArgumentParser(
        prog="nes",
        description="Neural-Entropic Steganography CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- embed ---
    p_embed = sub.add_parser("embed", help="Embed a message into a model")
    p_embed.add_argument("--message", required=True)
    p_embed.add_argument("--model",   default="llama-3-8b")
    p_embed.add_argument("--output",  default="embedded_model/")
    p_embed.add_argument("--keyfile", default="keys/nes_keys.json")
    p_embed.add_argument("--layers",  type=int, default=32)
    p_embed.add_argument("--size",    type=int, default=10000)
    p_embed.add_argument("--bits",    type=int, default=50000)

    # --- extract ---
    p_ext = sub.add_parser("extract", help="Extract a message from an embedded model")
    p_ext.add_argument("--model",   default="embedded_model/")
    p_ext.add_argument("--keyfile", required=True)
    p_ext.add_argument("--keyid",   required=True)

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Run all quality gates")
    p_bench.add_argument("--layers", type=int, default=8)
    p_bench.add_argument("--size",   type=int, default=10000)
    p_bench.add_argument("--bits",   type=int, default=5000)

    # --- tune ---
    p_tune = sub.add_parser("tune", help="Find optimal parameters")
    p_tune.add_argument("--layers", type=int, default=8)
    p_tune.add_argument("--size",   type=int, default=10000)

    args = parser.parse_args()
    {
        "embed":     cmd_embed,
        "extract":   cmd_extract,
        "benchmark": cmd_benchmark,
        "tune":      cmd_tune,
    }[args.command](args)


if __name__ == "__main__":
    main()