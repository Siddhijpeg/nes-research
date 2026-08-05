"""Tests for Phase 7D — Adaptive Meta-Strategy."""

import os
import torch
from src.embedding.strategies.adaptive_strategy import AdaptiveStrategy
from src.extraction.adaptive_extractor          import AdaptiveExtractor
from src.core.types                             import EmbeddingConfig


def make_residuals(n=4, size=5000, std=0.025):
    return {i: torch.randn(size) * std for i in range(n)}

def make_indices(residuals, total_bits):
    n   = len(residuals)
    per = total_bits // n
    return {lid: list(range(per)) for lid in residuals}


class TestNoiseEstimation:

    def test_low_noise_residuals(self):
        """Very smooth residuals → low sigma estimate."""
        config   = EmbeddingConfig(total_payload_bits=100, embedding_strategy="adaptive")
        strategy = AdaptiveStrategy(config)
        # Smooth residuals: linearly spaced, near-zero noise
        residuals = {0: torch.linspace(-0.1, 0.1, 10000)}
        sigma = strategy.estimate_noise(residuals)
        print(f"\n    Smooth residuals σ_est={sigma:.6f}")
        assert sigma < 0.01

    def test_noisy_residuals_higher_estimate(self):
        """Noisy residuals → higher sigma estimate."""
        config    = EmbeddingConfig(total_payload_bits=100, embedding_strategy="adaptive")
        strategy  = AdaptiveStrategy(config)
        clean_res = {0: torch.randn(10000) * 0.001}
        noisy_res = {0: torch.randn(10000) * 0.1}
        sigma_clean = strategy.estimate_noise(clean_res)
        sigma_noisy = strategy.estimate_noise(noisy_res)
        print(f"\n    σ_clean={sigma_clean:.6f}, σ_noisy={sigma_noisy:.6f}")
        assert sigma_noisy > sigma_clean

    def test_estimate_consistent(self):
        """Same residuals → same estimate (deterministic)."""
        config    = EmbeddingConfig(total_payload_bits=100, embedding_strategy="adaptive")
        strategy  = AdaptiveStrategy(config)
        residuals = make_residuals()
        s1 = strategy.estimate_noise(residuals)
        s2 = strategy.estimate_noise(residuals)
        assert abs(s1 - s2) < 1e-10


class TestStrategySelection:

    def test_low_noise_selects_lwe(self):
        config   = EmbeddingConfig(total_payload_bits=100, embedding_strategy="adaptive")
        strategy = AdaptiveStrategy(config, force_strategy=None)
        selected = strategy.select_strategy(0.0001)
        assert selected == "lwe"

    def test_medium_noise_selects_neural(self):
        config   = EmbeddingConfig(total_payload_bits=100, embedding_strategy="adaptive")
        strategy = AdaptiveStrategy(config)
        selected = strategy.select_strategy(0.001)
        assert selected == "neural"

    def test_high_noise_selects_sign(self):
        config   = EmbeddingConfig(total_payload_bits=100, embedding_strategy="adaptive")
        strategy = AdaptiveStrategy(config)
        selected = strategy.select_strategy(0.01)
        assert selected == "sign"

    def test_force_strategy_overrides(self):
        config   = EmbeddingConfig(total_payload_bits=100, embedding_strategy="adaptive")
        strategy = AdaptiveStrategy(config, force_strategy="magnitude_aware")
        # Should always return forced strategy regardless of sigma
        assert strategy.select_strategy(0.0) == "magnitude_aware"
        assert strategy.select_strategy(0.01) == "magnitude_aware"


class TestAdaptiveEmbedSign:
    """Force sign strategy to avoid needing neural model."""

    def test_embed_extract_roundtrip(self):
        config   = EmbeddingConfig(total_payload_bits=1000, embedding_strategy="adaptive")
        strategy = AdaptiveStrategy(config, force_strategy="sign")
        residuals = make_residuals(n=4, size=3000)
        bits      = [i % 2 for i in range(1000)]
        indices   = make_indices(residuals, 1000)

        result = strategy.embed(residuals, bits, indices)
        assert result.success
        assert result.metadata["selected_strategy"] == "sign"
        assert result.bits_embedded == 1000

        extractor = AdaptiveExtractor(
            strategy_name="sign",
        )
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        n   = min(len(bits), len(recovered))
        ber = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)
        assert ber == 0.0

    def test_metadata_contains_strategy_and_sigma(self):
        config   = EmbeddingConfig(total_payload_bits=500, embedding_strategy="adaptive")
        strategy = AdaptiveStrategy(config, force_strategy="sign")
        residuals = make_residuals(n=2, size=2000)
        bits      = [0, 1] * 250
        indices   = {lid: list(range(250)) for lid in residuals}

        result = strategy.embed(residuals, bits, indices)
        assert "selected_strategy" in result.metadata
        assert "estimated_sigma"   in result.metadata
        assert result.metadata["selected_strategy"] == "sign"


class TestAdaptiveWithLWE:

    def test_lwe_embed_extract(self):
        import secrets
        key      = secrets.token_bytes(32)
        config   = EmbeddingConfig(total_payload_bits=1000, embedding_strategy="adaptive")
        strategy = AdaptiveStrategy(config, force_strategy="lwe", secret_key=key)
        residuals = make_residuals(n=4, size=3000)
        bits      = [i % 2 for i in range(1000)]
        indices   = make_indices(residuals, 1000)

        result = strategy.embed(residuals, bits, indices)
        assert result.metadata["selected_strategy"] == "lwe"

        extractor = AdaptiveExtractor(
            strategy_name="lwe",
            secret_key=key,
            residuals_ref=residuals,
        )
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        n   = min(len(bits), len(recovered))
        ber = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)
        assert ber == 0.0, f"LWE roundtrip BER={ber}"


class TestAdaptiveWithNeural:

    def test_neural_embed_extract(self):
        neural_path = "models/tinyllama_neural_embedder.pt"
        if not os.path.exists(neural_path):
            print(f"\n    Skipping — {neural_path} not found")
            return

        config   = EmbeddingConfig(total_payload_bits=1000, embedding_strategy="adaptive")
        strategy = AdaptiveStrategy(
            config,
            force_strategy="neural",
            neural_model_path=neural_path,
        )
        residuals = make_residuals(n=4, size=3000)
        bits      = [i % 2 for i in range(1000)]
        indices   = make_indices(residuals, 1000)

        result = strategy.embed(residuals, bits, indices)
        assert result.metadata["selected_strategy"] == "neural"

        extractor = AdaptiveExtractor(
            strategy_name="neural",
            neural_model_path=neural_path,
        )
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        n   = min(len(bits), len(recovered))
        ber = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)
        print(f"\n    Neural adaptive BER: {ber:.4f}")
        assert ber < 0.05


class TestAdaptiveAutoSelection:

    def test_auto_selects_sign_for_real_residuals(self):
        """Real TinyLlama residuals have moderate noise → sign or neural."""
        config   = EmbeddingConfig(total_payload_bits=500, embedding_strategy="adaptive")
        # TinyLlama NF4 residuals have std≈0.025, noise≈medium
        residuals = make_residuals(n=4, size=5000, std=0.025)
        strategy  = AdaptiveStrategy(config, force_strategy=None)
        sigma     = strategy.estimate_noise(residuals)
        selected  = strategy.select_strategy(sigma)
        print(f"\n    σ_est={sigma:.6f} → {selected}")
        # Should select sign or neural (not LWE for moderate noise)
        assert selected in ("sign", "neural", "magnitude_aware")


if __name__ == "__main__":
    import sys
    classes = [
        TestNoiseEstimation(),
        TestStrategySelection(),
        TestAdaptiveEmbedSign(),
        TestAdaptiveWithLWE(),
        TestAdaptiveWithNeural(),
        TestAdaptiveAutoSelection(),
    ]
    passed = failed = 0
    for obj in classes:
        cls = type(obj).__name__
        for method in [m for m in dir(obj) if m.startswith("test_")]:
            try:
                getattr(obj, method)()
                print(f"  ✅ {cls}.{method}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {cls}.{method}: {e}")
                failed += 1
    print(f"\n{'='*55}\n  {passed} passed, {failed} failed\n{'='*55}")
    sys.exit(0 if failed == 0 else 1)