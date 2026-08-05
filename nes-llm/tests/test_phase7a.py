"""Tests for Phase 7A — Magnitude-Aware Embedding Strategy."""

import torch
from src.embedding.strategies.magnitude_aware_strategy import MagnitudeAwareStrategy
from src.extraction.magnitude_aware_extractor          import MagnitudeAwareExtractor
from src.embedding.strategies                          import get_strategy, STRATEGY_REGISTRY
from src.evaluation.strategy_comparator                import StrategyComparator
from src.core.types                                    import EmbeddingConfig


def make_residuals(n=4, size=5000):
    return {i: torch.randn(size) * 0.05 for i in range(n)}

def make_indices(residuals, total_bits):
    n         = len(residuals)
    per_layer = total_bits // n
    return {lid: list(range(per_layer)) for lid in residuals}


class TestMagnitudeAwareStrategy:

    def test_bit1_positive(self):
        config   = EmbeddingConfig(total_payload_bits=100,
                                   embedding_strategy="magnitude_aware", alpha=0.25)
        strategy = MagnitudeAwareStrategy(config)
        result   = strategy.get_bit_for_residual(0.05, 1)
        assert result > 0

    def test_bit0_negative(self):
        config   = EmbeddingConfig(total_payload_bits=100,
                                   embedding_strategy="magnitude_aware", alpha=0.25)
        strategy = MagnitudeAwareStrategy(config)
        result   = strategy.get_bit_for_residual(0.05, 0)
        assert result < 0

    def test_magnitude_boosted(self):
        """Embedded magnitude should be larger than original."""
        config   = EmbeddingConfig(total_payload_bits=100,
                                   embedding_strategy="magnitude_aware", alpha=0.25)
        strategy = MagnitudeAwareStrategy(config)
        original = 0.05
        embedded = strategy.get_bit_for_residual(original, 1)
        assert abs(embedded) > abs(original)

    def test_clean_ber_zero(self):
        config    = EmbeddingConfig(total_payload_bits=1000,
                                    embedding_strategy="magnitude_aware", alpha=0.25)
        strategy  = MagnitudeAwareStrategy(config)
        extractor = MagnitudeAwareExtractor()
        residuals = make_residuals(n=4, size=2000)
        bits      = [i % 2 for i in range(1000)]
        indices   = make_indices(residuals, 1000)

        result    = strategy.embed(residuals, bits, indices)
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        ber       = extractor.calculate_ber(bits, recovered)
        assert ber == 0.0, f"Expected BER=0, got {ber}"

    def test_noise_robustness_better_than_sign(self):
        """
        Magnitude-aware should have equal or better BER at σ=0.002
        because the margin creates a larger decision boundary.
        """
        from src.embedding.sign_strategy_v2 import SignEmbeddingStrategy
        from src.extraction.sign_extractor  import SignExtractor

        n_bits    = 2000
        residuals = make_residuals(n=4, size=5000)
        bits      = [i % 2 for i in range(n_bits)]
        indices   = make_indices(residuals, n_bits)
        sigma     = 0.002

        # Sign-based
        sign_config = EmbeddingConfig(total_payload_bits=n_bits,
                                      embedding_strategy="sign", alpha=0.25)
        sign_emb  = SignEmbeddingStrategy(sign_config)
        sign_ext  = SignExtractor()
        sign_res  = sign_emb.embed(residuals, bits, indices)

        # Magnitude-aware (alpha=0.25 means 25% margin boost)
        mag_config  = EmbeddingConfig(total_payload_bits=n_bits,
                                      embedding_strategy="magnitude_aware", alpha=0.25)
        mag_emb   = MagnitudeAwareStrategy(mag_config)
        mag_ext   = MagnitudeAwareExtractor()
        mag_res   = mag_emb.embed(residuals, bits, indices)

        # Add noise and measure BER
        def add_noise(weights, s):
            return {lid: t + torch.randn_like(t) * s for lid, t in weights.items()}

        sign_noisy = add_noise(sign_res.embedded_weights, sigma)
        mag_noisy  = add_noise(mag_res.embedded_weights, sigma)

        sign_rec = sign_ext.extract(sign_noisy, sign_res.carrier_indices)
        mag_rec  = mag_ext.extract(mag_noisy, mag_res.carrier_indices)

        ber_sign = sign_ext.calculate_ber(bits, sign_rec)
        ber_mag  = mag_ext.calculate_ber(bits, mag_rec)

        print(f"\n    BER sign={ber_sign:.6f}, BER mag_aware={ber_mag:.6f} at σ={sigma}")
        # Magnitude-aware should be <= sign BER (larger margin = more robust)
        assert ber_mag <= ber_sign + 0.01

    def test_precompute_margins(self):
        config    = EmbeddingConfig(total_payload_bits=500,
                                    embedding_strategy="magnitude_aware", alpha=0.25)
        strategy  = MagnitudeAwareStrategy(config)
        residuals = make_residuals(n=4, size=1000)
        strategy.precompute_margins(residuals)
        assert len(strategy._precomputed_margins) == 4
        for lid, margins in strategy._precomputed_margins.items():
            assert (margins > 0).all()

    def test_with_quality_scores(self):
        """High quality scores should produce larger margins."""
        config    = EmbeddingConfig(total_payload_bits=100,
                                    embedding_strategy="magnitude_aware", alpha=0.25)
        residuals = make_residuals(n=2, size=1000)

        low_quality  = {lid: torch.zeros(1000) for lid in residuals}
        high_quality = {lid: torch.ones(1000)  for lid in residuals}

        strat_low  = MagnitudeAwareStrategy(config, quality_scores=low_quality)
        strat_high = MagnitudeAwareStrategy(config, quality_scores=high_quality)

        strat_low.precompute_margins(residuals)
        strat_high.precompute_margins(residuals)

        for lid in residuals:
            assert strat_high._precomputed_margins[lid].mean() >= \
                   strat_low._precomputed_margins[lid].mean()


class TestMagnitudeAwareExtractor:

    def test_positive_gives_1(self):
        assert MagnitudeAwareExtractor().get_bit_from_residual(0.1) == 1

    def test_negative_gives_0(self):
        assert MagnitudeAwareExtractor().get_bit_from_residual(-0.1) == 0

    def test_is_extractor(self):
        from src.core import Extractor
        assert isinstance(MagnitudeAwareExtractor(), Extractor)


class TestStrategyRegistry:

    def test_get_sign_strategy(self):
        config = EmbeddingConfig(total_payload_bits=100, embedding_strategy="sign")
        emb, ext = get_strategy("sign", config)
        assert emb is not None
        assert ext is not None

    def test_get_magnitude_aware_strategy(self):
        config = EmbeddingConfig(total_payload_bits=100,
                                 embedding_strategy="magnitude_aware")
        emb, ext = get_strategy("magnitude_aware", config)
        assert emb is not None
        assert ext is not None

    def test_unknown_strategy_raises(self):
        config = EmbeddingConfig(total_payload_bits=100, embedding_strategy="unknown")
        try:
            get_strategy("unknown", config)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_all_strategies_registered(self):
        assert "sign"            in STRATEGY_REGISTRY
        assert "magnitude_aware" in STRATEGY_REGISTRY


class TestStrategyComparator:

    def test_compare_returns_all_strategies(self):
        residuals  = make_residuals(n=4, size=3000)
        comparator = StrategyComparator(sigmas=[0.0, 0.001], num_trials=1)
        results    = comparator.compare(residuals, total_bits=1000)
        assert "sign"            in results
        assert "magnitude_aware" in results

    def test_all_metrics_present(self):
        residuals  = make_residuals(n=2, size=2000)
        comparator = StrategyComparator(sigmas=[0.0, 0.001], num_trials=1)
        results    = comparator.compare(residuals, total_bits=500)
        expected   = {
            "ppl_degradation", "sign_flip_rate", "mean_abs_change",
            "kl_fidelity", "ber_curve", "kl_security",
            "detector_accuracy", "security_status", "robustness_status"
        }
        for strategy in results:
            assert expected.issubset(results[strategy].keys())

    def test_clean_ber_zero_all_strategies(self):
        residuals  = make_residuals(n=4, size=3000)
        comparator = StrategyComparator(sigmas=[0.0], num_trials=1)
        results    = comparator.compare(residuals, total_bits=1000)
        for name, metrics in results.items():
            ber = metrics["ber_curve"].get(0.0, 1.0)
            assert ber == 0.0, f"{name}: Expected BER=0 clean, got {ber}"

    def test_magnitude_aware_larger_distortion(self):
        """Magnitude-aware trades more distortion for more robustness."""
        residuals  = make_residuals(n=4, size=3000)
        comparator = StrategyComparator(sigmas=[0.0], num_trials=1)
        results    = comparator.compare(residuals, total_bits=1000)
        dist_sign  = results["sign"]["mean_abs_change"]
        dist_mag   = results["magnitude_aware"]["mean_abs_change"]
        # Magnitude-aware always has >= distortion than sign-based
        assert dist_mag >= dist_sign


if __name__ == "__main__":
    import sys
    classes = [
        TestMagnitudeAwareStrategy(),
        TestMagnitudeAwareExtractor(),
        TestStrategyRegistry(),
        TestStrategyComparator(),
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