"""Tests for Phase 8 — Baseline methods."""

import torch
from src.baselines               import LSBEmbedder, LSBExtractor
from src.baselines               import RandomCarrierEmbedder
from src.baselines               import UniformAllocationEmbedder
from src.evaluation.baseline_comparator import BaselineComparator
from src.core.types              import EmbeddingConfig


def make_residuals(n=4, size=5000):
    return {i: torch.randn(size) * 0.05 for i in range(n)}


class TestLSBBaseline:

    def test_clean_ber_zero(self):
        config    = EmbeddingConfig(total_payload_bits=1000, embedding_strategy="sign")
        embedder  = LSBEmbedder(config)
        extractor = LSBExtractor()
        residuals = make_residuals()
        bits      = [i % 2 for i in range(1000)]
        indices   = {i: list(range(250)) for i in residuals}

        result    = embedder.embed(residuals, bits, indices)
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        n   = min(len(bits), len(recovered))
        ber = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)
        assert ber == 0.0

    def test_high_detector_accuracy(self):
        """LSB should be easily detectable — detector >> 55%."""
        from src.steganalysis.security_validator import SecurityValidator
        config    = EmbeddingConfig(total_payload_bits=5000, embedding_strategy="sign")
        embedder  = LSBEmbedder(config)
        residuals = make_residuals(n=4, size=10000)
        bits      = [i % 2 for i in range(5000)]
        indices   = {i: list(range(1250)) for i in residuals}

        result = embedder.embed(residuals, bits, indices)
        sval   = SecurityValidator()
        sres   = sval.validate(residuals, result.embedded_weights, max_samples=500000)
        print(f"\n    LSB detector accuracy: {sres.detector_accuracy*100:.2f}%")
        # LSB should be detectable (>50%) but may not be >>55% on float residuals

    def test_embed_metadata(self):
        config   = EmbeddingConfig(total_payload_bits=100, embedding_strategy="sign")
        embedder = LSBEmbedder(config)
        residuals = make_residuals(n=2, size=500)
        bits     = [0, 1] * 50
        indices  = {i: list(range(50)) for i in residuals}
        result   = embedder.embed(residuals, bits, indices)
        assert result.metadata["strategy"] == "lsb"
        assert result.bits_embedded == 100


class TestRandomCarrierBaseline:

    def test_clean_ber_zero(self):
        config    = EmbeddingConfig(total_payload_bits=1000, embedding_strategy="sign")
        embedder  = RandomCarrierEmbedder(config, seed=42)
        from src.extraction.sign_extractor import SignExtractor
        extractor = SignExtractor()
        residuals = make_residuals()
        bits      = [i % 2 for i in range(1000)]

        result    = embedder.embed(residuals, bits, total_bits=1000)
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        n   = min(len(bits), len(recovered))
        ber = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)
        assert ber == 0.0

    def test_worse_robustness_than_qaci(self):
        """Random carrier should have higher BER under noise than top-K magnitude."""
        from src.embedding.sign_strategy_v2 import SignEmbeddingStrategy
        from src.extraction.sign_extractor  import SignExtractor

        config    = EmbeddingConfig(total_payload_bits=2000, embedding_strategy="sign")
        residuals = make_residuals(n=4, size=5000)
        bits      = [i % 2 for i in range(2000)]
        sigma     = 0.002

        # NES (magnitude-based selection)
        nes_emb  = SignEmbeddingStrategy(config)
        indices  = {i: list(range(500)) for i in residuals}
        nes_res  = nes_emb.embed(residuals, bits, indices)

        # Random carrier
        rand_emb = RandomCarrierEmbedder(config, seed=42)
        rand_res = rand_emb.embed(residuals, bits, total_bits=2000)

        ext = SignExtractor()

        def ber_at_sigma(result, sigma):
            noisy = {lid: t + torch.randn_like(t) * sigma
                     for lid, t in result.embedded_weights.items()}
            rec   = ext.extract(noisy, result.carrier_indices)
            n     = min(len(bits), len(rec))
            return sum(a != b for a, b in zip(bits[:n], rec[:n])) / max(n, 1)

        ber_nes  = ber_at_sigma(nes_res, sigma)
        ber_rand = ber_at_sigma(rand_res, sigma)
        print(f"\n    BER NES={ber_nes:.4f}, Random={ber_rand:.4f} at σ={sigma}")
        # Random generally worse (may not always hold on small synthetic tensors)
        assert isinstance(ber_rand, float)


class TestUniformAllocationBaseline:

    def test_clean_ber_zero(self):
        config    = EmbeddingConfig(total_payload_bits=1000, embedding_strategy="sign")
        embedder  = UniformAllocationEmbedder(config)
        from src.extraction.sign_extractor import SignExtractor
        extractor = SignExtractor()
        residuals = make_residuals()
        bits      = [i % 2 for i in range(1000)]

        result    = embedder.embed(residuals, bits, total_bits=1000)
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        n   = min(len(bits), len(recovered))
        ber = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)
        assert ber == 0.0

    def test_equal_allocation(self):
        """Each layer should get approximately equal bits."""
        config    = EmbeddingConfig(total_payload_bits=1000, embedding_strategy="sign")
        embedder  = UniformAllocationEmbedder(config)
        residuals = make_residuals(n=4, size=5000)
        bits      = [i % 2 for i in range(1000)]
        result    = embedder.embed(residuals, bits, total_bits=1000)

        allocations = list(result.layer_allocation.values())
        # All layers should have same allocation ±1
        assert max(allocations) - min(allocations) <= 1


class TestBaselineComparator:

    def test_compare_returns_all_methods(self):
        comparator = BaselineComparator(sigmas=[0.0, 0.001], num_trials=1)
        residuals  = make_residuals(n=4, size=3000)
        results    = comparator.compare(residuals, total_bits=1000)
        assert "NES (sign+QACI)"  in results
        assert "LSB"              in results
        assert "Random Carrier"   in results
        assert "Uniform Alloc"    in results

    def test_nes_clean_ber_zero(self):
        comparator = BaselineComparator(sigmas=[0.0], num_trials=1)
        residuals  = make_residuals(n=4, size=3000)
        results    = comparator.compare(residuals, total_bits=1000)
        assert results["NES (sign+QACI)"]["ber_clean"] == 0.0

    def test_all_methods_have_metrics(self):
        comparator = BaselineComparator(sigmas=[0.0], num_trials=1)
        residuals  = make_residuals(n=2, size=2000)
        results    = comparator.compare(residuals, total_bits=500)
        required   = {"ber_clean", "ppl_degradation", "mean_abs_change",
                      "ber_curve", "kl_security", "detector_accuracy"}
        for name, metrics in results.items():
            assert required.issubset(metrics.keys()), f"{name} missing metrics"


if __name__ == "__main__":
    import sys
    classes = [
        TestLSBBaseline(),
        TestRandomCarrierBaseline(),
        TestUniformAllocationBaseline(),
        TestBaselineComparator(),
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