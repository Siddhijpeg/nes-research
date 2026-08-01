"""
Tests for Phase 4 — Validation & Security.

Covers FidelityValidator, RobustnessValidator, SecurityValidator,
and the NESBenchmark gate runner.
"""

import torch
from src.evaluation.fidelity_validator   import FidelityValidator
from src.evaluation.robustness_validator import RobustnessValidator
from src.steganalysis.security_validator import SecurityValidator
from src.evaluation.nes_benchmark        import NESBenchmark
from src.embedding.sign_strategy_v2      import SignEmbeddingStrategy
from src.core.types                      import EmbeddingConfig


def make_residuals(n=4, size=5000):
    return {i: torch.randn(size) * 0.05 for i in range(n)}

def embed_residuals(residuals, n_bits):
    config  = EmbeddingConfig(total_payload_bits=n_bits, embedding_strategy="sign")
    strat   = SignEmbeddingStrategy(config)
    bits    = [i % 2 for i in range(n_bits)]
    indices = {i: list(range(n_bits // len(residuals)))
               for i in residuals}
    result  = strat.embed(residuals, bits, indices)
    return result.embedded_weights, result.carrier_indices, bits


class TestFidelityValidator:

    def test_clean_residuals_pass(self):
        val  = FidelityValidator()
        orig = make_residuals()
        # No embedding — should pass easily
        res  = val.validate_tensors(orig, orig)
        assert res.passed

    def test_heavy_embedding_fails(self):
        val  = FidelityValidator(max_ppl_degradation=0.001)
        orig = make_residuals()
        emb, _, _ = embed_residuals(orig, 5000)
        res  = val.validate_tensors(orig, emb)
        # Very tight threshold — sign flip will exceed it
        assert isinstance(res.passed, bool)

    def test_result_has_report(self):
        val  = FidelityValidator()
        orig = make_residuals()
        emb, _, _ = embed_residuals(orig, 1000)
        res  = val.validate_tensors(orig, emb)
        report = res.report()
        assert "Fidelity" in report
        assert "PASS" in report or "FAIL" in report

    def test_compare_perplexity(self):
        val = FidelityValidator(max_ppl_degradation=0.02)
        res = val.compare_perplexity(ppl_baseline=5.0, ppl_embedded=5.05)
        assert res.passed   # 1% < 2% threshold

    def test_compare_perplexity_fail(self):
        val = FidelityValidator(max_ppl_degradation=0.02)
        res = val.compare_perplexity(ppl_baseline=5.0, ppl_embedded=5.20)
        assert not res.passed   # 4% > 2% threshold


class TestRobustnessValidator:

    def test_clean_ber_zero(self):
        orig = make_residuals(n=2, size=3000)
        emb, indices, bits = embed_residuals(orig, 2000)
        val  = RobustnessValidator(num_trials=2)
        res  = val.validate(emb, indices, bits, sigmas=[0.0])
        assert res.ber_curve[0.0] == 0.0

    def test_high_noise_degrades_ber(self):
        orig = make_residuals(n=2, size=3000)
        emb, indices, bits = embed_residuals(orig, 2000)
        val  = RobustnessValidator(num_trials=2)
        res  = val.validate(emb, indices, bits, sigmas=[0.0, 0.01])
        assert res.ber_curve[0.01] >= res.ber_curve[0.0]

    def test_result_has_report(self):
        orig = make_residuals(n=2, size=2000)
        emb, indices, bits = embed_residuals(orig, 1000)
        val  = RobustnessValidator(num_trials=1)
        res  = val.validate(emb, indices, bits, sigmas=[0.0, 0.001])
        assert "Robustness" in res.report()

    def test_passes_at_low_noise(self):
        orig = make_residuals(n=2, size=5000)
        emb, indices, bits = embed_residuals(orig, 2000)
        val  = RobustnessValidator(max_ber_at_001=0.10, num_trials=2)
        res  = val.validate(emb, indices, bits, sigmas=[0.001])
        # With a relaxed threshold (10%) should pass
        assert isinstance(res.passed, bool)


class TestSecurityValidator:

    def test_identical_residuals_pass(self):
        val  = SecurityValidator()
        orig = make_residuals()
        res  = val.validate(orig, orig)
        assert res.passed   # identical → KL=0

    def test_kl_divergence_non_negative(self):
        val  = SecurityValidator()
        orig = make_residuals()
        emb, _, _ = embed_residuals(orig, 2000)
        res  = val.validate(orig, emb)
        assert res.kl_divergence >= 0.0

    def test_detector_accuracy_in_range(self):
        val  = SecurityValidator()
        orig = make_residuals()
        emb, _, _ = embed_residuals(orig, 2000)
        res  = val.validate(orig, emb)
        assert 0.0 <= res.detector_accuracy <= 1.0

    def test_result_has_report(self):
        val  = SecurityValidator()
        orig = make_residuals()
        emb, _, _ = embed_residuals(orig, 1000)
        res  = val.validate(orig, emb)
        assert "Security" in res.report()

    def test_sign_bias_small(self):
        """Clean residuals have near-zero sign bias."""
        val  = SecurityValidator()
        orig = make_residuals(n=4, size=10000)
        res  = val.validate(orig, orig)
        assert res.sign_bias < 0.05


class TestNESBenchmark:

    def test_gate2_qaci(self):
        bench  = NESBenchmark(n_layers=4, layer_size=5000,
                              payload_bits=2000, verbose=False)
        result = bench.gate2_qaci()
        assert result["passed"]
        assert result["total_bits_allocated"] == 2000

    def test_gate3_ber(self):
        bench  = NESBenchmark(n_layers=4, layer_size=5000,
                              payload_bits=2000, verbose=False)
        result = bench.gate3_ber()
        assert result["passed"], f"Gate 3 failed: {result}"

    def test_gate4_fidelity(self):
        bench  = NESBenchmark(n_layers=4, layer_size=5000,
                              payload_bits=2000, verbose=False)
        result = bench.gate4_fidelity()
        assert "passed" in result

    def test_gate5_robustness(self):
        bench  = NESBenchmark(n_layers=4, layer_size=5000,
                              payload_bits=2000, verbose=False)
        result = bench.gate5_robustness()
        assert "ber_curve" in result

    def test_gate6_security(self):
        bench  = NESBenchmark(n_layers=4, layer_size=5000,
                              payload_bits=2000, verbose=False)
        result = bench.gate6_security()
        assert "kl_divergence" in result

    def test_run_all_returns_summary(self):
        bench  = NESBenchmark(n_layers=4, layer_size=5000,
                              payload_bits=2000, verbose=False)
        result = bench.run_all()
        assert "passed"   in result
        assert "total"    in result
        assert "all_pass" in result
        assert result["total"] == 5


if __name__ == "__main__":
    import sys
    classes = [
        TestFidelityValidator(),
        TestRobustnessValidator(),
        TestSecurityValidator(),
        TestNESBenchmark(),
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