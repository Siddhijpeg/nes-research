"""Tests for Phase 5 — parameter tuning."""

import torch
from src.evaluation.capacity_robustness_tradeoff import CapacityRobustnessAnalyser
from src.evaluation.alpha_tuner                  import AlphaTuner
from src.evaluation.gamma_tuner                  import GammaTuner
from src.evaluation.optimal_config_finder        import OptimalConfigFinder


def make_residuals(n=4, size=3000):
    return {i: torch.randn(size) * 0.05 for i in range(n)}


class TestCapacityRobustnessAnalyser:

    def test_surface_keys(self):
        res     = make_residuals()
        analyser = CapacityRobustnessAnalyser(num_trials=1)
        surface  = analyser.sweep(res, payload_sizes=[500, 1000], sigmas=[0.0, 0.001])
        assert (500, 0.0)   in surface
        assert (1000, 0.001) in surface

    def test_ber_zero_at_clean(self):
        res     = make_residuals()
        analyser = CapacityRobustnessAnalyser(num_trials=1)
        surface  = analyser.sweep(res, payload_sizes=[500], sigmas=[0.0])
        assert surface[(500, 0.0)] == 0.0

    def test_find_optimal_returns_dict(self):
        res     = make_residuals()
        analyser = CapacityRobustnessAnalyser(num_trials=1)
        surface  = analyser.sweep(res, payload_sizes=[500, 1000], sigmas=[0.001])
        result   = analyser.find_optimal(surface, target_sigma=0.001, max_ber=0.05)
        assert "optimal_payload" in result
        assert "status" in result


class TestAlphaTuner:

    def test_recommended_alpha_in_range(self):
        res     = make_residuals()
        bits    = [i % 2 for i in range(1000)]
        indices = {i: list(range(250)) for i in range(4)}
        tuner   = AlphaTuner(num_trials=1)
        result  = tuner.sweep(res, bits, indices, alphas=[0.10, 0.25, 0.40])
        assert result["recommended_alpha"] in [0.10, 0.25, 0.40]

    def test_higher_alpha_lower_ber(self):
        res     = make_residuals(size=5000)
        bits    = [i % 2 for i in range(1000)]
        indices = {i: list(range(250)) for i in range(4)}
        tuner   = AlphaTuner(num_trials=2)
        result  = tuner.sweep(res, bits, indices,
                              alphas=[0.05, 0.50], sigmas=[0.002])
        ber_low  = next(r["ber_per_sigma"][0.002]
                        for r in result["results"] if r["alpha"] == 0.05)
        ber_high = next(r["ber_per_sigma"][0.002]
                        for r in result["results"] if r["alpha"] == 0.50)
        assert ber_high <= ber_low + 0.05


class TestGammaTuner:

    def test_recommended_gamma_in_input(self):
        res    = make_residuals()
        tuner  = GammaTuner(num_trials=1, target_sigma=0.001)
        result = tuner.sweep(res, total_bits=1000, gammas=[1.0, 2.5, 4.0])
        assert result["recommended_gamma"] in [1.0, 2.5, 4.0]

    def test_results_have_expected_keys(self):
        res    = make_residuals()
        tuner  = GammaTuner(num_trials=1)
        result = tuner.sweep(res, total_bits=500, gammas=[2.5])
        r = result["results"][0]
        assert "gamma"   in r
        assert "entropy" in r
        assert "ber"     in r


class TestOptimalConfigFinder:

    def test_returns_config_and_params(self):
        res    = make_residuals(n=4, size=3000)
        finder = OptimalConfigFinder(verbose=False)
        config, params = finder.find(res)
        assert "optimal_payload_bits" in params
        assert "optimal_alpha"        in params
        assert "optimal_gamma"        in params

    def test_config_has_payload(self):
        res    = make_residuals(n=4, size=3000)
        finder = OptimalConfigFinder(verbose=False)
        config, params = finder.find(res)
        assert config.total_payload_bits > 0


if __name__ == "__main__":
    import sys
    classes = [TestCapacityRobustnessAnalyser(), TestAlphaTuner(),
               TestGammaTuner(), TestOptimalConfigFinder()]
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