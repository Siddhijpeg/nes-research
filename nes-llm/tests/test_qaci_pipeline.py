"""
Tests for Phase 2 — QACI carrier intelligence pipeline.

Validates LayerProfiler, CarrierScheduler, QualityScore,
AdaptiveMarginController, and full QACIPipeline without requiring
a real GPU or model (uses random tensors).
"""

import torch
from src.carrier_intelligence.layer_profiler     import LayerProfiler
from src.carrier_intelligence.quality_score      import QualityScore
from src.carrier_intelligence.feature_normalizer import FeatureNormalizer
from src.carrier_intelligence.carrier_scheduler  import CarrierScheduler
from src.carrier_intelligence.adaptive_margin    import AdaptiveMarginController
from src.carrier_intelligence.selector           import CarrierSelector
from src.carrier_intelligence.qaci_pipeline      import QACIPipeline


class TestLayerProfiler:

    def test_profile_returns_expected_keys(self):
        profiler = LayerProfiler()
        residual = torch.randn(4096, 14336) * 0.05
        profile  = profiler.profile(residual, layer_id=0, total_layers=32)
        expected = {"layer_id", "module_name", "num_params", "mag_mean",
                    "mag_std", "mag_max", "entropy",
                    "quality_score", "position_bias", "adjusted_quality"}
        assert expected.issubset(profile.keys())

    def test_quality_score_in_range(self):
        profiler = LayerProfiler()
        residual = torch.randn(1000, 1000) * 0.05
        profile  = profiler.profile(residual, layer_id=15, total_layers=32)
        assert 0.0 <= profile["quality_score"]    <= 1.0
        assert 0.0 <= profile["adjusted_quality"] <= 1.0
        assert 0.0 <= profile["position_bias"]    <= 1.0

    def test_position_bias_middle_layers_higher(self):
        profiler = LayerProfiler()
        residual = torch.randn(100, 100) * 0.05
        early_p  = profiler.profile(residual, layer_id=0,  total_layers=32)
        mid_p    = profiler.profile(residual, layer_id=15, total_layers=32)
        late_p   = profiler.profile(residual, layer_id=31, total_layers=32)
        assert mid_p["position_bias"] >= early_p["position_bias"]
        assert mid_p["position_bias"] >= late_p["position_bias"]


class TestQualityScore:

    def test_output_shape(self):
        scorer   = QualityScore()
        features = torch.rand(10000, 9)
        scores   = scorer.compute(features)
        assert scores.shape == (10000,)

    def test_output_range(self):
        scorer   = QualityScore()
        features = torch.rand(1000, 9)
        scores   = scorer.compute(features)
        assert scores.min().item() >= 0.0
        assert scores.max().item() <= 1.0

    def test_layer_quality_factor_scales_output(self):
        scorer   = QualityScore()
        features = torch.rand(1000, 9)
        scores_1 = scorer.compute(features, layer_quality_factor=1.0)
        scores_h = scorer.compute(features, layer_quality_factor=0.5)
        assert scores_h.mean().item() <= scores_1.mean().item() + 1e-6


class TestFeatureNormalizer:

    def test_output_in_unit_range(self):
        normalizer = FeatureNormalizer()
        features   = torch.randn(5000, 9)
        normed     = normalizer.normalize(features)
        assert normed.min().item() >= -1e-6
        assert normed.max().item() <= 1.0 + 1e-6

    def test_constant_column_handled(self):
        normalizer = FeatureNormalizer()
        features   = torch.ones(100, 9)
        normed     = normalizer.normalize(features)
        assert not torch.isnan(normed).any()
        assert not torch.isinf(normed).any()


class TestCarrierScheduler:

    def test_total_bits_exact(self):
        scheduler = CarrierScheduler()
        profiles  = [
            {"layer_id": i, "module_name": "x",
             "adjusted_quality": 0.5 + 0.01 * i,
             "num_params": 50000}
            for i in range(32)
        ]
        allocations = scheduler.allocate(profiles, total_payload_bits=100000)
        total = sum(a.allocated_bits for a in allocations)
        assert total == 100000, f"Expected 100000, got {total}"

    def test_no_layer_exceeds_capacity(self):
        scheduler = CarrierScheduler()
        capacity  = 3000
        profiles  = [
            {"layer_id": i, "module_name": "x",
             "adjusted_quality": 0.8,
             "num_params": capacity}
            for i in range(10)
        ]
        allocations = scheduler.allocate(profiles, total_payload_bits=20000)
        for alloc in allocations:
            assert alloc.allocated_bits <= capacity

    def test_zero_quality_gets_no_bits(self):
        scheduler = CarrierScheduler()
        profiles  = [
            {"layer_id": 0, "module_name": "x",
             "adjusted_quality": 0.0, "num_params": 10000},
            {"layer_id": 1, "module_name": "x",
             "adjusted_quality": 1.0, "num_params": 10000},
        ]
        allocations = scheduler.allocate(profiles, total_payload_bits=5000)
        alloc_dict  = scheduler.allocation_to_dict(allocations)
        assert alloc_dict[0] == 0
        assert alloc_dict[1] == 5000


class TestAdaptiveMarginController:

    def test_margins_positive(self):
        amc      = AdaptiveMarginController(alpha=0.25)
        residual = torch.randn(10000) * 0.05
        scores   = torch.rand(500)
        margins  = amc.compute(residual, scores)
        assert (margins > 0).all()

    def test_embed_with_margin_bit1_positive(self):
        amc     = AdaptiveMarginController()
        values  = torch.tensor([-0.02, 0.03, -0.01])
        bits    = torch.tensor([1, 1, 1])
        margins = torch.tensor([0.005, 0.005, 0.005])
        result  = amc.embed_with_margin(values, bits, margins)
        assert (result > 0).all()

    def test_embed_with_margin_bit0_negative(self):
        amc     = AdaptiveMarginController()
        values  = torch.tensor([0.02, -0.03, 0.01])
        bits    = torch.tensor([0, 0, 0])
        margins = torch.tensor([0.005, 0.005, 0.005])
        result  = amc.embed_with_margin(values, bits, margins)
        assert (result < 0).all()


class TestQACIPipeline:

    def _make_residuals(self, n_layers=8, layer_size=1000):
        return {i: torch.randn(layer_size) * 0.05 for i in range(n_layers)}

    def test_total_bits_conserved(self):
        pipeline  = QACIPipeline(total_layers=8)
        residuals = self._make_residuals(n_layers=8, layer_size=2000)
        result    = pipeline.select(residuals, total_payload_bits=8000)
        total     = sum(len(v) for v in result.selected_indices.values())
        assert total == 8000, f"Expected 8000, got {total}"

    def test_all_layers_present(self):
        pipeline  = QACIPipeline(total_layers=8)
        residuals = self._make_residuals(n_layers=8)
        result    = pipeline.select(residuals, total_payload_bits=4000)
        assert set(result.selected_indices.keys()) == set(range(8))

    def test_indices_within_bounds(self):
        layer_size = 2000
        pipeline   = QACIPipeline(total_layers=4)
        residuals  = self._make_residuals(n_layers=4, layer_size=layer_size)
        result     = pipeline.select(residuals, total_payload_bits=2000)
        for lid, indices in result.selected_indices.items():
            for idx in indices:
                assert 0 <= idx < layer_size

    def test_no_duplicate_indices_per_layer(self):
        pipeline  = QACIPipeline(total_layers=4)
        residuals = self._make_residuals(n_layers=4, layer_size=5000)
        result    = pipeline.select(residuals, total_payload_bits=4000)
        for lid, indices in result.selected_indices.items():
            assert len(indices) == len(set(indices))

    def test_summary(self):
        pipeline  = QACIPipeline(total_layers=4)
        residuals = self._make_residuals(n_layers=4, layer_size=5000)
        result    = pipeline.select(residuals, total_payload_bits=2000)
        assert result.summary()["total_bits"] == 2000


if __name__ == "__main__":
    import sys
    tests  = [TestLayerProfiler(), TestQualityScore(), TestFeatureNormalizer(),
               TestCarrierScheduler(), TestAdaptiveMarginController(), TestQACIPipeline()]
    passed = failed = 0
    for obj in tests:
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