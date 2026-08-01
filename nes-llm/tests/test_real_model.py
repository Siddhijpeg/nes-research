"""
Real model integration tests — uses mocked BitsAndBytes model
so tests run without GPU/downloaded weights.
"""

import torch
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from src.model.residual_extractor import ResidualExtractor
from src.model.weight_patcher     import WeightPatcher
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.extraction.decrypt_pipeline    import DecryptPipeline
from src.core.types                     import EmbeddingConfig


def make_mock_model(num_layers=4, hidden=128, intermediate=512):
    """Build a minimal mock of a BitsAndBytes NF4 model."""
    import bitsandbytes as bnb

    class MockMLP:
        def __init__(self):
            # Simulate NF4 Linear4bit layer
            weight_data   = torch.randn(intermediate, hidden) * 0.05
            quant_weight, quant_state = bnb.functional.quantize_4bit(
                weight_data, quant_type="nf4", compress_statistics=True
            )
            self.down_proj        = MagicMock(spec=bnb.nn.Linear4bit)
            self.down_proj.weight = MagicMock()
            self.down_proj.weight.data        = quant_weight
            self.down_proj.weight.quant_state = quant_state
            self.down_proj.weight.device      = torch.device("cpu")

    class MockLayer:
        def __init__(self):
            self.mlp = MockMLP()

    class MockModelInner:
        def __init__(self):
            self.layers = [MockLayer() for _ in range(num_layers)]

    class MockModel:
        def __init__(self):
            self.model  = MockModelInner()
            self.config = MagicMock()
            self.config.num_hidden_layers = num_layers

    return MockModel()


class TestResidualExtractor:

    def test_extract_returns_dicts(self):
        try:
            import bitsandbytes
        except ImportError:
            print("  ⚠️  bitsandbytes not available — skipping")
            return

        model     = make_mock_model(num_layers=4)
        extractor = ResidualExtractor(target_modules=["down_proj"])
        residuals, fp16, names = extractor.extract(model)

        assert len(residuals) == 4
        assert len(fp16)      == 4
        assert len(names)     == 4

    def test_residuals_are_float_tensors(self):
        try:
            import bitsandbytes
        except ImportError:
            return

        model     = make_mock_model(num_layers=2)
        extractor = ResidualExtractor(target_modules=["down_proj"])
        residuals, _, _ = extractor.extract(model)

        for lid, tensor in residuals.items():
            assert tensor.dtype in (torch.float32, torch.float16, torch.bfloat16)


class TestFullPipelineWithMockModel:
    """
    Full embed → extract roundtrip using synthetic residuals
    (same shape as real model residuals).
    """

    def test_embed_extract_roundtrip(self):
        residuals = {i: torch.randn(10000) * 0.05 for i in range(4)}
        message   = "Real model integration test"
        config    = EmbeddingConfig(total_payload_bits=3000, embedding_strategy="sign")

        embedder     = IntelligentEmbedder(config)
        embed_result = embedder.embed(message, residuals)

        pipeline         = DecryptPipeline(key=embed_result.key)
        recovered, stats = pipeline.run(
            embed_result.embedded_residuals,
            embed_result.carrier_indices,
        )
        assert stats["success"]
        assert recovered == message

    def test_key_carrier_map_save_load(self):
        """Keys and carrier map persist to disk and reload correctly."""
        from src.crypto.key_manager import KeyManager

        residuals = {i: torch.randn(5000) * 0.05 for i in range(4)}
        config    = EmbeddingConfig(total_payload_bits=1000, embedding_strategy="sign")

        embedder     = IntelligentEmbedder(config)
        embed_result = embedder.embed("persist test", residuals)

        with tempfile.TemporaryDirectory() as tmpdir:
            key_path     = os.path.join(tmpdir, "keys.json")
            carrier_path = os.path.join(tmpdir, "carriers.json")

            km  = KeyManager()
            kid = km.add_key(embed_result.key)
            km.save(key_path)

            with open(carrier_path, "w") as f:
                json.dump(
                    {str(k): v for k, v in embed_result.carrier_indices.items()}, f
                )

            # Reload
            km2 = KeyManager()
            km2.load(key_path)
            key = km2.get_key(kid)

            with open(carrier_path) as f:
                loaded_indices = {int(k): v for k, v in json.load(f).items()}

            pipeline         = DecryptPipeline(key=key)
            recovered, stats = pipeline.run(
                embed_result.embedded_residuals, loaded_indices
            )
            assert stats["success"]
            assert recovered == "persist test"

    def test_large_payload(self):
        """50K carrier capacity across 8 layers — mirrors real Llama-3-8B scale."""
        residuals = {i: torch.randn(500000) * 0.05 for i in range(8)}
        message   = "A" * 200   # ~200 bytes → ~1,856 encrypted bits + 32-bit header
        config    = EmbeddingConfig(total_payload_bits=50000, embedding_strategy="sign")

        embedder     = IntelligentEmbedder(config)
        embed_result = embedder.embed(message, residuals)

        # bits_embedded = actual encrypted content size (not necessarily 50,000)
        # 50,000 is the CAPACITY reserved; actual usage depends on message size
        assert embed_result.success
        assert embed_result.bits_embedded > 0
        assert embed_result.bits_embedded <= 50000

        pipeline         = DecryptPipeline(key=embed_result.key)
        recovered, stats = pipeline.run(
            embed_result.embedded_residuals,
            embed_result.carrier_indices,
        )
        assert stats["success"], f"Decrypt failed: {stats.get('error')}"
        assert recovered == message
        print(f"    bits_embedded={embed_result.bits_embedded}, "
            f"message_len={len(message)}, recovered_len={len(recovered)}")


if __name__ == "__main__":
    import sys
    classes = [TestResidualExtractor(), TestFullPipelineWithMockModel()]
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