"""Tests for Phase 1 core infrastructure."""

import torch
from src.core import NESConfig, EmbeddingConfig, CryptoKey, EmbeddingResult
from src.embedding.payload_encoder   import PayloadEncoder
from src.embedding.sign_strategy_v2  import SignEmbeddingStrategy
from src.extraction.sign_extractor   import SignExtractor


class TestCoreTypes:

    def test_embedding_config_defaults(self):
        config = EmbeddingConfig(total_payload_bits=1000)
        assert config.total_payload_bits == 1000
        assert config.embedding_strategy == 'sign'
        assert config.use_qaci == True
        assert config.alpha == 0.25

    def test_crypto_key_size(self):
        key = CryptoKey(key=b'0' * 32, key_id="k1", model_id="llama")
        assert key.key_size == 256

    def test_embedding_result_fields(self):
        r = EmbeddingResult(
            success=True,
            embedded_weights={0: torch.randn(100)},
            carrier_indices={0: [1, 2, 3]},
            layer_allocation={0: 3},
            bits_embedded=3, total_bits=10, efficiency=0.3,
        )
        assert r.success
        assert r.efficiency == 0.3


class TestNESConfig:

    def test_default_config(self):
        c = NESConfig()
        assert c.total_payload_bits == 50000

    def test_llama3_config(self):
        c = NESConfig.for_llama3_8b()
        assert "Llama" in c.model.model_id
        assert c.model.num_layers == 32

    def test_testing_config(self):
        c = NESConfig.for_testing()
        assert c.total_payload_bits == 10000


class TestPayloadEncoding:

    def test_roundtrip(self):
        msg  = "Hello, NES!"
        bits = PayloadEncoder.text_to_bits(msg)
        assert PayloadEncoder.bits_to_text(bits) == msg

    def test_bits_are_binary(self):
        bits = PayloadEncoder.text_to_bits("test")
        assert all(b in (0, 1) for b in bits)


class TestSignStrategy:

    def test_bit1_positive(self):
        config   = EmbeddingConfig(total_payload_bits=10, embedding_strategy='sign')
        strategy = SignEmbeddingStrategy(config)
        assert strategy.get_bit_for_residual(0.05, 1) > 0

    def test_bit0_negative(self):
        config   = EmbeddingConfig(total_payload_bits=10, embedding_strategy='sign')
        strategy = SignEmbeddingStrategy(config)
        assert strategy.get_bit_for_residual(0.05, 0) < 0

    def test_extractor_positive_gives_1(self):
        assert SignExtractor().get_bit_from_residual(0.05) == 1

    def test_extractor_negative_gives_0(self):
        assert SignExtractor().get_bit_from_residual(-0.05) == 0

    def test_full_cycle_ber_zero(self):
        config   = EmbeddingConfig(total_payload_bits=100, embedding_strategy='sign')
        embedder = SignEmbeddingStrategy(config)
        extractor = SignExtractor()

        residuals = {0: torch.randn(10000) * 0.05}
        bits      = [i % 2 for i in range(100)]
        indices   = {0: list(range(100))}

        result    = embedder.embed(residuals, bits, indices)
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)
        ber       = extractor.calculate_ber(bits, recovered)

        assert ber == 0.0

    def test_noise_robustness(self):
        config    = EmbeddingConfig(total_payload_bits=100, embedding_strategy='sign')
        embedder  = SignEmbeddingStrategy(config)
        extractor = SignExtractor()

        residuals = {0: torch.randn(10000) * 0.05}
        bits      = [i % 2 for i in range(100)]
        indices   = {0: list(range(100))}

        result    = embedder.embed(residuals, bits, indices)
        noisy     = {0: result.embedded_weights[0] + torch.randn(10000) * 0.001}
        recovered = extractor.extract(noisy, result.carrier_indices)
        ber       = extractor.calculate_ber(bits, recovered)

        assert ber < 0.1


class TestInterfaces:

    def test_sign_strategy_is_embedder(self):
        from src.core import Embedder
        config = EmbeddingConfig(total_payload_bits=10, embedding_strategy='sign')
        assert isinstance(SignEmbeddingStrategy(config), Embedder)

    def test_sign_extractor_is_extractor(self):
        from src.core import Extractor
        assert isinstance(SignExtractor(), Extractor)


if __name__ == "__main__":
    import sys
    classes = [TestCoreTypes(), TestNESConfig(), TestPayloadEncoding(),
               TestSignStrategy(), TestInterfaces()]
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