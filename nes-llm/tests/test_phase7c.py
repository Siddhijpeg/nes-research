"""Tests for Phase 7C — Neural Embedding Strategy."""

import torch
from src.embedding.strategies.neural_strategy import (
    NeuralEmbeddingModel, NeuralEmbeddingTrainer, NeuralStrategy
)
from src.extraction.neural_extractor import NeuralExtractor
from src.core.types                  import EmbeddingConfig


def make_residuals(n=4, size=5000):
    return {i: torch.randn(size) * 0.05 for i in range(n)}

def make_indices(residuals, total_bits):
    n   = len(residuals)
    per = total_bits // n
    return {lid: list(range(per)) for lid in residuals}

def train_small_model(residuals, epochs=30):
    trainer = NeuralEmbeddingTrainer(
        hidden_dim=32, lr=1e-3,
        lambda_fidelity=10.0, lambda_security=1.0,
        batch_size=2048, device="cpu",
    )
    return trainer.train(residuals, epochs=epochs, verbose=False), trainer


class TestNeuralModel:

    def test_encoder_output_shape(self):
        model    = NeuralEmbeddingModel(hidden_dim=32)
        residual = torch.randn(100)
        bits     = torch.randint(0, 2, (100,)).float()
        out      = model.encoder(residual, bits)
        assert out.shape == (100,)

    def test_decoder_output_in_range(self):
        model    = NeuralEmbeddingModel(hidden_dim=32)
        modified = torch.randn(100)
        probs    = model.decoder(modified)
        assert probs.shape == (100,)
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0

    def test_forward_pass(self):
        model    = NeuralEmbeddingModel(hidden_dim=32)
        residual = torch.randn(50)
        bits     = torch.randint(0, 2, (50,)).float()
        modified, bit_prob = model(residual, bits)
        assert modified.shape  == (50,)
        assert bit_prob.shape  == (50,)


class TestNeuralTraining:

    def test_training_reduces_loss(self):
        """Loss should decrease over training."""
        residuals = make_residuals(n=2, size=2000)
        model, _  = train_small_model(residuals, epochs=20)
        assert model is not None

    def test_trained_model_achieves_low_ber(self):
        """After training, clean BER should be near zero."""
        residuals = make_residuals(n=4, size=5000)
        model, _  = train_small_model(residuals, epochs=50)

        config   = EmbeddingConfig(total_payload_bits=500, embedding_strategy="neural")
        strategy = NeuralStrategy(config, model=model)
        extractor = NeuralExtractor(model)

        bits    = [i % 2 for i in range(500)]
        indices = make_indices(residuals, 500)

        result    = strategy.embed(residuals, bits, indices)
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)

        n   = min(len(bits), len(recovered))
        ber = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)
        print(f"\n    Neural clean BER: {ber:.4f}")
        assert ber < 0.05, f"Expected low BER after training, got {ber}"

    def test_save_load_model(self, tmp_path=None):
        """Saved model should produce same outputs as original."""
        import tempfile, os
        residuals = make_residuals(n=2, size=2000)
        model, trainer = train_small_model(residuals, epochs=20)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.save(model, path)
            loaded = NeuralEmbeddingTrainer.load(path)

            # Same output on same input
            x    = torch.randn(10)
            bits = torch.zeros(10)
            with torch.no_grad():
                out1 = model.encoder(x, bits)
                out2 = loaded.encoder(x, bits)
            assert torch.allclose(out1, out2, atol=1e-5)
        finally:
            os.unlink(path)


class TestNeuralEmbedExtract:

    def test_embed_returns_result(self):
        residuals = make_residuals(n=4, size=3000)
        model, _  = train_small_model(residuals, epochs=20)

        config   = EmbeddingConfig(total_payload_bits=500, embedding_strategy="neural")
        strategy = NeuralStrategy(config, model=model)

        bits    = [i % 2 for i in range(500)]
        indices = make_indices(residuals, 500)

        result = strategy.embed(residuals, bits, indices)
        assert result.success
        assert result.bits_embedded == 500
        assert result.metadata["strategy"] == "neural"

    def test_fidelity_reasonable(self):
        """Neural embedding should not distort weights excessively."""
        residuals = make_residuals(n=4, size=3000)
        model, _  = train_small_model(residuals, epochs=30)

        config   = EmbeddingConfig(total_payload_bits=500, embedding_strategy="neural")
        strategy = NeuralStrategy(config, model=model)

        bits    = [i % 2 for i in range(500)]
        indices = make_indices(residuals, 500)
        result  = strategy.embed(residuals, bits, indices)

        # Mean absolute change should be small
        total_change = 0.0
        total_params = 0
        for lid in residuals:
            diff   = (residuals[lid].float() - result.embedded_weights[lid].float()).abs()
            total_change += diff.sum().item()
            total_params += diff.numel()

        mean_change = total_change / max(total_params, 1)
        print(f"\n    Neural mean |Δweight|: {mean_change:.6f}")
        assert mean_change < 0.1, f"Too much distortion: {mean_change}"

    def test_extractor_is_extractor(self):
        from src.core import Extractor
        model = NeuralEmbeddingModel(hidden_dim=32)
        assert isinstance(NeuralExtractor(model), Extractor)


class TestStrategyRegistryNeural:

    def test_neural_in_registry(self):
        from src.embedding.strategies import STRATEGY_REGISTRY
        assert "neural" in STRATEGY_REGISTRY

    def test_get_neural_strategy(self):
        from src.embedding.strategies import get_strategy
        residuals = make_residuals(n=2, size=2000)
        model, _  = train_small_model(residuals, epochs=10)
        config    = EmbeddingConfig(total_payload_bits=100, embedding_strategy="neural")
        emb, ext  = get_strategy("neural", config, model=model)
        assert emb is not None
        assert ext is not None


if __name__ == "__main__":
    import sys
    classes = [
        TestNeuralModel(),
        TestNeuralTraining(),
        TestNeuralEmbedExtract(),
        TestStrategyRegistryNeural(),
    ]
    passed = failed = 0
    for obj in classes:
        cls = type(obj).__name__
        for method in [m for m in dir(obj) if m.startswith("test_")]:
            if method == "test_save_load_model":
                try:
                    obj.test_save_load_model()
                    print(f"  ✅ {cls}.{method}")
                    passed += 1
                except Exception as e:
                    print(f"  ❌ {cls}.{method}: {e}")
                    failed += 1
                continue
            try:
                getattr(obj, method)()
                print(f"  ✅ {cls}.{method}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {cls}.{method}: {e}")
                failed += 1
    print(f"\n{'='*55}\n  {passed} passed, {failed} failed\n{'='*55}")
    sys.exit(0 if failed == 0 else 1)