"""Tests for Phase 6 — CLI and production packaging."""

import json
import os
import sys
import tempfile
import torch

from src.crypto.key_manager          import KeyManager
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.extraction.decrypt_pipeline import DecryptPipeline
from src.core.types                  import EmbeddingConfig
from src.core.config                 import NESConfig


class TestCLICommands:

    def _make_residuals(self, n=4, size=5000):
        return {i: torch.randn(size) * 0.05 for i in range(n)}

    def test_embed_extract_via_key_manager(self):
        """Full pipeline using KeyManager for key storage."""
        config   = EmbeddingConfig(total_payload_bits=1000, embedding_strategy="sign")
        embedder = IntelligentEmbedder(config)
        residuals = self._make_residuals()
        message  = "CLI integration test"

        result = embedder.embed(message, residuals)
        assert result.success

        km  = KeyManager()
        kid = km.add_key(result.key, model_id="test-model")
        assert km.get_key(kid) == result.key

        pipeline = DecryptPipeline(key=km.get_key(kid))
        recovered, stats = pipeline.run(
            result.embedded_residuals,
            result.carrier_indices,
        )
        assert stats["success"]
        assert recovered == message

    def test_key_manager_save_load_roundtrip(self):
        """Save keys to disk and reload them."""
        km  = KeyManager()
        kid = km.create_key(model_id="llama-3-8b")
        key = km.get_key(kid)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            km.save(path)
            km2 = KeyManager()
            km2.load(path)
            assert km2.get_key(kid) == key
        finally:
            os.unlink(path)

    def test_config_json_roundtrip(self):
        """Save and reload NESConfig as JSON."""
        config = NESConfig.for_llama3_8b()
        config.total_payload_bits = 42000

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name

        try:
            config.to_json(path)
            loaded = NESConfig.from_json(path)
            assert loaded.total_payload_bits == 42000
            assert loaded.model.model_id == "meta-llama/Llama-3-8B"
        finally:
            os.unlink(path)

    def test_config_for_testing_preset(self):
        config = NESConfig.for_testing()
        assert config.total_payload_bits == 10000

    def test_config_for_mistral(self):
        config = NESConfig.for_mistral_7b()
        assert "Mistral" in config.model.model_id


class TestEndToEndPipeline:
    """Full embed → key save → key load → extract roundtrip."""

    def test_full_roundtrip_with_file_keys(self):
        residuals = {i: torch.randn(8000) * 0.05 for i in range(4)}
        message   = "End to end phase 6 test"
        config    = EmbeddingConfig(total_payload_bits=2000, embedding_strategy="sign")

        # Embed
        embedder     = IntelligentEmbedder(config)
        embed_result = embedder.embed(message, residuals)
        assert embed_result.success

        # Save key + carrier map
        km  = KeyManager()
        kid = km.add_key(embed_result.key)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            key_path = f.name
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            carrier_path = f.name

        try:
            km.save(key_path)
            with open(carrier_path, "w") as f:
                json.dump(
                    {str(lid): idx
                     for lid, idx in embed_result.carrier_indices.items()}, f
                )

            # Load and extract
            km2 = KeyManager()
            km2.load(key_path)
            key = km2.get_key(kid)

            with open(carrier_path) as f:
                raw = json.load(f)
            carrier_indices = {int(k): v for k, v in raw.items()}

            pipeline = DecryptPipeline(key=key)
            recovered, stats = pipeline.run(
                embed_result.embedded_residuals,
                carrier_indices,
            )
            assert stats["success"], f"Failed: {stats}"
            assert recovered == message

        finally:
            os.unlink(key_path)
            os.unlink(carrier_path)


if __name__ == "__main__":
    classes = [TestCLICommands(), TestEndToEndPipeline()]
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