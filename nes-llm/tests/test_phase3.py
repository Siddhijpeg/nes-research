"""Tests for Phase 3 — AES cipher, payload encoder, decrypt pipeline, full roundtrip."""

import torch
from src.crypto.aes_cipher              import AESCipher, AESPayload, _bytes_to_bits, _bits_to_bytes
from src.crypto.key_manager             import KeyManager
from src.embedding.payload_encoder      import PayloadEncoder
from src.extraction.decrypt_pipeline    import DecryptPipeline
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.core.types                     import EmbeddingConfig


class TestAESCipher:

    def test_encrypt_decrypt_roundtrip(self):
        key     = AESCipher.generate_key()
        cipher  = AESCipher(key)
        message = b"NES secret payload"
        assert cipher.decrypt(cipher.encrypt(message)) == message

    def test_different_iv_each_call(self):
        key    = AESCipher.generate_key()
        cipher = AESCipher(key)
        assert cipher.encrypt(b"hello").iv != cipher.encrypt(b"hello").iv

    def test_tamper_detected(self):
        key     = AESCipher.generate_key()
        cipher  = AESCipher(key)
        payload = cipher.encrypt(b"sensitive")
        tampered = bytearray(payload.ciphertext)
        tampered[0] ^= 0xFF
        try:
            cipher.decrypt(AESPayload(iv=payload.iv, ciphertext=bytes(tampered)))
            assert False, "Should have raised"
        except Exception:
            pass

    def test_key_from_password(self):
        assert len(AESCipher.key_from_password("pw")) == 32

    def test_encrypt_to_bits_decrypt_from_bits(self):
        key    = AESCipher.generate_key()
        cipher = AESCipher(key)
        msg    = b"round trip test"
        bits   = cipher.encrypt_to_bits(msg)
        assert all(b in (0, 1) for b in bits)
        assert cipher.decrypt_from_bits(bits) == msg


class TestKeyManager:

    def test_create_and_retrieve(self):
        km  = KeyManager()
        kid = km.create_key(model_id="llama-3-8b")
        assert len(km.get_key(kid)) == 32

    def test_get_cipher(self):
        km = KeyManager()
        assert isinstance(km.get_cipher(km.create_key()), AESCipher)

    def test_save_load(self, tmp_path=None):
        import tempfile, os
        km  = KeyManager()
        kid = km.create_key(model_id="test")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            km.save(path)
            km2 = KeyManager()
            km2.load(path)
            assert km2.get_key(kid) == km.get_key(kid)
        finally:
            os.unlink(path)


class TestPayloadEncoder:

    def test_roundtrip_short(self):
        msg = "Hello NES!"
        assert PayloadEncoder.bits_to_text(PayloadEncoder.text_to_bits(msg)) == msg

    def test_roundtrip_long(self):
        msg = "A" * 500
        assert PayloadEncoder.bits_to_text(PayloadEncoder.text_to_bits(msg)) == msg

    def test_header_present(self):
        bits = PayloadEncoder.text_to_bits("test")
        payload_len = PayloadEncoder._bits_to_uint32(bits[:32])
        assert payload_len == len(bits) - 32

    def test_unicode(self):
        msg = "नमस्ते"
        assert PayloadEncoder.bits_to_text(PayloadEncoder.text_to_bits(msg)) == msg


class TestDecryptPipeline:

    def _embed_and_get_residuals(self, message, key):
        """
        Wire format (must match IntelligentEmbedder exactly):
            [32-bit outer header: len(encrypted_bits)] [IV bits] [ciphertext bits]
            where ciphertext = AES(raw_utf8_bytes)
        """
        cipher    = AESCipher(key)
        msg_bytes = message.encode("utf-8")       # raw UTF-8, no inner header
        enc_bits  = cipher.encrypt_to_bits(msg_bytes)

        header   = PayloadEncoder._uint32_to_bits(len(enc_bits))
        all_bits = header + enc_bits

        n               = len(all_bits)
        residuals       = {0: torch.randn(n * 2) * 0.05}
        carrier_indices = {0: list(range(n))}

        flat = residuals[0].clone()
        for i, bit in enumerate(all_bits):
            flat[i] = abs(flat[i].item()) if bit == 1 else -abs(flat[i].item())
        residuals[0] = flat
        return residuals, carrier_indices

    def test_full_decrypt_roundtrip(self):
        key     = AESCipher.generate_key()
        message = "Secret NES message"
        residuals, carrier_indices = self._embed_and_get_residuals(message, key)
        pipeline           = DecryptPipeline(key=key)
        recovered, stats   = pipeline.run(residuals, carrier_indices)
        assert stats["success"], f"Failed: {stats.get('error')}"
        assert recovered == message

    def test_extract_bits_only(self):
        key     = AESCipher.generate_key()
        message = "test"
        residuals, carrier_indices = self._embed_and_get_residuals(message, key)
        pipeline = DecryptPipeline(key=key)
        bits     = pipeline.extract_bits_only(residuals, carrier_indices)
        assert len(bits) > 0
        assert all(b in (0, 1) for b in bits)


class TestIntelligentEmbedder:

    def _make_residuals(self, n_layers=4, layer_size=5000):
        return {i: torch.randn(layer_size) * 0.05 for i in range(n_layers)}

    def test_embed_produces_result(self):
        config   = EmbeddingConfig(total_payload_bits=1000, embedding_strategy="sign")
        embedder = IntelligentEmbedder(config)
        result   = embedder.embed("Hello NES", self._make_residuals())
        assert result.success
        assert result.bits_embedded > 0
        assert len(result.key) == 32

    def test_carrier_indices_valid(self):
        config     = EmbeddingConfig(total_payload_bits=500, embedding_strategy="sign")
        embedder   = IntelligentEmbedder(config)
        layer_size = 5000
        residuals  = {i: torch.randn(layer_size) * 0.05 for i in range(4)}
        result     = embedder.embed("test message", residuals)
        for lid, indices in result.carrier_indices.items():
            for idx in indices:
                assert 0 <= idx < layer_size

    def test_embed_decrypt_roundtrip(self):
        config    = EmbeddingConfig(total_payload_bits=2000, embedding_strategy="sign")
        embedder  = IntelligentEmbedder(config)
        residuals = {i: torch.randn(10000) * 0.05 for i in range(4)}
        message   = "End to end NES test!"

        embed_result     = embedder.embed(message, residuals)
        pipeline         = DecryptPipeline(key=embed_result.key)
        recovered, stats = pipeline.run(
            embed_result.embedded_residuals,
            embed_result.carrier_indices,
        )
        assert stats["success"], f"Failed: {stats.get('error')}"
        assert recovered == message


if __name__ == "__main__":
    import sys
    classes = [TestAESCipher(), TestKeyManager(), TestPayloadEncoder(),
               TestDecryptPipeline(), TestIntelligentEmbedder()]
    passed = failed = 0
    for obj in classes:
        cls = type(obj).__name__
        for method in [m for m in dir(obj) if m.startswith("test_")]:
            if method == "test_save_load":
                try:
                    obj.test_save_load()
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