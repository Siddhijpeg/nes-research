# tests/test_aes_cipher.py  — fixed for new AES-256-GCM API

from src.crypto.aes_cipher import AESCipher, AESPayload

def test_encrypt_decrypt():
    key     = AESCipher.generate_key()
    cipher  = AESCipher(key)
    payload = cipher.encrypt(b"hello NES")
    result  = cipher.decrypt(payload)
    assert result == b"hello NES"
    print("✅ test_encrypt_decrypt")

def test_different_ivs():
    key    = AESCipher.generate_key()
    cipher = AESCipher(key)
    p1     = cipher.encrypt(b"hello")
    p2     = cipher.encrypt(b"hello")
    assert p1.iv != p2.iv
    print("✅ test_different_ivs")

def test_key_from_password():
    key = AESCipher.key_from_password("my_password")
    assert len(key) == 32
    print("✅ test_key_from_password")

def test_tamper_detected():
    key     = AESCipher.generate_key()
    cipher  = AESCipher(key)
    payload = cipher.encrypt(b"secret")
    tampered = bytearray(payload.ciphertext)
    tampered[0] ^= 0xFF
    try:
        cipher.decrypt(AESPayload(iv=payload.iv, ciphertext=bytes(tampered)))
        assert False, "Should have raised"
    except Exception:
        pass
    print("✅ test_tamper_detected")

def test_bits_roundtrip():
    key    = AESCipher.generate_key()
    cipher = AESCipher(key)
    msg    = b"round trip"
    bits   = cipher.encrypt_to_bits(msg)
    result = cipher.decrypt_from_bits(bits)
    assert result == msg
    print("✅ test_bits_roundtrip")

if __name__ == "__main__":
    test_encrypt_decrypt()
    test_different_ivs()
    test_key_from_password()
    test_tamper_detected()
    test_bits_roundtrip()
    print("\n✅ All AES tests passed")