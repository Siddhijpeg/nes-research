from src.crypto.aes_cipher import (
    AESCipher
)

message = (
    "HELLO NES"
)

password = (
    "mehar123"
)

ciphertext = (
    AESCipher.encrypt(
        message,
        password
    )
)

print(
    "\nCiphertext:"
)

print(
    ciphertext
)

recovered = (
    AESCipher.decrypt(
        ciphertext,
        password
    )
)

print(
    "\nRecovered:"
)

print(
    recovered
)