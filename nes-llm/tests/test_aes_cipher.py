from src.crypto.aes_cipher import (
    AESCipher
)

message = (
    "In this case, ✌️ all the students of group will participate in presentation. Group members may co-ordinate among themselves and distribute their part of presentation accordingly"
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