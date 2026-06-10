from cryptography.fernet import Fernet
import base64
import hashlib


class AESCipher:

    @staticmethod
    def derive_key(password):

        digest = hashlib.sha256(
            password.encode()
        ).digest()

        return base64.urlsafe_b64encode(
            digest
        )

    @staticmethod
    def encrypt(
        plaintext,
        password
    ):

        key = AESCipher.derive_key(
            password
        )

        cipher = Fernet(key)

        ciphertext = cipher.encrypt(
            plaintext.encode()
        )

        return ciphertext

    @staticmethod
    def decrypt(
        ciphertext,
        password
    ):

        key = AESCipher.derive_key(
            password
        )

        cipher = Fernet(key)

        plaintext = cipher.decrypt(
            ciphertext
        )

        return plaintext.decode()