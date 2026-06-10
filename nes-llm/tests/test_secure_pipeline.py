from src.crypto.aes_cipher import AESCipher
from src.embedding.payload_encoder import PayloadEncoder
from src.embedding.keyed_residual_embedder import (
    KeyedResidualEmbedder
)

import torch


def main():

    original_message = (
        "In this case, ✌️ all the students of group will participate in presentation. Group members may co-ordinate among themselves and distribute their part of presentation accordingly"
    )

    encryption_password = (
        "mehar123"
    )

    carrier_key = (
        "carrier_secret"
    )

    print("\nORIGINAL MESSAGE")
    print(original_message)

    ciphertext = (
        AESCipher.encrypt(
            original_message,
            encryption_password
        )
    )

    print("\nCIPHERTEXT")
    print(ciphertext)

    ciphertext_string = (
        ciphertext.decode()
    )

    bits = (
        PayloadEncoder.text_to_bits(
            ciphertext_string
        )
    )

    print("\nTOTAL BITS")
    print(len(bits))

    residual_tensor = torch.randn(
        1000000
    )

    embedded_tensor = (
        KeyedResidualEmbedder.embed_bits(
            residual_tensor,
            bits,
            carrier_key
        )
    )

    recovered_bits = (
        KeyedResidualEmbedder.extract_bits(
            embedded_tensor,
            carrier_key,
            len(bits)
        )
    )

    recovered_ciphertext_string = (
        PayloadEncoder.bits_to_text(
            recovered_bits
        )
    )

    recovered_ciphertext = (
        recovered_ciphertext_string.encode()
    )

    recovered_message = (
        AESCipher.decrypt(
            recovered_ciphertext,
            encryption_password
        )
    )

    print("\nRECOVERED MESSAGE")
    print(recovered_message)

    print("\nSUCCESS")
    print(
        recovered_message
        ==
        original_message
    )


if __name__ == "__main__":
    main()