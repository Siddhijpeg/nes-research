import torch


class ResidualEmbedder:

    @staticmethod
    def embed_bits(
        residual_tensor,
        bits
    ):

        modified = (
            residual_tensor
            .clone()
            .flatten()
        )

        capacity = len(modified)

        if len(bits) > capacity:

            raise ValueError(
                "Payload exceeds capacity"
            )

        for idx, bit in enumerate(bits):

            value = max(
                abs(modified[idx].item()),
                0.001
            )

            if bit == 1:

                modified[idx] = value

            else:

                modified[idx] = -value

        return modified

    @staticmethod
    def extract_bits(
        modified_tensor,
        num_bits
    ):

        flat = modified_tensor.flatten()

        bits = []

        for i in range(num_bits):

            if flat[i] >= 0:

                bits.append(1)

            else:

                bits.append(0)

        return bits


def main():

    residual = torch.tensor([
        0.001,
        -0.002,
        0.003,
        -0.004,
        0.005,
        -0.006,
        0.007,
        -0.008,
    ])

    payload = [
        1, 0, 1, 1,
        0, 0, 1, 0
    ]

    embedded = (
        ResidualEmbedder.embed_bits(
            residual,
            payload
        )
    )

    recovered = (
        ResidualEmbedder.extract_bits(
            embedded,
            len(payload)
        )
    )

    print("\nPayload:")
    print(payload)

    print("\nRecovered:")
    print(recovered)

    print("\nSuccess:")
    print(payload == recovered)


if __name__ == "__main__":
    main()