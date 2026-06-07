class PayloadEncoder:

    @staticmethod
    def text_to_bits(text: str):

        bits = []

        for char in text:

            binary = format(
                ord(char),
                "08b"
            )

            bits.extend(
                [int(bit) for bit in binary]
            )

        return bits

    @staticmethod
    def bits_to_text(bits):

        chars = []

        for i in range(
            0,
            len(bits),
            8
        ):

            byte = bits[i:i+8]

            if len(byte) < 8:
                break

            value = int(
                "".join(map(str, byte)),
                2
            )

            chars.append(chr(value))

        return "".join(chars)


def main():

    message = "HELLO NES"

    bits = PayloadEncoder.text_to_bits(
        message
    )

    print("\nOriginal Message:")
    print(message)

    print("\nBit Length:")
    print(len(bits))

    print("\nFirst 64 Bits:")
    print(bits[:64])

    recovered = (
        PayloadEncoder.bits_to_text(
            bits
        )
    )

    print("\nRecovered:")
    print(recovered)


if __name__ == "__main__":
    main()