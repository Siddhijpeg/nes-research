"""
model_loader.py

Loads a quantized language model and tokenizer for the
Carrier Intelligence phase.
Project: Neural-Entropic Steganography (NES v2)
"""

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
import torch


class ModelLoader:
    """
    Handles loading of quantized LLMs.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quantization: bool = True,
    ):
        self.model_name = model_name
        self.quantization = quantization

    def load(self):
        """
        Loads tokenizer and model.

        Returns
        -------
        tokenizer
        model
        """

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.quantization:

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
            )

        else:

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        return tokenizer, model


if __name__ == "__main__":

    loader = ModelLoader()

    tokenizer, model = loader.load()

    print("=" * 60)
    print("Model Loaded Successfully")
    print(f"Model : {loader.model_name}")
    print(f"Quantized : {loader.quantization}")
    print("=" * 60)