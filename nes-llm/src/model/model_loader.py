"""
Model loader — loads a quantized LLM in NF4 via BitsAndBytes.

Usage:
    loader = ModelLoader()
    model, tokenizer = loader.load("meta-llama/Llama-3-8B")
"""

from typing import Optional, Tuple
import torch


class ModelLoader:
    """
    Loads a HuggingFace causal LM in NF4 quantization (BitsAndBytes).

    Supports:
        - meta-llama/Llama-3-8B
        - mistralai/Mistral-7B-v0.1
        - Qwen/Qwen2.5-7B
        - Any BitsAndBytes-compatible model
    """

    def __init__(
        self,
        device:             str  = "mps:0",
        use_double_quant:   bool = True,
        compute_dtype:      str  = "float16",
    ):
        self.device           = device
        self.use_double_quant = use_double_quant
        self.compute_dtype    = compute_dtype

    def load(
        self,
        model_id:   str,
        cache_dir:  Optional[str] = None,
        token:      Optional[str] = None,
    ) -> Tuple:
        """
        Load model and tokenizer in NF4.

        Args:
            model_id:  HuggingFace model ID.
            cache_dir: Local cache directory.
            token:     HF token for gated models (Llama requires this).

        Returns:
            (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=              True,
            bnb_4bit_quant_type=       "nf4",
            bnb_4bit_use_double_quant= self.use_double_quant,
            bnb_4bit_compute_dtype=    getattr(torch, self.compute_dtype),
        )

        print(f"[ModelLoader] Loading {model_id} in NF4...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=         "auto",
            cache_dir=          cache_dir,
            token=              token,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            token=token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[ModelLoader] Loaded. Layers: {model.config.num_hidden_layers}")
        return model, tokenizer

    def get_layer_names(self, model) -> list:
        """Return all quantized linear layer names."""
        names = []
        for name, module in model.named_modules():
            if hasattr(module, "weight") and hasattr(module.weight, "quant_type"):
                names.append(name)
        return names