"""
NES Real Model Pipeline — full end-to-end on a quantized LLM.

Usage:
    pipeline = NESRealPipeline(model_id="meta-llama/Llama-3-8B")
    result   = pipeline.embed("secret message", hf_token="hf_...")
    pipeline.save("embedded_model/")

    # To extract later:
    pipeline2 = NESRealPipeline(model_id="meta-llama/Llama-3-8B")
    message   = pipeline2.extract("embedded_model/", key, carrier_map)
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch

from src.model.model_loader       import ModelLoader, LLAMA_TARGET_MODULES
from src.model.residual_extractor import ResidualExtractor
from src.model.weight_patcher     import WeightPatcher
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.extraction.decrypt_pipeline    import DecryptPipeline
from src.crypto.key_manager             import KeyManager
from src.core.types                     import EmbeddingConfig


class NESRealPipeline:
    """
    Full NES pipeline operating on a real quantized LLM.

    Steps (embed):
        1. Load model in NF4 (BitsAndBytes)
        2. Extract dequantized residuals from target layers
        3. Run IntelligentEmbedder (QACI + sign embedding)
        4. Patch embedded weights back into model
        5. Save model + key + carrier map

    Steps (extract):
        1. Load embedded model in NF4
        2. Extract residuals from same target layers
        3. Run DecryptPipeline (sign extraction + AES decrypt)
        4. Return plaintext message
    """

    def __init__(
        self,
        model_id:       str,
        target_modules: List[str] = None,
        layer_range:    Optional[Tuple[int, int]] = None,
        payload_bits:   int = 50000,
        cache_dir:      Optional[str] = None,
    ):
        self.model_id       = model_id
        self.target_modules = target_modules or LLAMA_TARGET_MODULES
        self.layer_range    = layer_range
        self.payload_bits   = payload_bits
        self.cache_dir      = cache_dir

        self.loader    = ModelLoader()
        self.extractor = ResidualExtractor(
            target_modules=self.target_modules,
            layer_range=   self.layer_range,
        )
        self.patcher   = WeightPatcher()
        self.km        = KeyManager()

        self._model     = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    def embed(
        self,
        message:    str,
        hf_token:   Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Embed message into model and optionally save.

        Returns:
            {
                "key":            bytes,
                "key_id":         str,
                "carrier_indices":{layer_id: [indices]},
                "module_names":   {layer_id: str},
                "bits_embedded":  int,
                "output_dir":     str or None,
            }
        """
        # Load model
        model, tokenizer = self.loader.load(
            self.model_id,
            cache_dir=self.cache_dir,
            token=hf_token,
        )
        self._model     = model
        self._tokenizer = tokenizer

        # Extract residuals
        residuals, fp16_weights, module_names = self.extractor.extract(model)

        total_capacity = sum(t.numel() for t in residuals.values())
        print(f"[NESPipeline] Total carrier capacity: {total_capacity:,} positions")

        # Embed
        config   = EmbeddingConfig(
            total_payload_bits=min(self.payload_bits, total_capacity),
            embedding_strategy="sign",
        )
        embedder = IntelligentEmbedder(config)
        result   = embedder.embed(
            message,
            residuals,
            fp16_weights=     fp16_weights,
            quantized_weights=fp16_weights,
        )

        print(f"[NESPipeline] Embedded {result.bits_embedded:,} bits across "
              f"{sum(1 for b in result.layer_allocation.values() if b>0)} layers")

        # Patch model
        self.patcher.patch(model, result.embedded_residuals, module_names)

        # Store key
        kid = self.km.add_key(result.key, model_id=self.model_id)

        # Save
        if output_dir:
            self._save(model, tokenizer, kid, result.carrier_indices,
                       module_names, output_dir)

        return {
            "key":             result.key,
            "key_id":          kid,
            "carrier_indices": result.carrier_indices,
            "module_names":    module_names,
            "bits_embedded":   result.bits_embedded,
            "output_dir":      output_dir,
        }

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def extract(
        self,
        model_dir:       str,
        key:             bytes,
        carrier_indices: Dict[int, List[int]],
        hf_token:        Optional[str] = None,
    ) -> str:
        """
        Extract message from embedded model.

        Args:
            model_dir:       Path to saved embedded model.
            key:             32-byte AES key.
            carrier_indices: {layer_id: [flat_indices]}

        Returns:
            Recovered plaintext message.
        """
        model, tokenizer = self.loader.load(
            model_dir,
            cache_dir=self.cache_dir,
            token=hf_token,
        )

        residuals, _, _ = self.extractor.extract(model)
        pipeline        = DecryptPipeline(key=key)
        message, stats  = pipeline.run(residuals, carrier_indices)

        if not stats.get("success"):
            raise RuntimeError(f"Extraction failed: {stats.get('error')}")

        print(f"[NESPipeline] Extracted message ({len(message)} chars)")
        return message

    # ------------------------------------------------------------------
    # Save / Load helpers
    # ------------------------------------------------------------------

    def _save(
        self,
        model,
        tokenizer,
        key_id:          str,
        carrier_indices: Dict[int, List[int]],
        module_names:    Dict[int, str],
        output_dir:      str,
    ):
        os.makedirs(output_dir, exist_ok=True)

        # Save model + tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save key
        self.km.save(os.path.join(output_dir, "nes_keys.json"))

        # Save carrier map
        with open(os.path.join(output_dir, "nes_carriers.json"), "w") as f:
            json.dump({
                "key_id":          key_id,
                "carrier_indices": {str(k): v for k, v in carrier_indices.items()},
                "module_names":    {str(k): v for k, v in module_names.items()},
            }, f, indent=2)

        print(f"[NESPipeline] Saved to {output_dir}/")
        print(f"  Model     : {output_dir}/")
        print(f"  Keys      : {output_dir}/nes_keys.json")
        print(f"  Carriers  : {output_dir}/nes_carriers.json")