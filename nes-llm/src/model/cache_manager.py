"""
Model Tensor Cache Manager.

Stores expensive preprocessing artifacts produced from an NF4/FP16
model pair:

    - quantization residuals
    - FP16 reference weights
    - dequantized NF4 weights

Design goals:
    - Per-layer storage
    - Atomic writes
    - Manifest-based validation
    - Cache versioning
    - CPU-only serialized tensors
    - Safe cache invalidation when preprocessing configuration changes
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import torch


CACHE_VERSION = "1.0"


class ModelTensorCache:
    """
    Persistent cache for model preprocessing artifacts.

    Cache layout:

        cache/models/<model_name>/
            metadata.json
            manifest.json
            residuals/
                layer_000.pt
                ...
            fp16/
                layer_000.pt
                ...
            nf4_dequantized/
                layer_000.pt
                ...

    All tensors are serialized on CPU.
    """

    def __init__(
        self,
        model_id: str,
        cache_root: str = "cache/models",
        quantization_type: str = "nf4",
        use_double_quant: bool = True,
        compute_dtype: str = "float16",
    ):
        self.model_id = model_id
        self.quantization_type = quantization_type
        self.use_double_quant = use_double_quant
        self.compute_dtype = compute_dtype

        safe_model_name = self._sanitize_model_id(model_id)

        self.root = (
            Path(cache_root)
            / safe_model_name
        )

        self.residual_dir = self.root / "residuals"
        self.fp16_dir = self.root / "fp16"
        self.nf4_dir = self.root / "nf4_dequantized"

        self.metadata_path = self.root / "metadata.json"
        self.manifest_path = self.root / "manifest.json"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_model_id(model_id: str) -> str:
        """
        Convert a HuggingFace model ID into a filesystem-safe name.
        """
        return (
            model_id
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

    def _layer_path(
        self,
        directory: Path,
        layer_id: int,
    ) -> Path:
        return directory / f"layer_{layer_id:03d}.pt"

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _configuration(self) -> dict:
        return {
            "cache_version": CACHE_VERSION,
            "model_id": self.model_id,
            "quantization_type": self.quantization_type,
            "use_double_quant": self.use_double_quant,
            "compute_dtype": self.compute_dtype,
        }

    def initialize(self) -> None:
        """
        Create cache directories and metadata if necessary.
        """
        self.residual_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.fp16_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.nf4_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.metadata_path.exists():
            self._atomic_json_write(
                self.metadata_path,
                self._configuration(),
            )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}

        try:
            with self.manifest_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                return json.load(handle)

        except (OSError, json.JSONDecodeError):
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        self._atomic_json_write(
            self.manifest_path,
            manifest,
        )

    @staticmethod
    def _tensor_checksum(tensor: torch.Tensor) -> str:
        """
        Generate a deterministic checksum for a tensor.
        """
        cpu_tensor = (
            tensor.detach()
            .cpu()
            .contiguous()
        )

        digest = hashlib.sha256(
            cpu_tensor.numpy().tobytes()
        )

        return digest.hexdigest()

    # ------------------------------------------------------------------
    # Atomic serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_torch_save(
        tensor: torch.Tensor,
        destination: Path,
    ) -> None:
        """
        Atomically save a tensor.

        The temporary file is created in the same directory so that
        os.replace() remains atomic on the same filesystem.
        """
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fd, temporary = tempfile.mkstemp(
            dir=destination.parent,
            suffix=".tmp",
        )

        os.close(fd)

        temporary_path = Path(temporary)

        try:
            torch.save(
                tensor.detach().cpu(),
                temporary_path,
            )

            os.replace(
                temporary_path,
                destination,
            )

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _atomic_json_write(
        destination: Path,
        data: dict,
    ) -> None:
        """
        Atomically write JSON metadata.
        """
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fd, temporary = tempfile.mkstemp(
            dir=destination.parent,
            suffix=".tmp",
            text=True,
        )

        os.close(fd)

        temporary_path = Path(temporary)

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    data,
                    handle,
                    indent=2,
                    sort_keys=True,
                )

            os.replace(
                temporary_path,
                destination,
            )

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_layer(
        self,
        layer_id: int,
        residual: torch.Tensor,
        fp16_weight: torch.Tensor,
        nf4_dequantized: torch.Tensor,
    ) -> None:
        """
        Save all preprocessing artifacts for one layer.
        """

        self.initialize()

        tensors = {
            "residuals": (
                residual,
                self.residual_dir,
            ),
            "fp16": (
                fp16_weight,
                self.fp16_dir,
            ),
            "nf4_dequantized": (
                nf4_dequantized,
                self.nf4_dir,
            ),
        }

        manifest = self._load_manifest()

        layer_key = str(layer_id)

        manifest[layer_key] = {
            "residual": {
                "shape": list(residual.shape),
                "dtype": str(residual.dtype),
                "numel": residual.numel(),
                "checksum": self._tensor_checksum(residual),
            },
            "fp16": {
                "shape": list(fp16_weight.shape),
                "dtype": str(fp16_weight.dtype),
                "numel": fp16_weight.numel(),
                "checksum": self._tensor_checksum(fp16_weight),
            },
            "nf4_dequantized": {
                "shape": list(nf4_dequantized.shape),
                "dtype": str(nf4_dequantized.dtype),
                "numel": nf4_dequantized.numel(),
                "checksum": self._tensor_checksum(
                    nf4_dequantized
                ),
            },
        }

        # Complexity:
        # O(number of parameters in this layer) for serialization.
        for name, (tensor, directory) in tensors.items():

            path = self._layer_path(
                directory,
                layer_id,
            )

            self._atomic_torch_save(
                tensor,
                path,
            )

        self._save_manifest(manifest)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def has_layer(self, layer_id: int) -> bool:
        """
        Check whether all three artifacts for a layer exist.
        """
        return (
            self._layer_path(
                self.residual_dir,
                layer_id,
            ).exists()
            and
            self._layer_path(
                self.fp16_dir,
                layer_id,
            ).exists()
            and
            self._layer_path(
                self.nf4_dir,
                layer_id,
            ).exists()
        )

    def validate_layer(self, layer_id: int) -> bool:
        """
        Validate cached files against the manifest.

        Returns False instead of trusting corrupted/incomplete data.
        """

        if not self.has_layer(layer_id):
            return False

        manifest = self._load_manifest()
        entry = manifest.get(str(layer_id))

        if entry is None:
            return False

        files = {
            "residual": self._layer_path(
                self.residual_dir,
                layer_id,
            ),
            "fp16": self._layer_path(
                self.fp16_dir,
                layer_id,
            ),
            "nf4_dequantized": self._layer_path(
                self.nf4_dir,
                layer_id,
            ),
        }

        try:
            for name, path in files.items():

                tensor = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                )

                expected = entry[name]

                if list(tensor.shape) != expected["shape"]:
                    return False

                if tensor.numel() != expected["numel"]:
                    return False

                if self._tensor_checksum(tensor) != expected["checksum"]:
                    return False

        except (
            OSError,
            RuntimeError,
            ValueError,
        ):
            return False

        return True

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_layer(
        self,
        layer_id: int,
        device: Optional[torch.device] = None,
    ):
        """
        Load a validated layer from disk.

        Returns:
            residual, fp16_weight, nf4_dequantized
        """

        if not self.validate_layer(layer_id):
            raise RuntimeError(
                f"Cache validation failed for "
                f"{self.model_id}, layer {layer_id}"
            )

        residual = torch.load(
            self._layer_path(
                self.residual_dir,
                layer_id,
            ),
            map_location="cpu",
            weights_only=True,
        )

        fp16_weight = torch.load(
            self._layer_path(
                self.fp16_dir,
                layer_id,
            ),
            map_location="cpu",
            weights_only=True,
        )

        nf4_dequantized = torch.load(
            self._layer_path(
                self.nf4_dir,
                layer_id,
            ),
            map_location="cpu",
            weights_only=True,
        )

        if device is not None:
            residual = residual.to(device)
            fp16_weight = fp16_weight.to(device)
            nf4_dequantized = nf4_dequantized.to(device)

        return (
            residual,
            fp16_weight,
            nf4_dequantized,
        )

    # ------------------------------------------------------------------
    # Cache status
    # ------------------------------------------------------------------

    def cached_layers(self) -> list[int]:
        """
        Return all layers that pass validation.
        """
        manifest = self._load_manifest()

        valid = []

        for layer_id in manifest:
            layer = int(layer_id)

            if self.validate_layer(layer):
                valid.append(layer)

        return sorted(valid)

    def clear(self) -> None:
        """
        Delete this model's cache.
        """
        import shutil

        if self.root.exists():
            shutil.rmtree(self.root)