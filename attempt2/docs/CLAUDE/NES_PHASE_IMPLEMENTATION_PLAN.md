# NES - PHASE IMPLEMENTATION PLAN
## Step-by-Step Execution Guide
**Classification:** CONFIDENTIAL  
**Version:** 2.0  
**Last Updated:** 2026-07-18  

---

## QUICK START CHECKLIST

```
Phase 1 (Weeks 1-2): ☐ START HERE
├─ ☐ Code cleanup (day 1-2)
├─ ☐ Architecture setup (day 3-4)
├─ ☐ Core modules (day 5-7)
└─ ☐ Unit tests (day 8-10)

Phase 2 (Weeks 3-4): ☐ AFTER PHASE 1
├─ ☐ Layer profiling
├─ ☐ Feature extraction
├─ ☐ Quality scoring
└─ ☐ Carrier scheduling

Phase 3 (Weeks 5-6): ☐ AFTER PHASE 2
├─ ☐ Embedding strategies
├─ ☐ Encryption pipeline
└─ ☐ Extraction pipeline

Phase 4 (Weeks 7-9): ☐ AFTER PHASE 3
├─ ☐ Fidelity validation
├─ ☐ Security testing
└─ ☐ Robustness analysis

Phase 5 (Weeks 10-11): ☐ OPTIMIZATION
Phase 6 (Weeks 12-13): ☐ PRODUCTION
```

---

# PHASE 1: FOUNDATION (Weeks 1-2)

## Overview
Establish clean, maintainable codebase with core abstractions and test infrastructure.

## Week 1: Code Cleanup & Architecture

### Day 1-2: Initial Cleanup

**Task 1.1: Delete Failed Experiments**

Files to DELETE:
```
nes-llm/src/embedding/residual_embedder_v2.py
nes-llm/src/embedding/constraint_controller.py
nes-llm/src/embedding/distribution_matcher.py
nes-llm/src/embedding/payload_extractor.py
nes-llm/src/embedding/reconstruction.py
nes-llm/src/carrier_selection/random_selector.py
nes-llm/src/crypto/lwe_decoder.py
nes-llm/src/crypto/lwe_encoder.py
```

**Action:**
```bash
git rm <file> # for each file above
git commit -m "Remove failed experiments and incomplete modules"
```

**Verification:**
- Imports still work (run `python -c "import src"`)
- No broken references

---

**Task 1.2: Consolidate Empty/Stub Files**

Files to CONSOLIDATE OR DELETE:
```
nes-llm/src/carrier_selection/carrier_sampler.py
nes-llm/src/crypto/qrng_provider.py  (keep, placeholder for future)
nes-llm/src/utils/io.py (stub)
nes-llm/src/utils/logging.py (stub)
nes-llm/src/utils/seeds.py (stub)
```

**Action:**
1. Review each file
2. If truly empty, DELETE
3. If stub with purpose, ADD DOCSTRING

**Example for QRNG provider:**
```python
# nes-llm/src/crypto/qrng_provider.py

"""
Quantum Random Number Generator Provider.

Provides interface for integrating hardware QRNG
(e.g., Entropy API, QuantumRNG hardware).

For now, uses Python's secrets module as fallback.
TODO: Implement hardware QRNG integration
"""

import secrets

class QRNGProvider:
    """QRNG provider interface."""
    
    @staticmethod
    def get_random_bits(num_bits: int) -> bytes:
        """Get random bits from QRNG."""
        return secrets.token_bytes(num_bits // 8)
```

**Verification:**
- All modules importable
- No circular dependencies

---

### Day 3-4: Architecture Setup

**Task 1.3: Create Core Module Structure**

Create NEW file: `nes-llm/src/core/__init__.py`

```python
"""
Core abstractions and base classes for NES.

This module defines the fundamental interfaces and types
used throughout the NES system.
"""

from .types import (
    TensorType,
    BitSequence,
    ResidualTensor,
    QuantizedWeights,
    EmbeddingConfig,
    ExtractionResult,
)

from .interfaces import (
    Embedder,
    Extractor,
    FeatureExtractor,
    Selector,
    Scorer,
)

from .exceptions import (
    NESException,
    CapacityExceeded,
    RecoveryFailed,
    SecurityViolation,
)

__all__ = [
    'TensorType',
    'BitSequence',
    'ResidualTensor',
    'QuantizedWeights',
    'EmbeddingConfig',
    'ExtractionResult',
    'Embedder',
    'Extractor',
    'FeatureExtractor',
    'Selector',
    'Scorer',
    'NESException',
    'CapacityExceeded',
    'RecoveryFailed',
    'SecurityViolation',
]
```

Create NEW file: `nes-llm/src/core/types.py`

```python
"""Type definitions for NES system."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch

TensorType = torch.Tensor
BitSequence = List[int]

@dataclass
class ResidualTensor:
    """Represents residual tensors from quantization."""
    tensor: TensorType
    layer_id: int
    module_name: str  # 'q_proj', 'down_proj', etc.
    magnitude: float  # mean magnitude
    entropy: float  # Shannon entropy

@dataclass
class QuantizedWeights:
    """Represents quantized weights."""
    weights: TensorType
    quantization_format: str  # 'nf4', 'int4', etc.
    scale_factor: Optional[TensorType] = None
    zero_point: Optional[TensorType] = None

@dataclass
class EmbeddingConfig:
    """Configuration for embedding process."""
    total_payload_bits: int
    embedding_strategy: str  # 'sign', 'magnitude_aware'
    carrier_selection: str  # 'magnitude', 'adaptive'
    use_qaci: bool = True
    alpha: float = 0.25  # margin parameter
    noise_level: float = 0.001  # expected noise sigma

@dataclass
class EmbeddingResult:
    """Result of embedding process."""
    success: bool
    embedded_weights: Dict[str, TensorType]
    carrier_indices: Dict[int, List[int]]  # layer_id -> indices
    metadata: Dict[str, any]

@dataclass
class ExtractionResult:
    """Result of extraction process."""
    success: bool
    recovered_payload: Optional[bytes]
    ber: float  # bit error rate
    accuracy: float  # recovery accuracy
    metadata: Dict[str, any]
```

Create NEW file: `nes-llm/src/core/interfaces.py`

```python
"""Abstract base classes for NES system."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import torch

class Embedder(ABC):
    """Abstract embedder interface."""
    
    @abstractmethod
    def embed(
        self,
        residuals: Dict[int, torch.Tensor],
        bits: List[int],
        config: 'EmbeddingConfig',
    ) -> 'EmbeddingResult':
        """Embed bits into residuals."""
        pass

class Extractor(ABC):
    """Abstract extractor interface."""
    
    @abstractmethod
    def extract(
        self,
        weights: Dict[str, torch.Tensor],
        num_bits: int,
        key: 'CryptoKey',
    ) -> 'ExtractionResult':
        """Extract bits from weights."""
        pass

class FeatureExtractor(ABC):
    """Abstract feature extractor."""
    
    @abstractmethod
    def extract_features(
        self,
        tensor: torch.Tensor,
        window_size: int = 32,
    ) -> torch.Tensor:
        """Extract features from tensor."""
        pass

class Selector(ABC):
    """Abstract carrier selector."""
    
    @abstractmethod
    def select(
        self,
        scores: torch.Tensor,
        num_carriers: int,
    ) -> List[int]:
        """Select carrier indices based on scores."""
        pass

class Scorer(ABC):
    """Abstract quality scorer."""
    
    @abstractmethod
    def score(
        self,
        features: torch.Tensor,
        layer_quality: float,
    ) -> torch.Tensor:
        """Score quality of potential carriers."""
        pass
```

Create NEW file: `nes-llm/src/core/exceptions.py`

```python
"""Custom exceptions for NES system."""

class NESException(Exception):
    """Base exception for NES system."""
    pass

class CapacityExceeded(NESException):
    """Raised when payload exceeds carrier capacity."""
    pass

class RecoveryFailed(NESException):
    """Raised when extraction/recovery fails."""
    pass

class SecurityViolation(NESException):
    """Raised when security constraint violated."""
    pass

class QuantizationError(NESException):
    """Raised when quantization fails."""
    pass

class FidelityError(NESException):
    """Raised when model fidelity constraint violated."""
    pass
```

**Verification:**
```bash
python -c "from src.core import *; print('✓ Core module imports')"
```

---

**Task 1.4: Create Configuration System**

Create NEW file: `nes-llm/src/core/config.py`

```python
"""Configuration management for NES."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

@dataclass
class ModelConfig:
    """Configuration for LLM model."""
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    num_layers: int = 22
    hidden_dim: int = 1024
    dtype: str = "float16"

@dataclass
class EmbeddingStrategyConfig:
    """Embedding strategy configuration."""
    strategy: str = "sign"  # 'sign', 'magnitude_aware', 'adaptive'
    alpha: float = 0.25  # margin parameter
    min_magnitude: float = 0.001

@dataclass
class CarrierSelectionConfig:
    """Carrier selection configuration."""
    method: str = "magnitude"  # 'magnitude', 'adaptive', 'random'
    percentile_threshold: int = 25  # top 25%
    use_qaci: bool = True
    layer_aware: bool = True

@dataclass
class CryptoConfig:
    """Cryptography configuration."""
    algorithm: str = "aes"
    key_size: int = 256
    cipher_mode: str = "gcm"

@dataclass
class NESConfig:
    """Master configuration for NES."""
    model: ModelConfig = field(default_factory=ModelConfig)
    embedding_strategy: EmbeddingStrategyConfig = field(
        default_factory=EmbeddingStrategyConfig
    )
    carrier_selection: CarrierSelectionConfig = field(
        default_factory=CarrierSelectionConfig
    )
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    
    # Payload and noise
    total_payload_bits: int = 50000
    expected_noise_sigma: float = 0.001
    
    # Validation constraints
    max_ppl_degradation: float = 0.02  # 2%
    max_accuracy_loss: float = 0.01  # 1%
    max_kl_divergence: float = 0.05
    
    @classmethod
    def from_json(cls, path: str) -> 'NESConfig':
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str):
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
```

**Verification:**
```bash
python -c "
from src.core.config import NESConfig
cfg = NESConfig()
print(f'✓ Config created: {cfg.total_payload_bits} bits')
"
```

---

### Day 5-7: Core Modules Implementation

**Task 1.5: Implement Core Embedding/Extraction Classes**

Create NEW file: `nes-llm/src/embedding/base_embedder.py`

```python
"""Base embedder class."""

from abc import ABC, abstractmethod
from typing import List, Dict
import torch

from src.core.interfaces import Embedder
from src.core.types import EmbeddingConfig, EmbeddingResult

class BaseEmbedder(Embedder):
    """Base class for embedding strategies."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.alpha = config.alpha
    
    @abstractmethod
    def _embed_single_bit(
        self,
        residual: float,
        bit: int,
    ) -> float:
        """Embed single bit into single residual."""
        pass
    
    def embed(
        self,
        residuals: Dict[int, torch.Tensor],
        bits: List[int],
        selector_indices: Dict[int, List[int]],
    ) -> EmbeddingResult:
        """Embed bits into residuals at selected positions."""
        
        embedded = {}
        bit_idx = 0
        actual_carrier_indices = {}
        
        for layer_id in sorted(residuals.keys()):
            residual_tensor = residuals[layer_id].clone()
            indices = selector_indices.get(layer_id, [])
            
            embedded_flat = residual_tensor.flatten()
            actual_indices = []
            
            for carrier_idx in indices:
                if bit_idx < len(bits):
                    bit = bits[bit_idx]
                    residual_val = embedded_flat[carrier_idx].item()
                    embedded_val = self._embed_single_bit(
                        residual_val,
                        bit
                    )
                    embedded_flat[carrier_idx] = embedded_val
                    actual_indices.append(carrier_idx)
                    bit_idx += 1
            
            embedded[layer_id] = embedded_flat.reshape(
                residual_tensor.shape
            )
            actual_carrier_indices[layer_id] = actual_indices
        
        return EmbeddingResult(
            success=True,
            embedded_weights=embedded,
            carrier_indices=actual_carrier_indices,
            metadata={
                'bits_embedded': bit_idx,
                'total_bits': len(bits),
                'efficiency': bit_idx / len(bits) if bits else 0.0,
            }
        )
```

Create NEW file: `nes-llm/src/extraction/base_extractor.py`

```python
"""Base extractor class."""

from abc import ABC, abstractmethod
from typing import List, Dict
import torch

from src.core.interfaces import Extractor
from src.core.types import ExtractionResult

class BaseExtractor(Extractor):
    """Base class for extraction strategies."""
    
    @abstractmethod
    def _recover_single_bit(self, residual: float) -> int:
        """Recover single bit from single residual."""
        pass
    
    def extract(
        self,
        weights: Dict[int, torch.Tensor],
        carrier_indices: Dict[int, List[int]],
    ) -> ExtractionResult:
        """Extract bits from weights at selected positions."""
        
        recovered_bits = []
        
        for layer_id in sorted(weights.keys()):
            weight_tensor = weights[layer_id]
            indices = carrier_indices.get(layer_id, [])
            
            weight_flat = weight_tensor.flatten()
            
            for carrier_idx in indices:
                residual_val = weight_flat[carrier_idx].item()
                bit = self._recover_single_bit(residual_val)
                recovered_bits.append(bit)
        
        return ExtractionResult(
            success=True,
            recovered_payload=None,  # Will be set by decryption
            ber=0.0,  # Will be calculated
            accuracy=1.0,
            metadata={
                'bits_recovered': len(recovered_bits),
            }
        )
```

**Verification:**
```bash
python -m pytest tests/test_core_embedder.py -v
python -m pytest tests/test_core_extractor.py -v
```

---

### Day 8-10: Unit Testing

**Task 1.6: Create Comprehensive Test Suite**

Create NEW file: `nes-llm/tests/test_core_types.py`

```python
"""Tests for core types."""

import pytest
import torch
from src.core.types import (
    ResidualTensor,
    QuantizedWeights,
    EmbeddingConfig,
)

def test_residual_tensor_creation():
    """Test ResidualTensor dataclass."""
    tensor = torch.randn(100, 256)
    res = ResidualTensor(
        tensor=tensor,
        layer_id=5,
        module_name='down_proj',
        magnitude=0.045,
        entropy=4.2,
    )
    assert res.layer_id == 5
    assert res.magnitude == 0.045
    assert res.tensor.shape == (100, 256)

def test_embedding_config_defaults():
    """Test EmbeddingConfig defaults."""
    cfg = EmbeddingConfig(total_payload_bits=10000)
    assert cfg.total_payload_bits == 10000
    assert cfg.embedding_strategy == 'sign'
    assert cfg.use_qaci == True

def test_quantized_weights():
    """Test QuantizedWeights."""
    weights = torch.randint(0, 16, (1000,), dtype=torch.uint8)
    qw = QuantizedWeights(
        weights=weights,
        quantization_format='nf4',
    )
    assert qw.quantization_format == 'nf4'
```

**File:** `nes-llm/tests/test_core_config.py`

```python
"""Tests for configuration system."""

import pytest
from src.core.config import NESConfig, ModelConfig

def test_default_config():
    """Test default configuration."""
    cfg = NESConfig()
    assert cfg.total_payload_bits == 50000
    assert cfg.model.model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert cfg.max_ppl_degradation == 0.02

def test_config_serialization(tmp_path):
    """Test JSON serialization."""
    cfg = NESConfig(total_payload_bits=100000)
    path = tmp_path / "config.json"
    
    cfg.to_json(str(path))
    cfg_loaded = NESConfig.from_json(str(path))
    
    assert cfg_loaded.total_payload_bits == 100000
```

**Task 1.7: Run Full Test Suite**

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run tests
pytest tests/ -v --cov=src --cov-report=html

# Check coverage
open htmlcov/index.html  # Should show coverage
```

**Success Criteria:**
- All tests pass
- Coverage ≥ 80%
- No warnings

---

## Week 2: Core Module Completion

### Day 1-2: Payload Encoding/Decoding

**Task 2.1: Fix PayloadEncoder**

Update: `nes-llm/src/embedding/payload_encoder.py`

```python
"""Payload encoding/decoding utilities."""

from typing import List, Union

class PayloadEncoder:
    """Encode/decode payloads to/from bits."""
    
    @staticmethod
    def text_to_bits(text: str) -> List[int]:
        """Convert text to bit sequence."""
        # UTF-8 encode
        bytes_data = text.encode('utf-8')
        
        # Length header (32 bits)
        length = len(bytes_data)
        length_bits = [int(b) for b in format(length, '032b')]
        
        # Data bits
        data_bits = []
        for byte in bytes_data:
            byte_bits = [int(b) for b in format(byte, '08b')]
            data_bits.extend(byte_bits)
        
        return length_bits + data_bits
    
    @staticmethod
    def bits_to_text(bits: List[int]) -> str:
        """Convert bit sequence to text."""
        if len(bits) < 32:
            raise ValueError("Insufficient bits for length header")
        
        # Parse length
        length_bits = bits[:32]
        length = int(''.join(map(str, length_bits)), 2)
        
        # Parse data
        data_bits = bits[32:32 + length * 8]
        bytes_list = []
        
        for i in range(0, len(data_bits), 8):
            byte_bits = data_bits[i:i+8]
            if len(byte_bits) == 8:
                byte_val = int(''.join(map(str, byte_bits)), 2)
                bytes_list.append(byte_val)
        
        return bytes(bytes_list).decode('utf-8')

def test_payload_encoding():
    """Quick test."""
    message = "Hello, World!"
    bits = PayloadEncoder.text_to_bits(message)
    recovered = PayloadEncoder.bits_to_text(bits)
    assert recovered == message
    print(f"✓ Payload encoding: {len(bits)} bits for '{message}'")
```

---

### Day 3-5: Strategy Implementations

**Task 2.2: Implement Sign-Based Strategy**

Update/Create: `nes-llm/src/embedding/strategies/sign_strategy.py`

```python
"""Sign-based embedding strategy."""

import torch
from src.embedding.base_embedder import BaseEmbedder
from src.core.types import EmbeddingConfig

class SignStrategy(BaseEmbedder):
    """
    Sign-based embedding strategy.
    
    Encodes bit as sign of residual:
    - bit=1 → positive
    - bit=0 → negative
    """
    
    def _embed_single_bit(
        self,
        residual: float,
        bit: int,
    ) -> float:
        """Embed bit into residual sign."""
        magnitude = max(abs(residual), self.config.alpha)
        return magnitude if bit == 1 else -magnitude
    
    def _recover_single_bit(self, residual: float) -> int:
        """Recover bit from residual sign."""
        return 1 if residual >= 0 else 0

def test_sign_strategy():
    """Quick test."""
    cfg = EmbeddingConfig(total_payload_bits=100)
    strategy = SignStrategy(cfg)
    
    # Test embedding
    residual = 0.045
    bit = 1
    embedded = strategy._embed_single_bit(residual, bit)
    assert embedded > 0
    
    # Test extraction
    recovered = strategy._recover_single_bit(embedded)
    assert recovered == bit
    
    print(f"✓ Sign strategy: embedded bit={bit}, recovered={recovered}")
```

---

### Day 6-10: Integration & Testing

**Task 2.3: Create Integration Test Suite**

Create: `nes-llm/tests/test_integration_embedding.py`

```python
"""Integration tests for embedding pipeline."""

import pytest
import torch
from src.core.types import EmbeddingConfig
from src.core.config import NESConfig
from src.embedding.payload_encoder import PayloadEncoder
from src.embedding.strategies.sign_strategy import SignStrategy

def test_full_embedding_cycle():
    """Test complete embedding->extraction cycle."""
    
    # Setup
    cfg = EmbeddingConfig(
        total_payload_bits=1000,
        embedding_strategy='sign',
    )
    strategy = SignStrategy(cfg)
    
    # Encode message
    message = "Test message for embedding"
    bits = PayloadEncoder.text_to_bits(message)
    
    # Create residuals
    residuals = {
        0: torch.randn(100, 256) * 0.05,  # Small magnitude
    }
    
    # Create selector indices (top 10%)
    num_bits = len(bits)
    selector_indices = {
        0: list(range(min(num_bits, 2560))),
    }
    
    # Embed
    result = strategy.embed(residuals, bits, selector_indices)
    assert result.success
    assert result.metadata['bits_embedded'] == len(bits)
    
    # Extract (sign-based)
    recovered_bits = []
    for layer_id in result.embedded_weights:
        embedded = result.embedded_weights[layer_id]
        indices = result.carrier_indices[layer_id]
        flat = embedded.flatten()
        for idx in indices:
            bit = 1 if flat[idx] >= 0 else 0
            recovered_bits.append(bit)
    
    # Decode
    recovered_message = PayloadEncoder.bits_to_text(recovered_bits)
    assert recovered_message == message
    
    print(f"✓ Full cycle: '{message}' -> {len(bits)} bits -> recovered")

def test_noise_robustness():
    """Test robustness to noise."""
    cfg = EmbeddingConfig(total_payload_bits=100)
    strategy = SignStrategy(cfg)
    
    # Embed
    bits = [1, 0, 1, 1, 0] * 20  # 100 bits
    residuals = {0: torch.randn(10000) * 0.05}
    selector_indices = {0: list(range(100))}
    
    result = strategy.embed(residuals, bits, selector_indices)
    embedded = result.embedded_weights[0].flatten()
    
    # Add noise
    noise = torch.randn_like(embedded) * 0.001
    noisy = embedded + noise
    
    # Extract under noise
    recovered = []
    for i in result.carrier_indices[0]:
        bit = 1 if noisy[i] >= 0 else 0
        recovered.append(bit)
    
    # Calculate BER
    ber = sum(1 for i, b in enumerate(recovered) if b != bits[i]) / len(bits)
    print(f"✓ Noise robustness: BER={ber:.4f} at σ=0.001")
```

**Run Tests:**
```bash
pytest tests/test_integration_embedding.py -v -s
```

---

## Phase 1 Completion Checklist

```
Week 1: Code Cleanup & Architecture
☐ Day 1-2: Delete failed experiments
☐ Day 1-2: Consolidate stub files
☐ Day 3-4: Create core module structure
☐ Day 3-4: Create configuration system
☐ Day 5-7: Implement base embedder/extractor
☐ Day 8-10: Unit test suite

Week 2: Core Module Completion
☐ Day 1-2: Fix payload encoder
☐ Day 3-5: Implement sign strategy
☐ Day 6-10: Integration tests

Verification:
☐ All tests pass (100% success)
☐ Code coverage ≥ 80%
☐ No import errors
☐ No circular dependencies
☐ All type hints present
☐ Docstrings complete

Next: Proceed to Phase 2
```

---

# PHASE 2: CARRIER INTELLIGENCE (Weeks 3-4)

## Overview
Implement layer profiling, feature extraction, and quality scoring (QACI system).

## Week 3: Layer Profiling & Feature Extraction

### Day 1-2: Layer Quality Profiling

**Task 3.1: Implement LayerProfiler**

File: `nes-llm/src/carrier_intelligence/layer_profiler.py`

```python
"""Layer quality profiling for QACI."""

import torch
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class LayerProfile:
    """Profile of a single layer."""
    layer_id: int
    module_name: str
    num_params: int
    mag_mean: float
    mag_std: float
    mag_max: float
    entropy: float
    quality_score: float  # [0, 1]
    position_bias: float
    adjusted_quality: float

class LayerProfiler:
    """Profile layers for carrier capacity."""
    
    def profile_layer(
        self,
        layer_id: int,
        residuals: torch.Tensor,
        module_name: str,
        model_depth: int,
    ) -> LayerProfile:
        """Profile single layer."""
        
        flat = residuals.flatten()
        magnitudes = torch.abs(flat)
        
        # Basic statistics
        mag_mean = magnitudes.mean().item()
        mag_std = magnitudes.std().item()
        mag_max = magnitudes.max().item()
        
        # Entropy
        entropy = self._compute_entropy(magnitudes)
        
        # Quality score (magnitude-based)
        quality_score = (
            0.6 * (mag_mean / (mag_mean + mag_std + 1e-8)) +
            0.4 * (mag_max / (mag_max + 1))
        )
        
        # Position bias (early/late layers less robust)
        if layer_id < 5:
            position_bias = 0.7
        elif layer_id < model_depth - 5:
            position_bias = 1.0
        else:
            position_bias = 0.8
        
        adjusted_quality = quality_score * position_bias
        
        return LayerProfile(
            layer_id=layer_id,
            module_name=module_name,
            num_params=len(flat),
            mag_mean=mag_mean,
            mag_std=mag_std,
            mag_max=mag_max,
            entropy=entropy,
            quality_score=quality_score,
            position_bias=position_bias,
            adjusted_quality=adjusted_quality,
        )
    
    @staticmethod
    def _compute_entropy(values: torch.Tensor) -> float:
        """Compute Shannon entropy of distribution."""
        hist, _ = torch.histogram(
            values,
            bins=256,
            range=(0, values.max()),
        )
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        return entropy

def test_layer_profiler():
    """Quick test."""
    profiler = LayerProfiler()
    residuals = torch.randn(10000) * 0.05
    profile = profiler.profile_layer(
        layer_id=15,
        residuals=residuals,
        module_name='down_proj',
        model_depth=32,
    )
    print(f"✓ Profile: quality={profile.quality_score:.3f}, "
          f"adjusted={profile.adjusted_quality:.3f}")
```

---

### Day 3-5: Feature Extraction

**Task 3.2: Implement FeatureExtractor**

File: `nes-llm/src/carrier_intelligence/feature_extractor.py`

Update to include:

```python
"""Feature extraction for carrier intelligence."""

import torch
from typing import Tuple

class CarrierFeatureExtractor:
    """Extract features from residuals."""
    
    def __init__(self, window_size: int = 32):
        self.window_size = window_size
    
    def extract_features(
        self,
        residuals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract feature matrix from residuals.
        
        Output shape: [num_residuals, num_features]
        Features: magnitude, entropy, variance, kurtosis
        """
        
        flat = residuals.flatten()
        num_residuals = len(flat)
        
        features = torch.zeros(
            num_residuals,
            4,  # magnitude, entropy, variance, kurtosis
            dtype=torch.float32,
        )
        
        # Feature 1: Magnitude
        features[:, 0] = torch.abs(flat)
        
        # Features 2-4: Local statistics (windowed)
        for i in range(num_residuals):
            start = max(0, i - self.window_size // 2)
            end = min(num_residuals, i + self.window_size // 2)
            window = flat[start:end]
            
            # Local entropy
            hist, _ = torch.histogram(window, bins=32)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            entropy = -torch.sum(probs * torch.log2(probs))
            features[i, 1] = entropy
            
            # Local variance
            features[i, 2] = window.var()
            
            # Local kurtosis (approximation)
            centered = window - window.mean()
            kurtosis = (
                (centered ** 4).mean() /
                (window.var() ** 2 + 1e-8)
            )
            features[i, 3] = kurtosis
        
        return features

def test_feature_extractor():
    """Quick test."""
    extractor = CarrierFeatureExtractor()
    residuals = torch.randn(1000) * 0.05
    features = extractor.extract_features(residuals)
    assert features.shape == (1000, 4)
    print(f"✓ Features extracted: shape={features.shape}")
```

---

### Day 6-10: Quality Scoring & Scheduling

**Task 3.3: Implement Quality Score Engine**

File: `nes-llm/src/carrier_intelligence/quality_score.py`

```python
"""Quality scoring for carriers."""

import torch

class QualityScorer:
    """Score quality of potential carriers."""
    
    def __init__(
        self,
        mag_weight: float = 0.6,
        entropy_weight: float = 0.2,
        variance_weight: float = 0.1,
        kurtosis_weight: float = 0.1,
    ):
        self.mag_weight = mag_weight
        self.entropy_weight = entropy_weight
        self.variance_weight = variance_weight
        self.kurtosis_weight = kurtosis_weight
    
    def score(
        self,
        features: torch.Tensor,
        layer_quality: float = 1.0,
    ) -> torch.Tensor:
        """Score carriers."""
        
        # Normalize features
        features_norm = self._normalize(features)
        
        # Weighted combination
        scores = (
            self.mag_weight * features_norm[:, 0] +
            self.entropy_weight * features_norm[:, 1] +
            self.variance_weight * features_norm[:, 2] +
            self.kurtosis_weight * features_norm[:, 3]
        )
        
        # Apply layer quality factor
        scores = scores * layer_quality
        
        return scores
    
    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        """Normalize features to [0, 1]."""
        min_vals = features.min(dim=0)[0]
        max_vals = features.max(dim=0)[0]
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0  # Avoid division by zero
        return (features - min_vals) / ranges
```

**Task 3.4: Implement CarrierScheduler (QACI Allocation)**

File: `nes-llm/src/carrier_intelligence/carrier_scheduler.py`

```python
"""Carrier scheduling for intelligent payload allocation."""

from typing import Dict, List
from src.carrier_intelligence.layer_profiler import LayerProfile

class CarrierScheduler:
    """Allocate payload across layers."""
    
    def schedule_allocation(
        self,
        total_payload: int,
        layer_profiles: Dict[int, LayerProfile],
        num_layers: int,
    ) -> Dict[int, int]:
        """
        Allocate payload bits to each layer.
        
        Returns: {layer_id: num_bits_for_layer}
        """
        
        # Compute total quality
        qualities = []
        for i in range(num_layers):
            if i in layer_profiles:
                qualities.append(layer_profiles[i].adjusted_quality)
            else:
                qualities.append(1.0)
        
        total_quality = sum(qualities)
        
        # Allocate proportionally
        allocation = {}
        remaining = total_payload
        
        for i, quality in enumerate(qualities[:-1]):
            portion = int((quality / total_quality) * total_payload)
            allocation[i] = portion
            remaining -= portion
        
        # Last layer gets remainder
        allocation[num_layers - 1] = remaining
        
        return allocation
```

---

## Week 4: Integration & Testing

### Day 1-5: Comprehensive Testing

**Task 4.1: Create QACI Integration Tests**

File: `nes-llm/tests/test_qaci_system.py`

```python
"""Tests for QACI system."""

import pytest
import torch
from src.carrier_intelligence.layer_profiler import LayerProfiler
from src.carrier_intelligence.feature_extractor import CarrierFeatureExtractor
from src.carrier_intelligence.quality_score import QualityScorer
from src.carrier_intelligence.carrier_scheduler import CarrierScheduler

def test_full_qaci_pipeline():
    """Test complete QACI pipeline."""
    
    # Profile layers
    profiler = LayerProfiler()
    profiles = {}
    for layer_id in range(32):
        residuals = torch.randn(100, 256) * 0.05
        profile = profiler.profile_layer(
            layer_id=layer_id,
            residuals=residuals,
            module_name='down_proj',
            model_depth=32,
        )
        profiles[layer_id] = profile
    
    # Extract features
    extractor = CarrierFeatureExtractor()
    features = extractor.extract_features(
        torch.randn(25600) * 0.05
    )
    
    # Score carriers
    scorer = QualityScorer()
    scores = scorer.score(features, layer_quality=0.8)
    assert scores.min() >= 0 and scores.max() <= 1
    
    # Schedule allocation
    scheduler = CarrierScheduler()
    allocation = scheduler.schedule_allocation(
        total_payload=100000,
        layer_profiles=profiles,
        num_layers=32,
    )
    
    assert sum(allocation.values()) == 100000
    print(f"✓ QACI pipeline: allocated {sum(allocation.values())} bits")

def test_quality_score_distribution():
    """Test quality score distribution."""
    
    profiler = LayerProfiler()
    qualities = []
    
    for layer_id in [0, 15, 31]:  # Early, mid, late
        residuals = torch.randn(10000) * 0.05
        profile = profiler.profile_layer(
            layer_id=layer_id,
            residuals=residuals,
            module_name='down_proj',
            model_depth=32,
        )
        qualities.append(profile.adjusted_quality)
    
    # Early layer should have lower quality
    assert qualities[0] < qualities[1]
    # Late layer should have lower quality
    assert qualities[2] < qualities[1]
    
    print(f"✓ Quality distribution: early={qualities[0]:.3f}, "
          f"mid={qualities[1]:.3f}, late={qualities[2]:.3f}")
```

---

## Phase 2 Completion Checklist

```
Week 3: Layer Profiling & Feature Extraction
☐ Day 1-2: Implement LayerProfiler
☐ Day 3-5: Implement FeatureExtractor
☐ Day 6-10: QualityScorer & CarrierScheduler

Week 4: Integration & Testing
☐ Day 1-5: QACI integration tests
☐ Day 6-10: Full pipeline testing

Verification:
☐ All layer profiles generate correctly
☐ Features have expected shape
☐ Quality scores in [0, 1]
☐ Allocation sums to total_payload
☐ All tests pass

Next: Proceed to Phase 3
```

---

# PHASE 3: EMBEDDING & EXTRACTION (Weeks 5-6)

## Overview
Implement complete embedding, encryption, and extraction pipelines.

## Week 5: Embedding Pipeline

### Task 5.1: Intelligent Embedder Completion

Update: `nes-llm/src/embedding/intelligent_embedder.py`

Key methods to complete:
- `embed_bits()` - Main embedding orchestrator
- `_select_carriers()` - Use carrier selector
- `_embed_per_layer()` - Per-layer embedding
- `_verify_fidelity()` - Fidelity checks

### Task 5.2: Implement AES Encryption

Update: `nes-llm/src/crypto/aes_cipher.py`

```python
"""AES encryption implementation."""

import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class AESCipher:
    """AES-256-GCM cipher."""
    
    def __init__(self, key: bytes):
        self.key = key
        self.cipher = AESGCM(key)
    
    def encrypt(self, plaintext: bytes) -> Dict[str, bytes]:
        """Encrypt plaintext."""
        iv = os.urandom(12)  # 96-bit IV for GCM
        ciphertext = self.cipher.encrypt(iv, plaintext, None)
        
        # Return IV + ciphertext + auth tag
        return {
            'iv': iv,
            'ciphertext': ciphertext[:-16],  # Remove auth tag
            'auth_tag': ciphertext[-16:],     # Last 16 bytes
        }
    
    def decrypt(
        self,
        iv: bytes,
        ciphertext: bytes,
        auth_tag: bytes,
    ) -> bytes:
        """Decrypt ciphertext."""
        return self.cipher.decrypt(
            iv,
            ciphertext + auth_tag,
            None
        )
```

---

## Week 6: Extraction Pipeline

### Task 6.1: Residual Recovery

File: `nes-llm/src/extraction/residual_recovery.py`

```python
"""Recover residuals from quantized weights."""

import torch

def recover_residuals(
    original_nf4_weights: torch.Tensor,
    extracted_fp16_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Recover residuals: residual = extracted_fp16 - original_nf4
    """
    return extracted_fp16_weights - original_nf4_weights
```

### Task 6.2: Bit & Payload Recovery

File: `nes-llm/src/extraction/payload_recovery.py`

```python
"""Recover payload from extracted bits."""

from typing import List
from src.embedding.payload_encoder import PayloadEncoder

class PayloadRecovery:
    """Recover payload from bits."""
    
    @staticmethod
    def recover_payload(bits: List[int]) -> str:
        """Recover text payload from bits."""
        return PayloadEncoder.bits_to_text(bits)
    
    @staticmethod
    def calculate_ber(
        original_bits: List[int],
        recovered_bits: List[int],
    ) -> float:
        """Calculate bit error rate."""
        if len(original_bits) != len(recovered_bits):
            raise ValueError("Bit sequences must have same length")
        
        errors = sum(
            1 for o, r in zip(original_bits, recovered_bits)
            if o != r
        )
        return errors / len(original_bits)
```

---

# PHASE 4: VALIDATION & SECURITY (Weeks 7-9)

## Overview
Test fidelity, security, and robustness.

### Key Experiments

**Exp 4.1: Clean Recovery Test**
- Payload: 1K, 10K, 50K, 100K bits
- Expected: BER=0, Accuracy=100%

**Exp 4.2: Quantization Robustness**
- Test NF4 requantization cycles
- Expected: No additional degradation after cycle 1

**Exp 4.3: Noise Robustness**
- Add Gaussian noise: σ ∈ [0.0, 0.05]
- Expected: Graceful degradation curve

**Exp 4.4: Fidelity Verification**
- Perplexity: WikiText-2
- Tasks: MMLU (5-shot), GSM8K (8-shot)
- Expected: PPL < 2%, accuracy loss < 1%

**Exp 4.5: Statistical Detection**
- KL divergence analysis
- Entropy comparison
- Expected: KL < 0.05

---

# PHASE 5 & 6: OPTIMIZATION & PRODUCTION

Once Phases 1-4 complete, follow similar pattern for:
- Performance optimization
- Final testing
- Documentation
- Release preparation

---

## MASTER CHECKLIST

```
PHASE 1: ✓ or ☐
PHASE 2: ✓ or ☐
PHASE 3: ✓ or ☐
PHASE 4: ✓ or ☐
PHASE 5: ✓ or ☐
PHASE 6: ✓ or ☐

READY FOR PRODUCTION: YES / NO
```

---

**End of Phase Implementation Plan**

Start with PHASE 1, Week 1, Day 1!
