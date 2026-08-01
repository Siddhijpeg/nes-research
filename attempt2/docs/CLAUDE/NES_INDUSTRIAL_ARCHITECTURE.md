# Neural-Entropic Steganography (NES)
## Industrial Architecture & Product Specification
**Classification:** HIGHLY CONFIDENTIAL - RESEARCH ONLY  
**Version:** 2.0 - Industrial Grade  
**Date:** 2026-07-18  
**Status:** Architecture Finalization Phase

---

## TABLE OF CONTENTS
1. [System Architecture](#system-architecture)
2. [Component Specifications](#component-specifications)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Execution Phases](#execution-phases)
5. [Quality Assurance Gates](#quality-assurance-gates)
6. [Deployment Strategy](#deployment-strategy)

---

## SECTION 1: SYSTEM ARCHITECTURE

### 1.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NES - INDUSTRIAL SYSTEM                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
        ┌──────────────────────┐  ┌────────────────────────┐
        │   INPUT LAYER        │  │  CONFIGURATION LAYER   │
        │                      │  │                        │
        │ • LLM Model          │  │ • Payload Size         │
        │ • Payload Data       │  │ • Noise Profile        │
        │ • Quantization Cfg   │  │ • Security Level       │
        └──────────┬───────────┘  └────────────┬───────────┘
                   │                           │
                   └───────────────┬───────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  PROFILING & ANALYSIS LAYER  │
                    │                              │
                    │ ┌────────────────────────┐   │
                    │ │  Model Profiler        │   │
                    │ │ • Layer analysis       │   │
                    │ │ • Weight distribution  │   │
                    │ └────────────────────────┘   │
                    │ ┌────────────────────────┐   │
                    │ │  Residual Extractor    │   │
                    │ │ • FP16 vs NF4 diff     │   │
                    │ │ • Entropy analysis     │   │
                    │ └────────────────────────┘   │
                    │ ┌────────────────────────┐   │
                    │ │  Layer Profiler (QACI) │   │
                    │ │ • Quality scores       │   │
                    │ │ • Capacity allocation  │   │
                    │ └────────────────────────┘   │
                    └──────────────┬────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │ CARRIER INTELLIGENCE LAYER      │
                    │                                 │
                    │ ┌────────────────────────────┐  │
                    │ │ Feature Extraction         │  │
                    │ │ • Magnitude               │  │
                    │ │ • Entropy                 │  │
                    │ │ • Local variance          │  │
                    │ │ • Kurtosis               │  │
                    │ └────────────────────────────┘  │
                    │ ┌────────────────────────────┐  │
                    │ │ Quality Scoring           │  │
                    │ │ • Carrier quality         │  │
                    │ │ • Layer importance        │  │
                    │ │ • Reliability estimates   │  │
                    │ └────────────────────────────┘  │
                    │ ┌────────────────────────────┐  │
                    │ │ Carrier Selection         │  │
                    │ │ • Magnitude-based         │  │
                    │ │ • Adaptive scoring        │  │
                    │ │ • Multi-layer strategy    │  │
                    │ └────────────────────────────┘  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   EMBEDDING LAYER               │
                    │                                 │
                    │ ┌────────────────────────────┐  │
                    │ │ Payload Encoding           │  │
                    │ │ • Text → Bits              │  │
                    │ │ • Bit packing              │  │
                    │ └────────────────────────────┘  │
                    │ ┌────────────────────────────┐  │
                    │ │ Embedding Strategies       │  │
                    │ │ • Sign-based               │  │
                    │ │ • Magnitude-aware          │  │
                    │ │ • Adaptive margins         │  │
                    │ └────────────────────────────┘  │
                    │ ┌────────────────────────────┐  │
                    │ │ Embedding Executor         │  │
                    │ │ • Per-layer embedding      │  │
                    │ │ • Multi-layer coordination │  │
                    │ │ • Constraint monitoring    │  │
                    │ └────────────────────────────┘  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   CRYPTOGRAPHY LAYER            │
                    │                                 │
                    │ ┌────────────────────────────┐  │
                    │ │ Key Management             │  │
                    │ │ • AES-256 keys             │  │
                    │ │ • QRNG seeds               │  │
                    │ │ • Key serialization        │  │
                    │ └────────────────────────────┘  │
                    │ ┌────────────────────────────┐  │
                    │ │ Encryption                 │  │
                    │ │ • AES-256-GCM              │  │
                    │ │ • Payload encryption       │  │
                    │ │ • Authentication tags      │  │
                    │ └────────────────────────────┘  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   QUANTIZATION LAYER            │
                    │                                 │
                    │ ┌────────────────────────────┐  │
                    │ │ NF4 Requantization         │  │
                    │ │ • Modified weights → NF4   │  │
                    │ │ • Codebook matching        │  │
                    │ │ • Artifact preservation    │  │
                    │ └────────────────────────────┘  │
                    │ ┌────────────────────────────┐  │
                    │ │ Fidelity Verification      │  │
                    │ │ • Perplexity check         │  │
                    │ │ • Task accuracy check      │  │
                    │ │ • Constraint validation    │  │
                    │ └────────────────────────────┘  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   STEGANALYSIS LAYER            │
                    │                                 │
                    │ ┌────────────────────────────┐  │
                    │ │ Detection Analysis         │  │
                    │ │ • Statistical tests        │  │
                    │ │ • KL divergence            │  │
                    │ │ • Entropy analysis         │  │
                    │ └────────────────────────────┘  │
                    │ ┌────────────────────────────┐  │
                    │ │ Security Verification      │  │
                    │ │ • Indistinguishability     │  │
                    │ │ • Attack resistance        │  │
                    │ │ • Robustness testing       │  │
                    │ └────────────────────────────┘  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   EXTRACTION LAYER              │
                    │                                 │
                    │ ┌────────────────────────────┐  │
                    │ │ Residual Recovery          │  │
                    │ │ • FP16 extraction          │  │
                    │ │ • NF4 dequantization       │  │
                    │ │ • Residual reconstruction  │  │
                    │ └────────────────────────────┘  │
                    │ ┌────────────────────────────┐  │
                    │ │ Bit Recovery               │  │
                    │ │ • Sign extraction          │  │
                    │ │ • Bit-wise recovery        │  │
                    │ │ • Error correction         │  │
                    │ └────────────────────────────┘  │
                    │ ┌────────────────────────────┐  │
                    │ │ Decryption                 │  │
                    │ │ • AES-256-GCM decryption   │  │
                    │ │ • Authentication verify    │  │
                    │ │ • Payload reconstruction   │  │
                    │ └────────────────────────────┘  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   OUTPUT & EVALUATION LAYER     │
                    │                                 │
                    │ ┌────────────────────────────┐  │
                    │ │ Recovery Validation        │  │
                    │ │ • BER calculation          │  │
                    │ │ • Accuracy metrics         │  │
                    │ │ • Extraction statistics    │  │
                    │ └────────────────────────────┘  │
                    │ ┌────────────────────────────┐  │
                    │ │ Output Generation          │  │
                    │ │ • Recovered payload        │  │
                    │ │ • Metadata/statistics      │  │
                    │ │ • Verification report      │  │
                    │ └────────────────────────────┘  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   OUTPUT: Results & Model       │
                    │                                 │
                    │ • Steganographic Model          │
                    │ • Extracted Payload             │
                    │ • Recovery Report               │
                    │ • Security Certificate          │
                    └──────────────────────────────────┘
```

### 1.2 Module Dependencies & Interactions

```
Input Models & Payload
    │
    ├──────────────────────────────────────────────────────────┐
    │                                                          │
    ▼                                                          ▼
Model Profiler ◄────────────────────────────────────── Payload Encoder
    │                                                          │
    ├─► Residual Extractor ◄──────────────────────────────────┤
    │       │                                                  │
    │       ├─► Entropy Profiler                              │
    │       │       │                                          │
    │       └─► Layer Profiler (QACI) ◄──────────────────────┤
    │               │                                          │
    │               └─► Quality Score Engine                  │
    │                       │                                  │
    │                       ├─► Carrier Scheduler ◄───────────┤
    │                       │       │                          │
    │                       │       ├─► Layer Importance       │
    │                       │       │                          │
    │                       │       └─► Carrier Reliability    │
    │                       │                                  │
    │                       └─► Feature Extractor             │
    │                               │                          │
    │                               ├─► Feature Normalizer     │
    │                               │                          │
    │                               └─► Carrier Score Engine   │
    │                                       │                  │
    │                                       ├─► Local Entropy  │
    │                                       │                  │
    │                                       └─► Confidence Est.│
    │                                                          │
    └──────► Carrier Selector ◄─────────────────────────────┬─┘
                    │                                       │
                    ├─► Adaptive Selector                  │
                    │                                      │
                    └─► Magnitude Selector                 │
                            │                              │
                            ├─► Adaptive Margin Controller │
                            │                              │
                            └─► Embedder Selection Strategy◄──┘
                                    │
                                    ├─► Sign-Based Embedder
                                    │
                                    └─► Intelligent Embedder
                                            │
                                            ├─► Embedding Executor
                                            │
                                            └─► Embedding Result
                                                    │
                                                    ├─► AES Cipher
                                                    │
                                                    ├─► Key Manager
                                                    │
                                                    └─► Crypto Pipeline
                                                            │
                                                            ├─► NF4 Loader
                                                            │
                                                            ├─► Quantization Model
                                                            │
                                                            └─► Weight Reconstructor
                                                                    │
                                                                    ├─► Residual Recovery
                                                                    │
                                                                    ├─► Payload Recovery
                                                                    │
                                                                    ├─► Decision Rule
                                                                    │
                                                                    └─► Extraction Result
                                                                            │
                                                                            ├─► Perplexity Evaluator
                                                                            │
                                                                            ├─► Task Evaluator (MMLU/GSM8K)
                                                                            │
                                                                            ├─► Real Detector
                                                                            │
                                                                            └─► Recovery Report
```

---

## SECTION 2: COMPONENT SPECIFICATIONS

### 2.1 Core Component Details

#### A. Profiling & Analysis Layer

**Purpose:** Analyze the LLM model and extract key properties needed for intelligent embedding.

**Subcomponents:**

1. **Model Profiler**
   - Input: LLM model (FP16)
   - Output: Layer-wise weight statistics
   - Key metrics: weight magnitude, entropy, variance
   - Implementation: `profiling/model_profiler.py`
   - Expected runtime: 2-5 minutes for Llama-3-8B

2. **Residual Extractor**
   - Input: FP16 weights, NF4 quantized weights
   - Output: Residual tensors per layer/matrix
   - Calculation: `residual = FP16_weight - NF4_weight`
   - Implementation: `residuals/extractor.py`
   - Handles: All weight matrices (q_proj, down_proj, etc.)

3. **Layer Profiler (QACI)**
   - Input: Residuals per layer
   - Output: Layer quality scores
   - Metrics computed:
     - Mean/std magnitude
     - Entropy of distribution
     - Local variance patterns
     - Position-based bias
   - Implementation: `carrier_intelligence/layer_profiler.py`
   - Output: `LayerProfile` objects with quality_score in [0, 1]

#### B. Carrier Intelligence Layer

**Purpose:** Identify and score carrier positions (residuals) that can safely hold embedded data.

**Subcomponents:**

1. **Feature Extraction Engine**
   - Inputs: Residual tensor, layer context
   - Outputs: Feature matrix [N_residuals × N_features]
   - Features extracted:
     - Magnitude: `|residual|`
     - Local entropy: Shannon entropy in 32-element windows
     - Local variance: Std dev in windows
     - Kurtosis: 4th moment measure
     - Neighborhood density: How clustered similar values are
   - Implementation: `carrier_intelligence/feature_extractor.py`
   - Dimensionality: Typically 5-10 features per residual

2. **Quality Scoring Engine**
   - Inputs: Feature matrix, layer quality profile
   - Outputs: Quality score per residual in [0, 1]
   - Scoring formula:
     ```
     quality_score[i] = (
       w_magnitude * normalize(magnitude[i]) +
       w_entropy * normalize(entropy[i]) +
       w_variance * normalize(variance[i]) +
       w_kurtosis * normalize(kurtosis[i]) +
       w_layer * layer_quality_factor
     )
     ```
   - Weights: Configurable, learned from validation data
   - Implementation: `carrier_intelligence/carrier_score.py`

3. **Carrier Scheduler (QACI Allocation)**
   - Inputs: Total payload size, layer profiles, quality scores
   - Outputs: Per-layer payload allocation
   - Algorithm: Weighted allocation based on layer quality
   - Example:
     ```
     Layer 0:  quality=0.6 → 100K * 0.6/30 ≈ 2,000 bits
     Layer 15: quality=0.95 → 100K * 0.95/30 ≈ 3,200 bits
     Layer 31: quality=0.7  → 100K * 0.7/30 ≈ 2,300 bits
     ```
   - Implementation: `carrier_intelligence/carrier_scheduler.py`

4. **Carrier Selector**
   - Inputs: Quality scores per residual, allocation for layer
   - Outputs: Indices of selected carrier positions
   - Strategies:
     - **Top-K**: Select highest quality residuals (default)
     - **Adaptive**: Use quality threshold + magnitude weighting
     - **Probabilistic**: Sample based on quality distribution
   - Implementation: `carrier_selection/selectors/adaptive_selector.py`
   - Selection size: Matches allocated payload size for layer

#### C. Embedding Layer

**Purpose:** Encode data into selected carrier positions while preserving model fidelity.

**Subcomponents:**

1. **Payload Encoder**
   - Input: Message string (or binary data)
   - Output: Bit sequence
   - Encoding: UTF-8 → bits, with length header
   - Implementation: `embedding/payload_encoder.py`
   - Header format: [32-bit length] + [payload bits]

2. **Embedding Strategy**
   - **Sign-Based Embedding** (Primary):
     ```
     For each bit b:
       magnitude = |residual[i]|
       if b == 1:
         embedded[i] = +magnitude
       else:
         embedded[i] = -magnitude
     ```
   - **Magnitude-Aware** (Advanced):
     ```
     margin = alpha * magnitude  # alpha typically 0.25
     if b == 1:
       embedded[i] = magnitude + margin
     else:
       embedded[i] = -magnitude - margin
     ```
   - Implementation: `embedding/strategies/sign_strategy.py`

3. **Embedding Executor**
   - Input: Residual tensor, selected indices, bits
   - Output: Modified residual tensor
   - Process:
     - For each (index, bit) pair:
       - Get residual at index
       - Apply embedding strategy
       - Update residual
   - Implementation: `embedding/intelligent_embedder.py`
   - Verification: Ensure no changes to non-selected carriers

#### D. Cryptography Layer

**Purpose:** Encrypt embedded payload for security and authentication.

**Subcomponents:**

1. **Key Manager**
   - Key generation: AES-256 random key
   - Key storage: Secure serialization
   - QRNG seed: For carrier selection randomization
   - Implementation: `crypto/key_manager.py`
   - Output: `CryptoKey` object with metadata

2. **AES Cipher**
   - Algorithm: AES-256-GCM (authenticated encryption)
   - Input: Plaintext bits, AES key
   - Output: Ciphertext + authentication tag
   - IV: Random, stored with ciphertext
   - Implementation: `crypto/aes_cipher.py`
   - Security level: 256-bit key = 2^256 search space

#### E. Quantization & Verification Layer

**Purpose:** Requantize modified weights to NF4 and verify model fidelity.

**Subcomponents:**

1. **NF4 Loader**
   - Input: Modified FP16 weights
   - Output: NF4 quantized weights
   - Codebook: NormalFloat-4 (8-point quantization)
   - Implementation: `quantization/nf4_loader.py`
   - Process:
     - Block-wise quantization (256-element blocks)
     - Min/max calculation per block
     - Centroid assignment

2. **Fidelity Verification**
   - Perplexity: Measure language modeling quality
   - Task accuracy: MMLU (5-shot), GSM8K (8-shot)
   - Constraints checked:
     - PPL degradation < 2%
     - Task accuracy loss < 1%
   - Implementation: `evaluation/perplexity.py`, `evaluation/mmlu.py`

#### F. Steganalysis Layer

**Purpose:** Detect and characterize steganography to verify security.

**Subcomponents:**

1. **Statistical Detector**
   - KL divergence: Between clean and embedded weight distributions
   - Entropy analysis: Residual entropy changes
   - Magnitude distribution: Histogram differences
   - Implementation: `steganalysis/feature_detector.py`
   - Expected: KL < 0.05 (indistinguishable)

2. **Neural Detector**
   - Trained classifier to detect steganography
   - Input: Weight features
   - Output: Detection probability
   - Implementation: `steganalysis/real_neural_detector.py`
   - Expected: Accuracy < 55% (random baseline)

#### G. Extraction Layer

**Purpose:** Recover embedded payload from modified weights.

**Subcomponents:**

1. **Residual Recovery**
   - Input: Modified NF4 weights
   - Output: Extracted residuals
   - Process:
     - Dequantize NF4 to FP16
     - Calculate residuals (FP16 - extracted)
   - Implementation: `extraction/residual_recovery.py`

2. **Bit Recovery**
   - Input: Extracted residuals at selected indices
   - Output: Recovered bits
   - Sign extraction:
     ```
     if residual >= 0:
       bit = 1
     else:
       bit = 0
     ```
   - Implementation: `extraction/payload_recovery.py`

3. **Decryption**
   - Input: Encrypted bits, AES key
   - Output: Decrypted plaintext
   - Algorithm: AES-256-GCM decryption
   - Verification: Authentication tag check
   - Implementation: `extraction/decrypt_pipeline.py`

---

## SECTION 3: DATA FLOW ARCHITECTURE

### 3.1 Embedding Pipeline Data Flow

```
┌─────────────────────────────────┐
│   LLM Model (Llama-3-8B)        │
│   FP16 Weights [8B × 4096]      │
└────────────────┬────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ Load NF4 Quantized │
        │ Model              │
        └────────────┬───────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │ Extract Residuals       │
        │ FP16 - NF4 = Residual   │
        │ Shape: [N_layers][*]    │
        └────────────┬────────────┘
                     │
         ┌───────────┴──────────────┐
         │                          │
         ▼                          ▼
    ┌────────────┐        ┌──────────────┐
    │ Profile    │        │ Extract      │
    │ Each Layer │        │ Features     │
    │            │        │              │
    │Quality:0.7 │        │Mag:0.045     │
    │            │        │Ent:4.2       │
    └────────────┘        │Var:0.032     │
         │                │Kur:2.1       │
         │                └──────┬───────┘
         │                       │
         └──────────┬────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │ Compute Quality      │
        │ Scores Per Residual  │
        │ Shape: [N]           │
        │ Range: [0.0, 1.0]    │
        └────────────┬─────────┘
                     │
         ┌───────────┴──────────────┐
         │                          │
         ▼                          ▼
    ┌────────────┐        ┌──────────────┐
    │ Allocate   │        │ Encode       │
    │ Payload    │        │ Payload      │
    │ Per Layer  │        │              │
    │Layer 0:    │        │Message→Bits  │
    │  2K bits   │        │8,000 bits    │
    │Layer 15:   │        │              │
    │  5K bits   │        │Length header │
    └────────────┘        └──────┬───────┘
         │                       │
         └──────────┬────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │ For Each Layer:      │
        │                      │
        │ 1. Select Carriers   │
        │    (Top-K quality)   │
        │    Indices: [i1,...] │
        │                      │
        │ 2. Embed Bits        │
        │    Apply strategy    │
        │    Residual[i] ← bit │
        │                      │
        │ 3. Collect Modified  │
        │    Residuals         │
        └────────────┬─────────┘
                     │
                     ▼
        ┌──────────────────────┐
        │ Reconstruct Weights  │
        │ Modified = FP16 +    │
        │   Modified Residual  │
        │ Shape: [8B × 4096]   │
        └────────────┬─────────┘
                     │
                     ▼
        ┌──────────────────────┐
        │ Requantize to NF4    │
        │                      │
        │ Embedded Model       │
        │ NF4 Weights [8B/32]  │
        └────────────┬─────────┘
                     │
         ┌───────────┴──────────────┐
         │                          │
         ▼                          ▼
    ┌────────────┐        ┌──────────────┐
    │ Encrypt    │        │ Verify       │
    │ Payload    │        │ Fidelity     │
    │            │        │              │
    │AES-256-GCM │        │PPL check     │
    │Key, IV, CT │        │Task accuracy │
    │Auth Tag    │        │              │
    └────────────┘        │All OK?       │
         │                └──────┬───────┘
         │                       │
         └──────────┬────────────┘
                    │ YES
                    ▼
        ┌──────────────────────┐
        │ Generate Output      │
        │                      │
        │ • Embedded Model     │
        │ • Crypto Key         │
        │ • Metadata           │
        └──────────────────────┘
```

### 3.2 Extraction Pipeline Data Flow

```
┌──────────────────────────────┐
│   Embedded Model (NF4)       │
│   Crypto Key (stored)        │
└────────────┬─────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ Load Embedded NF4  │
    │ Dequantize to FP16 │
    └────────────┬───────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Extract Residuals       │
    │ Residual = FP16 -       │
    │            Original NF4 │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Recover Selected        │
    │ Carrier Indices         │
    │ (using stored key)      │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Extract Bits from       │
    │ Signs at Selected Idx   │
    │                         │
    │ For each idx:           │
    │   bit = sign(residual)  │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Reconstruct Bitstream   │
    │ (concatenate all bits)  │
    │ Bits: [b1, b2, ..., bn] │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Decrypt with AES Key    │
    │                         │
    │ AES-256-GCM Decrypt     │
    │ Verify Auth Tag         │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Parse Length Header     │
    │ Extract Payload         │
    │ Decode to Text          │
    └──────────────────────────┘
```

---

## SECTION 4: EXECUTION PHASES

### Phase Overview

```
PHASE 1: FOUNDATION (Weeks 1-2)
├─ Code Cleanup & Architecture
├─ Core Module Implementation
└─ Unit Testing

PHASE 2: CARRIER INTELLIGENCE (Weeks 3-4)
├─ Feature Extraction
├─ Quality Scoring
└─ Carrier Selection

PHASE 3: EMBEDDING & EXTRACTION (Weeks 5-6)
├─ Embedding Strategies
├─ Encryption Pipeline
└─ Extraction Pipeline

PHASE 4: VALIDATION & SECURITY (Weeks 7-9)
├─ Fidelity Testing
├─ Security Analysis
└─ Robustness Testing

PHASE 5: OPTIMIZATION (Weeks 10-11)
├─ Performance Tuning
├─ Capacity Optimization
└─ Operational Envelope

PHASE 6: PRODUCTION (Weeks 12-13)
├─ Integration Testing
├─ Documentation
└─ Release Preparation
```

### Detailed Phase Breakdown

#### PHASE 1: Foundation (Weeks 1-2)

**Objectives:**
- Clean codebase
- Implement core abstractions
- Establish test infrastructure

**Deliverables:**
- Clean git history
- Core module interfaces
- Unit test suite

**Key Files to Create/Fix:**
1. `core/interfaces.py` - Abstract base classes
2. `core/config.py` - Configuration system
3. `core/types.py` - Type definitions
4. Test files for all modules

**Success Criteria:**
- All imports work
- No circular dependencies
- 80%+ code coverage for core

---

#### PHASE 2: Carrier Intelligence (Weeks 3-4)

**Objectives:**
- Implement layer profiling
- Build feature extraction
- Create quality scoring

**Deliverables:**
- Layer profiles
- Quality scores
- Carrier allocation

**Key Files:**
1. Complete `carrier_intelligence/layer_profiler.py`
2. Complete `carrier_intelligence/feature_extractor.py`
3. Complete `carrier_intelligence/carrier_scheduler.py`
4. Complete `carrier_intelligence/quality_score.py`

**Success Criteria:**
- Profile generation works
- Quality scores in [0, 1]
- Allocation matches capacity

---

#### PHASE 3: Embedding & Extraction (Weeks 5-6)

**Objectives:**
- Implement embedding strategies
- Build encryption/decryption
- Create extraction pipeline

**Deliverables:**
- Working embedding
- Working extraction
- Encryption system

**Key Files:**
1. `embedding/intelligent_embedder.py` - Orchestrator
2. `embedding/strategies/sign_strategy.py` - Implementation
3. `crypto/` - Full crypto suite
4. `extraction/` - Full extraction

**Success Criteria:**
- Perfect recovery (BER=0) in clean conditions
- <5% BER with quantization noise
- Encryption verified

---

#### PHASE 4: Validation & Security (Weeks 7-9)

**Objectives:**
- Verify model fidelity
- Validate security properties
- Test robustness

**Deliverables:**
- Fidelity reports
- Security certificates
- Robustness analysis

**Key Experiments:**
1. Perplexity impact (< 2%)
2. Task accuracy (< 1% loss)
3. Statistical detection (KL < 0.05)
4. Noise robustness (σ < 0.002)

---

#### PHASE 5: Optimization (Weeks 10-11)

**Objectives:**
- Optimize performance
- Maximize capacity
- Define operational envelope

**Deliverables:**
- Optimized parameters
- Capacity-robustness tradeoff
- Operational guidelines

---

#### PHASE 6: Production (Weeks 12-13)

**Objectives:**
- Integration testing
- Documentation
- Release preparation

**Deliverables:**
- Production-ready codebase
- Complete documentation
- Release notes

---

## SECTION 5: QUALITY ASSURANCE GATES

### Gate 1: Code Quality (Week 2)

**Checklist:**
- ✓ No circular imports
- ✓ Code coverage ≥ 80%
- ✓ All type hints present
- ✓ PEP 8 compliant
- ✓ Docstrings complete

**Tools:**
- pytest, coverage.py
- mypy, black, flake8

### Gate 2: Functional Correctness (Week 4)

**Checklist:**
- ✓ Perfect recovery (BER=0) clean
- ✓ All strategies implemented
- ✓ Encryption verified
- ✓ Extraction works

**Tests:**
- Unit tests for each module
- Integration tests
- End-to-end tests

### Gate 3: Security (Week 6)

**Checklist:**
- ✓ No key leakage
- ✓ Authentication working
- ✓ Indistinguishable from quantization
- ✓ Extraction attack resistant

**Tests:**
- Crypto tests
- Attack simulations
- Statistical tests

### Gate 4: Model Fidelity (Week 8)

**Checklist:**
- ✓ PPL degradation < 2%
- ✓ Task accuracy maintained
- ✓ No perceptible differences
- ✓ All models pass

**Benchmarks:**
- Perplexity: WikiText-2
- MMLU: 5-shot accuracy
- GSM8K: 8-shot accuracy
- HellaSwag: accuracy

### Gate 5: Robustness (Week 10)

**Checklist:**
- ✓ Noise handling characterized
- ✓ Capacity-robustness tradeoff known
- ✓ Operational envelope defined
- ✓ Failure modes understood

**Tests:**
- Noise degradation curves
- Payload size experiments
- Multi-layer validation

### Gate 6: Production Readiness (Week 12)

**Checklist:**
- ✓ All tests pass
- ✓ Documentation complete
- ✓ No known issues
- ✓ Performance acceptable
- ✓ Reproducible results

---

## SECTION 6: DEPLOYMENT STRATEGY

### 6.1 Development Environment

**Setup:**
```bash
# Environment
Python 3.11+
PyTorch 2.2+
CUDA 12.0+
16GB+ GPU VRAM

# Dependencies
pip install -r requirements.txt

# Testing
pytest tests/
coverage run -m pytest
```

### 6.2 Repository Structure

```
nes-research-main/
├── nes-llm/
│   ├── src/
│   │   ├── core/                    # Core abstractions
│   │   ├── config/                  # Configuration
│   │   ├── profiling/               # Model & layer profiling
│   │   ├── residuals/               # Residual extraction
│   │   ├── carrier_intelligence/    # QACI system
│   │   ├── carrier_selection/       # Carrier selection
│   │   ├── embedding/               # Embedding strategies
│   │   ├── crypto/                  # Encryption
│   │   ├── quantization/            # Quantization handling
│   │   ├── extraction/              # Extraction pipeline
│   │   ├── steganalysis/            # Detection & security
│   │   ├── evaluation/              # Benchmarks & tests
│   │   ├── utils/                   # Utilities
│   │   └── main.py                  # Entry point
│   ├── tests/                       # Unit & integration tests
│   ├── configs/                     # Configuration files
│   ├── notebooks/                   # Analysis notebooks
│   ├── scripts/                     # Utility scripts
│   ├── data/                        # Data & profiles
│   ├── outputs/                     # Results & models
│   ├── requirements.txt             # Dependencies
│   ├── setup.py                     # Package setup
│   └── README.md                    # Documentation
└── docs/                            # Documentation
    ├── architecture.md
    ├── user_guide.md
    ├── api_reference.md
    └── examples/
```

### 6.3 CI/CD Pipeline

```
┌─────────────────┐
│ Commit to main  │
└────────┬────────┘
         │
         ▼
    ┌─────────────┐
    │ Lint Check  │ (black, flake8, mypy)
    └────┬────────┘
         │
         ▼
    ┌─────────────┐
    │ Unit Tests  │ (pytest, coverage ≥ 80%)
    └────┬────────┘
         │
         ▼
    ┌─────────────┐
    │ Integration │ (End-to-end tests)
    │   Tests     │
    └────┬────────┘
         │
         ▼
    ┌─────────────┐
    │ Security    │ (Crypto, attack simulation)
    │   Tests     │
    └────┬────────┘
         │
         ▼
    ┌─────────────┐
    │ Build       │ (Package creation)
    │ Package     │
    └────┬────────┘
         │
         ▼
    ┌─────────────┐
    │ Deploy to   │
    │ Staging     │
    └─────────────┘
```

### 6.4 Release Checklist

**Before Release:**
- ✓ All tests pass (100% success)
- ✓ Code review completed
- ✓ Documentation updated
- ✓ Security audit passed
- ✓ Performance benchmarks recorded
- ✓ Reproducibility verified
- ✓ Release notes written

**Release Steps:**
1. Tag release: `git tag v2.0`
2. Build: `python setup.py sdist bdist_wheel`
3. Document: Update README, CHANGELOG
4. Archive: Save to secure location
5. Notify: Inform stakeholders

---

## APPENDIX: Key Metrics & Benchmarks

### Expected Performance Targets

| Metric | Target | Acceptable | Critical |
|--------|--------|-----------|----------|
| **Clean Extraction** |
| BER | 0.0 | - | > 0.001 |
| Accuracy | 100% | - | < 99.9% |
| **With Quantization Noise** |
| BER @ σ=0.001 | < 0.02 | < 0.05 | > 0.1 |
| Accuracy @ σ=0.001 | > 98% | > 95% | < 90% |
| **Model Fidelity** |
| PPL degradation | < 1% | < 2% | > 3% |
| Task accuracy loss | < 0.5% | < 1% | > 2% |
| **Security** |
| KL divergence | < 0.02 | < 0.05 | > 0.1 |
| Detection accuracy | < 52% | < 55% | > 60% |
| **Capacity** |
| Bits/layer | 5K-10K | 3K-15K | < 1K |
| Total payload | 50-100K | 30-150K | < 20K |

### Benchmark Models

| Model | Params | Layers | Test Set |
|-------|--------|--------|----------|
| TinyLlama | 1.1B | 22 | - (dev) |
| Llama-3-8B | 8B | 32 | WikiText-2, MMLU |
| Mistral-7B | 7B | 32 | (optional) |
| Qwen-2.5-7B | 7B | 32 | (optional) |

---

**Document Status:** FINAL - READY FOR IMPLEMENTATION  
**Next Step:** Begin Phase 1 - Code Cleanup & Architecture  
**Estimated Timeline:** 13 weeks to production readiness
