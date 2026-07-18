# NES Embedding Method - Quick Reference

## What You're Using (ONE SENTENCE)
**Carrier Intelligence Framework with Adaptive Margin-Based Sign Embedding** — a multi-stage system that selects high-quality residual positions and embeds bits using quality-adaptive margins to balance stealth vs robustness.

---

## The Pipeline (5 Stages)

```
[FP16 Model] + [NF4 Model]
       ↓
   Extract Residuals (fp16 - nf4_dequant)
       ↓
   Stage 1: Feature Extraction (9 dimensions per element)
       ├─ Information: magnitude, variance, std
       ├─ Robustness: quantization_error, stability, robustness
       └─ Distortion: cost, zscore, distortion
       ↓
   Stage 2: Quality Scoring (learned weighted combination)
       ↓
   Stage 3: Allocation (Hamilton method: proportional to quality)
       ├─ Scores carriers by quality
       ├─ Normalizes with non-linear power (γ=2.5)
       └─ Distributes payload bits fairly
       ↓
   Stage 4: Carrier Selection (Top-K by objective score)
       ↓
   Stage 5: ADAPTIVE MARGIN EMBEDDING
       ├─ Compute per-element margin = α × std × quality_norm
       ├─ For bit=1: embed +(|value| + margin)
       └─ For bit=0: embed -(|value| + margin)
       ↓
   Output: Stego Residuals
```

---

## Core Embedding (Lines 698-736 of intelligent_embedder.py)

```python
# For each selected position:
delta = margin[pos]                    # Quality-based delta
value = abs(residual[pos])             # Current magnitude
if bit == 1:
    residual_stego[pos] = +(value + delta)
else:
    residual_stego[pos] = -(value + delta)
```

**Key insight:** Margin is adaptive (larger for high-quality positions).

---

## Current Results

| Metric | Value | Status |
|--------|-------|--------|
| **Capacity** | 500k bits | ✅ |
| **Recovery BER (clean)** | 0.0 | ✅ |
| **Detectability** | 99% | ❌ FAIL |
| **Noise σ=0.01** | BER=0.44 | ❌ FAIL |
| **INT4 requant** | BER=0.50 | ❌ FAIL |

---

## Why It Fails

1. **Margin artifacts:** High-quality positions get large perturbations → creates **skewness shift**
2. **Sign bias:** Forced sign creates **distribution asymmetry**
3. **Selection pattern:** Quality-based selection → **non-uniform feature distribution**

Neural detector exploits these signatures → 99% accuracy.

---

## What Needs to Change

### Problem 1: Detectability (Priority 1)
- **Cause:** Static adaptive margins leak information to detector
- **Fix:** Implement distribution matching (Phase 1) + adversarial margin learning (Phase 2)
- **Expected result:** Detector accuracy 99% → 50% (random)

### Problem 2: Robustness (Priority 2)
- **Cause:** Single-level embedding, no error correction
- **Fix:** Add Reed-Solomon ECC (Phase 3) + iterative refinement
- **Expected result:** σ=0.01 BER: 0.44 → <0.01

### Problem 3: Extraction (Priority 3)
- **Cause:** No extraction pipeline implemented
- **Fix:** Implement `PayloadExtractor` (Phase 4)
- **Expected result:** Enable end-to-end testing

---

## Implementation Status

| Component | Status | Lines |
|-----------|--------|-------|
| IntelligentEmbedder | ✅ | 834 |
| AdaptiveMarginController | ✅ | 83 |
| CarrierScheduler | ✅ | 340 |
| CarrierSelector | ✅ | 71 |
| DistributionMatcher | ❌ EMPTY | 0 |
| AdversarialMarginOptimizer | ❌ EMPTY | 0 |
| PayloadExtractor | ❌ EMPTY | 0 |
| ErrorCorrectingCodec | ❌ EMPTY | 0 |

**Total working:** ~1500 lines  
**Total needed:** +1500 lines to fix detectability & robustness

---

## Experiment Plan (6 Phases)

| Phase | Goal | Duration | Success Metric |
|-------|------|----------|-----------------|
| 1 | Distribution Matching | 1-2 weeks | Detector acc. <70% |
| 2 | Adversarial Margins | 2-3 weeks | Detector acc. <55% |
| 3 | Noise Robustness (ECC) | 2 weeks | BER(σ=0.01) <0.01 |
| 4 | Extraction Pipeline | 1 week | E2E working |
| 5 | Comprehensive Testing | 2 weeks | All metrics pass |
| 6 | Multi-Model Sweep | 1.5 weeks | Generalization ✅ |

**Total effort:** 10-12 weeks  
**Critical path:** Phase 1 → Phase 2 → Phase 3

---

## Files to Modify/Create

### CRITICAL (Must implement):
1. **`src/embedding/distribution_matcher.py`** (250 lines)
2. **`src/carrier_intelligence/adversarial_margin.py`** (500 lines)
3. **`src/embedding/payload_extractor.py`** (150 lines)

### IMPORTANT (Should implement):
4. **`src/embedding/ecc_codec.py`** (400 lines)
5. **`src/evaluation/comprehensive_robustness_test.py`** (400 lines)

### NICE-TO-HAVE:
6. Multi-model sweep variants
7. Visualization & plotting utilities

---

## Key Configuration Parameters

```python
# IntelligentEmbedder
alpha = 0.25                    # Margin scale factor
                                # Increase → larger margins, more robust, more detectable
                                # Decrease → smaller margins, less detectable, less robust

# CarrierScheduler
gamma = 2.5                     # Non-linearity in allocation
                                # Increase → concentrate on best carriers
                                # Decrease → spread evenly

# AdversarialMarginOptimizer (NEW)
detector_loss_weight = 0.3      # Balance detectability vs recovery
                                # Higher → prioritize stealth
                                # Lower → prioritize robustness

# ErrorCorrectingCodec (NEW)
redundancy = 0.3                # 30% overhead for error correction
                                # Can correct up to 15% errors
```

---

## Testing Checklist

```
Phase 1: Distribution Matcher
☐ Implement compute_stats()
☐ Implement divergence_loss()
☐ Implement correct_margins()
☐ Test on TinyLlama 10k payload
☐ Verify detector acc drops to <70%

Phase 2: Adversarial Optimization
☐ Implement AdversarialMarginOptimizer
☐ Integrate with IntelligentEmbedder
☐ Test on TinyLlama 10k payload
☐ Verify detector acc drops to <55%
☐ Verify BER still <0.01

Phase 3: ECC
☐ Implement reed_solomon encoder/decoder
☐ Test noise robustness σ=0.01
☐ Verify BER <0.01

Phase 4: Extraction
☐ Implement PayloadExtractor
☐ End-to-end test embed → extract
☐ Verify BER <0.001

Phase 5: Comprehensive Testing
☐ Run all payload sizes (1k-100k)
☐ Run all noise levels (0-0.1)
☐ Run all quantization types
☐ Generate report

Phase 6: Multi-model
☐ Test on Llama-2-7b
☐ Test on Mistral-7B
☐ Test on Qwen-7B
☐ Verify generalization
```

---

## Success Criteria (Final)

✅ **Detectability:** Detector accuracy ≤ 52% (indistinguishable from random)  
✅ **Capacity:** 50k+ bits per model  
✅ **Recovery:** BER < 0.001 without corruption, < 0.01 with σ=0.01 noise  
✅ **Robustness:** Survives INT4 re-quantization, Gaussian noise σ<0.1  
✅ **Generalization:** Works on at least 3 different model families  
✅ **Paper-ready:** Clear ablation studies showing each component's contribution

---

## One Page Executive Summary

You are implementing a **neural weight steganography system** that:

1. **Hides** cryptographic payloads in LLM weight residuals (FP16 → NF4 quantization loss)
2. **Selects** high-quality positions using 9-feature deep analysis (avoid critical weights)
3. **Encodes** bits as residual sign + adaptive magnitude (trade-off quality-robustness)
4. **Allocates** payload fairly across layers using fair-division math (Hamilton method)
5. **Detects** as adversarially as possible (current: 99% accuracy ← PROBLEM)

**Current blockers:**
- Statistical signatures leak info to neural detectors
- No error correction for realistic channel conditions
- No extraction pipeline for round-trip testing

**Solution path:** Add distribution matching → adversarial margins → ECC → comprehensive testing.

**Timeline:** 10-12 weeks to production-ready system.

