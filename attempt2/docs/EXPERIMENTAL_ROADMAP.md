# Neural-Entropic Steganography: Experimental Roadmap
## Phase-Based Research Plan for Detectability-Resistant Embedding

---

## Current Problem Statement

**Baseline Metrics (from updated code):**
- Neural Detector Accuracy: **99%** (should be ~50% = random)
- Recovery BER: **0.0** (perfect) when noise σ ≤ 0 only
- Noise Robustness (σ=0.01): **BER ~0.44** (fails immediately)
- Quantization Robustness (INT4 re-quant): **BER ~0.50** (complete failure)

**Root Cause:** Adaptive margins + quality-based carrier selection create statistical fingerprints that neural detectors exploit easily.

---

# PHASE 1: Distribution Matching & Constraint Verification
**Duration: 1-2 weeks | Effort: 200 code lines**

## 1.1 Implement Distribution Matcher

**File:** `src/embedding/distribution_matcher.py`

**Goal:** Force stego residuals to exactly match clean residual distribution.

### Theory

After embedding, stego residuals `R_stego[i] = R_clean[i] + delta[i]` where delta is the embedding perturbation.

**Problem:** Delta is correlated with quality_score → creates detectable pattern.

**Solution:** Iteratively adjust margins until stego distribution matches clean distribution on multiple statistics:

```
Minimize: D(P_stego, P_clean)
where D is divergence between:
  - Mean
  - Variance
  - Skewness
  - Kurtosis
  - Percentile ranges (p25, p50, p75)
```

### Implementation Steps

```python
class DistributionMatcher:
    """
    Post-embedding correction to match distributions.
    """
    
    def __init__(self, stats_to_match=None):
        # Default: match mean, variance, skewness, kurtosis
        self.stats = stats_to_match or [
            'mean', 'std', 'skewness', 'kurtosis'
        ]
        self.tolerance = 1e-4
    
    def compute_stats(self, tensor):
        """
        Compute distribution statistics.
        
        Returns: dict with mean, std, skew, kurt
        """
        flat = tensor.flatten()
        
        mean = flat.mean()
        std = flat.std()
        
        # Skewness: E[(x - mean)^3] / std^3
        centered = flat - mean
        skewness = (centered**3).mean() / (std**3 + 1e-8)
        
        # Kurtosis: E[(x - mean)^4] / std^4 - 3
        kurtosis = (centered**4).mean() / (std**4 + 1e-8) - 3
        
        return {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min': flat.min(),
            'max': flat.max(),
        }
    
    def divergence_loss(self, stego_stats, clean_stats):
        """
        Compute divergence between distributions.
        """
        loss = 0
        
        # Mean matching
        loss += (stego_stats['mean'] - clean_stats['mean'])**2
        
        # Variance matching (penalize std ratio)
        ratio = stego_stats['std'] / (clean_stats['std'] + 1e-8)
        loss += (ratio - 1.0)**2
        
        # Skewness matching
        loss += (stego_stats['skewness'] - clean_stats['skewness'])**2
        
        # Kurtosis matching
        loss += (stego_stats['kurtosis'] - clean_stats['kurtosis'])**2
        
        return loss
    
    def correct_margins(self, stego_profiles, clean_profiles):
        """
        Adjust margins per element to match distribution.
        
        Iterative process:
        1. Compute stego stats
        2. Compare to clean stats
        3. Adjust margins proportionally
        4. Re-embed
        5. Repeat until convergence
        """
        
        corrections_made = 0
        
        for stego_prof, clean_prof in zip(stego_profiles, clean_profiles):
            
            stego_residual = stego_prof['residual']
            clean_residual = clean_prof['residual']
            
            stego_stats = self.compute_stats(stego_residual)
            clean_stats = self.compute_stats(clean_residual)
            
            # Check if correction needed
            div_loss = self.divergence_loss(stego_stats, clean_stats)
            
            if div_loss > self.tolerance:
                
                # Compute correction factor
                std_ratio = (
                    clean_stats['std'] / 
                    (stego_stats['std'] + 1e-8)
                )
                
                # Adjust: scale stego residual down, then re-center
                adjusted = (
                    (stego_residual - stego_stats['mean']) 
                    * std_ratio 
                    + clean_stats['mean']
                )
                
                stego_prof['residual'] = adjusted
                corrections_made += 1
        
        return corrections_made
    
    def validate(self, stego_profiles, clean_profiles, threshold=0.05):
        """
        Check if distribution matching succeeded.
        
        Returns: dict with match percentages
        """
        
        results = {
            'mean_match': 0,
            'std_match': 0,
            'skew_match': 0,
            'kurt_match': 0,
            'all_match': True,
        }
        
        for stego_prof, clean_prof in zip(stego_profiles, clean_profiles):
            stego_stats = self.compute_stats(stego_prof['residual'])
            clean_stats = self.compute_stats(clean_prof['residual'])
            
            # Percent difference
            mean_diff = abs(
                (stego_stats['mean'] - clean_stats['mean']) 
                / (clean_stats['mean'] + 1e-8)
            )
            
            if mean_diff < threshold:
                results['mean_match'] += 1
            else:
                results['all_match'] = False
            
            # Similar for std, skew, kurt...
        
        return results
```

### Expected Outcome

**Before DM:**
- Detector accuracy: 99%
- KL divergence: 0.0053 (high)
- Entropy shift: 0.11 (high)

**After DM:**
- Detector accuracy: **65-75%** (better but not yet good)
- KL divergence: **0.0001** (matched)
- Entropy shift: **0.001** (matched)

### Success Metric

If detector accuracy drops to <65%, proceed to Phase 2.

---

# PHASE 2: Adversarial Margin Adaptation
**Duration: 2-3 weeks | Effort: 500 lines | Prerequisite: Phase 1**

## 2.1 Problem: Quality-Based Margins Are Detectable

The issue: Positions with high quality_score get large margins → creates a distribution in margins that's **correlated with detector features**.

## 2.2 Solution: Learn Margins from Detector Feedback

```python
class AdversarialMarginOptimizer:
    """
    Use neural detector as a loss function to find
    margin patterns that minimize detectability.
    """
    
    def __init__(self, detector, learning_rate=1e-3):
        self.detector = detector  # Trained neural detector
        self.lr = learning_rate
        self.margin_optimizer = None
    
    def compute_adaptive_margins_v2(
        self,
        residual,
        quality_scores,
        positions,
        detector_loss_weight=0.5,
    ):
        """
        Learn margins that:
        1. Preserve carrier quality (maintain recovery)
        2. Minimize detector confidence (fool detector)
        
        Loss = α * reconstruction_loss + β * detector_loss
        """
        
        # Initialize margins from quality (baseline)
        global_scale = residual.std() * self.alpha
        quality_norm = (
            (quality_scores - quality_scores.min()) 
            / (quality_scores.max() - quality_scores.min() + 1e-8)
        )
        margins = global_scale * quality_norm
        
        # Make margins learnable
        margins_param = torch.nn.Parameter(
            margins.clone().detach(),
            requires_grad=True
        )
        
        # Optimizer for margins
        margin_opt = torch.optim.Adam(
            [margins_param],
            lr=self.lr,
        )
        
        # Optimization loop
        for iteration in range(100):
            
            margin_opt.zero_grad()
            
            # Embed with current margins
            stego = self._embed_with_margins(
                residual,
                margins_param,
                positions,
                bits=None,  # Use existing bits
            )
            
            # Detector feedback
            detector_output = self.detector(
                self._extract_features(stego)
            )  # → [0, 1], higher = more stego-like
            
            detector_loss = detector_output.mean()  # Want to minimize this
            
            # Recovery quality (margins shouldn't change meaning)
            recovery_bits = self._extract_bits_with_margins(
                stego,
                margins_param,
                positions,
            )
            recovery_loss = (
                1 - accuracy(recovery_bits, original_bits)
            )
            
            # Combined loss
            total_loss = (
                detector_loss_weight * detector_loss
                + (1 - detector_loss_weight) * recovery_loss
            )
            
            total_loss.backward()
            margin_opt.step()
            
            if iteration % 20 == 0:
                print(
                    f"Iter {iteration}: "
                    f"detector_loss={detector_loss:.4f}, "
                    f"recovery_loss={recovery_loss:.4f}"
                )
        
        return margins_param.detach()
```

### Integration with IntelligentEmbedder

```python
# In IntelligentEmbedder.embed()

# OLD: margins = self.margin.compute(residual, quality_scores)

# NEW:
if self.use_adversarial:
    margins = self.adversarial_optimizer.compute_adaptive_margins_v2(
        residual,
        quality_scores,
        positions,
        detector_loss_weight=0.3,  # Balance recovery vs detectability
    )
else:
    margins = self.margin.compute(residual, quality_scores)
```

### Expected Outcome

**Before Adversarial:**
- Detector accuracy: ~65%

**After Adversarial:**
- Detector accuracy: **50-55%** (approaching random)
- Recovery BER: **<0.001** (still perfect)
- Training time: ~30 seconds per embedding

### Success Metric

Detector accuracy should drop to <55%.

---

# PHASE 3: Robustness Hardening
**Duration: 2 weeks | Effort: 300 lines | Prerequisite: Phase 2**

## 3.1 Problem: Noise & Requantization Destroy Payload

Current results:
- σ=0.01 noise → BER 0.44 (fails)
- Re-quantize to INT4 → BER 0.50 (fails)

## 3.2 Solution A: Error Correcting Codes (ECC)

Add Reed-Solomon or Turbo codes before embedding:

```python
class ECCPayloadEncoder:
    """
    Encode payload with error-correcting code before embedding.
    """
    
    def __init__(self, ecc_type='reed_solomon', redundancy=0.3):
        self.ecc_type = ecc_type
        self.redundancy = redundancy
        # Initialize ecc encoder (use reedsolo library)
    
    def encode(self, bits):
        """
        Encode bits with ECC.
        
        Input: N bits
        Output: N / (1 - redundancy) bits (10% → 11 bits per 10)
        """
        # Convert to bytes, encode with RS(255, 223) = 32 parity bytes
        # Returns: original_bits + parity_bits
        pass
    
    def decode(self, bits, corrupted_positions=None):
        """
        Decode bits, correcting errors.
        
        If exactly where corruption happened, can correct up to
        t errors where 2t ≤ parity_bytes.
        """
        pass
```

### Expected Outcome

With RS(255, 223) = 32 parity bytes per 223 data bytes:
- σ=0.01 noise → BER 0.02-0.05 (recoverable!)
- Re-quantization to INT4 → BER 0.01-0.03 (recoverable!)

## 3.2 Solution B: Iterative Refinement

For each carrier, embed multiple times with different positions:

```python
class RobustEmbedder:
    """
    Embed same payload bits multiple times across different
    positions, then take majority vote on recovery.
    """
    
    def embed_robust(
        self,
        residual,
        bits,
        positions,
        repetitions=3,
    ):
        """
        Embed each bit in `repetitions` different positions.
        """
        
        stego = residual.clone()
        
        for rep in range(repetitions):
            
            # Select different positions for each repetition
            positions_rep = self.select_diverse_positions(
                residual,
                len(bits),
                exclude=positions,
            )
            
            # Embed same bits in new positions
            for bit, pos in zip(bits, positions_rep):
                margin = ...
                stego[pos] = (
                    abs(stego[pos]) + margin
                ) if bit == 1 else -(abs(stego[pos]) + margin)
        
        return stego
    
    def extract_robust(self, stego, positions_all):
        """
        Extract bits from all positions, take majority vote.
        """
        
        extracted_all = []
        
        for positions in positions_all:  # for each repetition
            extracted = []
            for pos in positions:
                extracted.append(1 if stego[pos] >= 0 else 0)
            extracted_all.append(extracted)
        
        # Majority vote
        bits_final = [
            sum(extracted_all[rep][i] for rep in range(len(extracted_all)))
            > len(extracted_all) // 2
            for i in range(len(extracted_all[0]))
        ]
        
        return bits_final
```

### Integration

```python
# In IntelligentEmbedder.embed()

if self.use_ecc:
    payload_encoded = self.ecc_encoder.encode(payload_bits)
else:
    payload_encoded = payload_bits

if self.use_repetition:
    # Embed with multiple repetitions per bit
    stego = self.robust_embedder.embed_robust(...)
else:
    stego = self.embedder.embed(...)
```

### Expected Outcome

**With ECC + Repetition:**
- σ=0.01 noise → BER <0.01 ✅
- INT4 re-quantization → BER <0.01 ✅
- Capacity loss: ~2x (due to ECC + repetition)

---

# PHASE 4: Payload Extraction Pipeline
**Duration: 1 week | Effort: 150 lines | Prerequisite: Phase 2**

## 4.1 Implement Extraction

**File:** `src/embedding/payload_extractor.py`

```python
class PayloadExtractor:
    """
    Extract embedded payload from stego residuals.
    """
    
    def __init__(self, ecc_decoder=None):
        self.ecc_decoder = ecc_decoder
    
    def extract_bits(
        self,
        stego_profiles,
        allocation_plan,
        layer_metadata,
    ):
        """
        Reverse the embedding process:
        1. Locate each carrier
        2. Extract signs from selected positions
        3. Recover bits
        4. Reorder according to allocation
        5. Decode ECC if used
        """
        
        extracted_bits = []
        
        for allocation in allocation_plan:
            
            if allocation.allocated_bits == 0:
                continue
            
            layer_id = allocation.layer
            module = allocation.module
            
            # Find stego profile
            profile = [
                p for p in stego_profiles
                if p['layer'] == layer_id
                and p['module'] == module
            ][0]
            
            stego_residual = profile['residual']
            
            # Find metadata
            metadata = [
                m for m in layer_metadata
                if m.layer == layer_id
                and m.module == module
            ][0]
            
            positions = metadata.positions.flatten()
            
            # Extract from signs
            flat = stego_residual.flatten()
            for pos in positions:
                bit = 1 if flat[pos] >= 0 else 0
                extracted_bits.append(bit)
        
        # Decode ECC if used
        if self.ecc_decoder:
            extracted_bits = self.ecc_decoder.decode(
                extracted_bits
            )
        
        return extracted_bits
    
    def verify_recovery(self, original, extracted):
        """
        Compute BER, accuracy.
        """
        
        errors = sum(
            1 for a, b in zip(original, extracted)
            if a != b
        )
        
        ber = errors / len(original)
        accuracy = 1 - ber
        
        return {
            'errors': errors,
            'ber': ber,
            'accuracy': accuracy,
        }
```

---

# PHASE 5: Comprehensive Evaluation
**Duration: 2 weeks | Effort: 400 lines**

## 5.1 Create Unified Test Suite

**File:** `src/evaluation/comprehensive_robustness_test.py`

```python
class ComprehensiveRobustnessTest:
    """
    Evaluate embedding across:
    - Detectability (neural detector)
    - Capacity (payload size)
    - Recovery (BER under various conditions)
    - Robustness (noise, quantization, etc.)
    """
    
    def __init__(self, model_id, detector_path):
        self.embedder = IntelligentEmbedder(use_adversarial=True)
        self.detector = load_detector(detector_path)
        self.extractor = PayloadExtractor(ecc_decoder=...)
        self.model_id = model_id
    
    def run_full_test(
        self,
        payload_sizes=[1000, 5000, 10000, 50000, 100000],
        noise_levels=[0, 0.001, 0.01, 0.1],
        quantization_types=['nf4', 'int8', 'int4'],
    ):
        """
        Run embedding → contamination → extraction for all configs.
        """
        
        results = {}
        
        for payload_size in payload_sizes:
            
            payload = random_bits(payload_size)
            
            # 1. Embed
            result = self.embedder.embed(fp16, nf4, payload)
            
            # 2. Check detectability
            detector_score = self._compute_detector_score(
                result.stego_profiles
            )
            
            results[f'detect_{payload_size}'] = detector_score
            
            # 3. Test noise robustness
            for sigma in noise_levels:
                
                stego_noisy = self._add_noise(
                    result.stego_profiles,
                    sigma,
                )
                
                extracted = self.extractor.extract_bits(
                    stego_noisy,
                    result.allocation_plan,
                    result.layer_metadata,
                )
                
                ber = compute_ber(payload, extracted)
                
                results[
                    f'ber_payload{payload_size}_noise{sigma}'
                ] = ber
            
            # 4. Test quantization robustness
            for quant_type in quantization_types:
                
                stego_requant = self._requantize(
                    result.stego_profiles,
                    quant_type,
                )
                
                extracted = self.extractor.extract_bits(
                    stego_requant,
                    result.allocation_plan,
                    result.layer_metadata,
                )
                
                ber = compute_ber(payload, extracted)
                
                results[
                    f'ber_payload{payload_size}_quant{quant_type}'
                ] = ber
        
        return results
    
    def _compute_detector_score(self, stego_profiles):
        """
        Average detector confidence across all profiles.
        """
        
        scores = []
        
        for profile in stego_profiles:
            residual = profile['residual']
            features = extract_features(residual)
            score = self.detector(features).mean()
            scores.append(score)
        
        return np.mean(scores)
```

## 5.2 Generate Report

```python
def generate_test_report(results):
    """
    Create markdown report of all results.
    """
    
    report = """
# NES Embedding Robustness Test Report

## Detectability
"""
    
    for key, value in results.items():
        if 'detect_' in key:
            report += f"\n- {key}: {value:.4f}"
    
    report += "\n\n## Recovery Quality\n"
    
    for key, value in results.items():
        if 'ber_' in key:
            report += f"\n- {key}: BER={value:.4f}"
    
    return report
```

### Expected Results

```
Detectability:
  - Payload 1000: detector_score = 0.51 (random)
  - Payload 50000: detector_score = 0.52 (random)
  ✅ PASS: detector_score ≈ 0.50 across all sizes

Recovery Quality (with ECC):
  - Noise σ=0.001: BER = 0.000
  - Noise σ=0.01: BER = 0.002
  - Noise σ=0.1: BER = 0.05
  - INT4 re-quant: BER = 0.001
  ✅ PASS: BER < 0.01 for realistic attacks

Capacity:
  - Payload 100k: ~95% of bits recovered
  ✅ PASS: high capacity maintained
```

---

# PHASE 6: Multi-Model Evaluation
**Duration: 1.5 weeks | Effort: 200 lines | Prerequisite: Phase 5**

## 6.1 Sweep Across Model Families

```python
class MultiModelEvaluation:
    
    MODELS = [
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'meta-llama/Llama-2-7b',  # If available
        'mistral-community/Mistral-7B-v0.1',
        'Qwen/Qwen2.5-7B',
    ]
    
    def evaluate_all(self):
        
        results = {}
        
        for model_id in self.MODELS:
            
            print(f"\n=== {model_id} ===")
            
            fp16 = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
            )
            
            nf4 = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=...,
            )
            
            tester = ComprehensiveRobustnessTest(
                model_id,
                self.detector_path,
            )
            
            test_results = tester.run_full_test(
                payload_sizes=[10000],
                noise_levels=[0, 0.01],
            )
            
            results[model_id] = test_results
        
        return results
```

---

# Summary: Expected Improvements

| Metric | Baseline | Phase 2 | Phase 3 | Phase 5+ |
|--------|----------|---------|---------|----------|
| **Detectability** | 99% | 52% | 51% | **50%** ✅ |
| **Capacity** | 500k | 500k | 250k* | 250k* |
| **BER (σ=0)** | 0.0 | 0.0 | 0.0 | 0.0 |
| **BER (σ=0.01)** | 0.44 ❌ | 0.35 | 0.002 ✅ | <0.001 ✅ |
| **BER (INT4)** | 0.50 ❌ | 0.48 | 0.005 ✅ | <0.001 ✅ |

*With ECC overhead; raw capacity still 500k.

---

# Immediate Next Steps (Week 1)

1. **Implement `DistributionMatcher`** (Phase 1)
   - Copy template from this document
   - Test against TinyLlama with 10k payload
   - Target: Detector accuracy <70%

2. **Create `comprehensive_robustness_test.py`** (Phase 5 prep)
   - Build test framework
   - Run baseline on current IntelligentEmbedder
   - Record all metrics

3. **Prepare `AdversarialMarginOptimizer`** (Phase 2)
   - Implement optimizer
   - Don't run yet; validate syntax

**Success Criteria:** Phase 1 + partial Phase 2 by end of week = detector accuracy drops from 99% → <70%.
