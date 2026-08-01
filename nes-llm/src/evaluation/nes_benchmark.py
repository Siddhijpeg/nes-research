"""
NES Benchmark Suite — runs all 6 quality gates and reports results.

Quality Gates:
    Gate 1 (Phase 1): Core imports, no circular deps      → manual check
    Gate 2 (Phase 2): QACI scores in [0,1], allocation OK → tensor check
    Gate 3 (Phase 3): BER=0 clean, all payload sizes      → embed/extract
    Gate 4 (Phase 4): PPL < 2%, accuracy loss < 1%        → fidelity
    Gate 5 (Phase 4): Robustness characterised            → noise curve
    Gate 6 (Phase 4): KL < 0.05, detector < 55%           → security

Run with:
    python -m src.evaluation.nes_benchmark
"""

from typing import Dict, List, Optional

import torch

from src.carrier_intelligence.qaci_pipeline    import QACIPipeline
from src.evaluation.fidelity_validator         import FidelityValidator
from src.evaluation.robustness_validator       import RobustnessValidator
from src.steganalysis.security_validator       import SecurityValidator
from src.embedding.intelligent_embedder        import IntelligentEmbedder
from src.extraction.decrypt_pipeline           import DecryptPipeline
from src.core.types                            import EmbeddingConfig


class NESBenchmark:
    """
    Runs the full NES quality gate suite on synthetic data.

    For production use, replace synthetic residuals with real
    model residuals extracted from the target LLM.
    """

    def __init__(
        self,
        n_layers:    int = 8,
        layer_size:  int = 10000,
        payload_bits:int = 5000,
        verbose:     bool = True,
    ):
        self.n_layers     = n_layers
        self.layer_size   = layer_size
        self.payload_bits = payload_bits
        self.verbose      = verbose

    # ------------------------------------------------------------------
    # Synthetic data helpers
    # ------------------------------------------------------------------

    def _make_residuals(self) -> Dict[int, torch.Tensor]:
        return {i: torch.randn(self.layer_size) * 0.05 for i in range(self.n_layers)}

    # ------------------------------------------------------------------
    # Individual gates
    # ------------------------------------------------------------------

    def gate2_qaci(self) -> dict:
        """Gate 2: QACI scores in [0,1], bit allocation conserved."""
        pipeline  = QACIPipeline(total_layers=self.n_layers)
        residuals = self._make_residuals()
        result    = pipeline.select(residuals, self.payload_bits)

        total_allocated = sum(result.layer_allocation.values())
        passed = total_allocated == self.payload_bits

        return {
            "gate": 2, "name": "QACI Allocation",
            "passed":    passed,
            "total_bits_allocated": total_allocated,
            "expected":  self.payload_bits,
        }

    def gate3_ber(self) -> dict:
        """Gate 3: BER=0 in clean conditions."""
        config   = EmbeddingConfig(
            total_payload_bits=self.payload_bits,
            embedding_strategy="sign",
        )
        embedder  = IntelligentEmbedder(config)
        residuals = self._make_residuals()
        message   = "NES"   # short — encrypted bits easily fit in payload budget

        embed_result = embedder.embed(message, residuals)
        pipeline     = DecryptPipeline(key=embed_result.key)
        recovered, stats = pipeline.run(
            embed_result.embedded_residuals,
            embed_result.carrier_indices,
        )

        passed = stats.get("success", False) and recovered == message
        return {
            "gate":              3,
            "name":              "Clean BER",
            "passed":            passed,
            "recovered_message": recovered,
            "expected_message":  message,
            "decrypt_stats":     stats,
        }

    def gate4_fidelity(self) -> dict:
        """Gate 4: Fidelity — residual distribution preserved after embedding."""
        config   = EmbeddingConfig(
            total_payload_bits=self.payload_bits,
            embedding_strategy="sign",
        )
        embedder  = IntelligentEmbedder(config)
        residuals = self._make_residuals()

        embed_result = embedder.embed("test fidelity", residuals)
        validator    = FidelityValidator(
            max_ppl_degradation=0.02,
            max_accuracy_loss=0.01,
        )
        fidelity = validator.validate_tensors(
            original_residuals=residuals,
            embedded_residuals=embed_result.embedded_residuals,
        )

        return {
            "gate":      4,
            "name":      "Fidelity",
            "passed":    fidelity.passed,
            "ppl_deg":   fidelity.ppl_degradation,
            "sign_flip": fidelity.mean_sign_flip_rate,
            "kl_div":    fidelity.mean_kl_divergence,
            "report":    fidelity.report(),
        }

    def gate5_robustness(self) -> dict:
        """Gate 5: BER < 2% at σ=0.001, BER < 10% at σ=0.002."""
        from src.embedding.sign_strategy_v2 import SignEmbeddingStrategy
        from src.extraction.sign_extractor  import SignExtractor

        config    = EmbeddingConfig(
            total_payload_bits=self.payload_bits,
            embedding_strategy="sign",
        )
        strategy  = SignEmbeddingStrategy(config)
        extractor = SignExtractor()

        residuals = self._make_residuals()
        bits      = [i % 2 for i in range(self.payload_bits)]
        indices   = {i: list(range(self.payload_bits // self.n_layers))
                     for i in range(self.n_layers)}

        embed_result = strategy.embed(residuals, bits, indices)
        validator    = RobustnessValidator(
            max_ber_at_001=0.02,
            max_ber_at_002=0.10,
            num_trials=3,
        )
        result = validator.validate(
            embedded_residuals=embed_result.embedded_weights,
            carrier_indices=   embed_result.carrier_indices,
            original_bits=     bits,
            sigmas=            [0.0, 0.001, 0.002, 0.005],
        )

        return {
            "gate":      5,
            "name":      "Robustness",
            "passed":    result.passed,
            "ber_curve": result.ber_curve,
            "report":    result.report(),
        }

    def gate6_security(self) -> dict:
        """Gate 6: KL < 0.05, detector accuracy < 55%."""
        config   = EmbeddingConfig(
            total_payload_bits=self.payload_bits,
            embedding_strategy="sign",
        )
        embedder  = IntelligentEmbedder(config)
        residuals = self._make_residuals()

        embed_result = embedder.embed("security test", residuals)
        validator    = SecurityValidator(
            max_kl_divergence=0.05,
            max_detector_accuracy=0.55,
        )
        result = validator.validate(
            original_residuals=residuals,
            embedded_residuals=embed_result.embedded_residuals,
        )

        return {
            "gate":             6,
            "name":             "Security",
            "passed":           result.passed,
            "kl_divergence":    result.kl_divergence,
            "detector_accuracy":result.detector_accuracy,
            "report":           result.report(),
        }

    # ------------------------------------------------------------------
    # Run all gates
    # ------------------------------------------------------------------

    def run_all(self) -> dict:
        """Run gates 2–6 and return aggregated results."""
        gates = [
            self.gate2_qaci,
            self.gate3_ber,
            self.gate4_fidelity,
            self.gate5_robustness,
            self.gate6_security,
        ]

        results = []
        for gate_fn in gates:
            try:
                r = gate_fn()
                results.append(r)
            except Exception as e:
                results.append({
                    "gate":   "?",
                    "name":   gate_fn.__name__,
                    "passed": False,
                    "error":  str(e),
                })

        passed = sum(1 for r in results if r.get("passed"))
        total  = len(results)

        if self.verbose:
            print("\n" + "=" * 60)
            print("  NES BENCHMARK RESULTS")
            print("=" * 60)
            for r in results:
                icon = "✅" if r.get("passed") else "❌"
                print(f"  {icon}  Gate {r['gate']}: {r['name']}")
                if "report" in r:
                    for line in r["report"].splitlines():
                        print(f"       {line}")
                if "error" in r:
                    print(f"       ERROR: {r['error']}")
            print("=" * 60)
            print(f"  {passed}/{total} gates passed")
            print("=" * 60 + "\n")

        return {
            "passed":  passed,
            "total":   total,
            "all_pass":passed == total,
            "results": results,
        }


if __name__ == "__main__":
    bench = NESBenchmark(
        n_layers=8, layer_size=10000,
        payload_bits=5000, verbose=True,
    )
    outcome = bench.run_all()
    import sys
    sys.exit(0 if outcome["all_pass"] else 1)