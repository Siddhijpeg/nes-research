"""Tests for Phase 9 — Experiment suite."""

import torch
from src.evaluation.perplexity_experiment    import PerplexityExperiment, PPLResult
from src.evaluation.task_accuracy_experiment import TaskAccuracyExperiment, TaskResult


class MockModel:
    """Minimal mock model for testing without real LLM."""

    class Config:
        pad_token_id = 0

    def __init__(self):
        import torch.nn as nn
        self.config = self.Config()
        self._linear = nn.Linear(100, 1000)

    def eval(self): return self
    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, input_ids=None, labels=None, **kwargs):
        class Output:
            loss = torch.tensor(2.3)  # PPL ≈ e^2.3 ≈ 9.97
            logits = torch.randn(1, input_ids.shape[1], 1000)
        return Output()


class MockTokenizer:
    pad_token_id = 0

    def __call__(self, texts, return_tensors=None, **kwargs):
        class Inputs:
            input_ids = torch.ones(len(texts) if isinstance(texts, list) else 1, 20, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            def __contains__(self, key): return key in ["input_ids", "attention_mask"]
            def __getitem__(self, key):
                if key == "attention_mask": return self.attention_mask
                return self.input_ids
        return Inputs()

    def encode(self, text, **kwargs):
        return [ord(text[0])] if text else []


class TestPerplexityExperiment:

    def test_pplresult_pass(self):
        result = PPLResult("test", 10.0, 10.05, 0.005, 100)
        assert result.passed   # 0.5% < 2%

    def test_pplresult_fail(self):
        result = PPLResult("test", 10.0, 10.5, 0.05, 100)
        assert not result.passed   # 5% > 2%

    def test_pplresult_report(self):
        result = PPLResult("test", 10.0, 10.05, 0.005, 100)
        report = result.report()
        assert "Baseline PPL" in report
        assert "PASS" in report

    def test_compute_perplexity_with_mock(self):
        model     = MockModel()
        tokenizer = MockTokenizer()
        exp       = PerplexityExperiment(model, tokenizer)
        texts     = ["The quick brown fox. " * 5] * 3
        ppl       = exp.compute_perplexity(texts)
        assert ppl > 0
        assert ppl < 1e6

    def test_run_with_embed_fn(self):
        model     = MockModel()
        tokenizer = MockTokenizer()
        exp       = PerplexityExperiment(model, tokenizer)

        embed_called = [False]
        def embed_fn(m):
            embed_called[0] = True

        texts = ["The quick brown fox. " * 5] * 2
        # Monkey-patch load_wikitext2
        exp.load_wikitext2 = lambda n: texts

        result = exp.run(embed_fn, n_samples=2, label="test")
        assert embed_called[0]
        assert isinstance(result.ppl_baseline, float)
        assert isinstance(result.ppl_embedded, float)


class TestTaskAccuracyExperiment:

    def test_taskresult_pass(self):
        result = TaskResult("test", 0.50, 0.495, 0.005, 100)
        assert result.passed   # 0.5% < 1%

    def test_taskresult_fail(self):
        result = TaskResult("test", 0.50, 0.48, 0.02, 100)
        assert not result.passed   # 2% > 1%

    def test_taskresult_report(self):
        result = TaskResult("test", 0.50, 0.495, 0.005, 100)
        assert "Baseline accuracy" in result.report()
        assert "PASS" in result.report()

    def test_synthetic_questions(self):
        model     = MockModel()
        tokenizer = MockTokenizer()
        exp       = TaskAccuracyExperiment(model, tokenizer)
        questions = exp._synthetic_questions(10)
        assert len(questions) == 10
        assert all("question" in q for q in questions)
        assert all("choices"  in q for q in questions)
        assert all("answer"   in q for q in questions)

    def test_format_question(self):
        model     = MockModel()
        tokenizer = MockTokenizer()
        exp       = TaskAccuracyExperiment(model, tokenizer)
        q         = {
            "question": "What is 2+2?",
            "choices":  ["3", "4", "5", "6"],
            "answer":   1,
        }
        prompt = exp._format_question(q)
        assert "A. 3" in prompt
        assert "B. 4" in prompt
        assert "Answer:" in prompt


class TestPPLResultMath:

    def test_degradation_calculation(self):
        """Degradation = (embedded - baseline) / baseline."""
        baseline = 10.0
        embedded = 10.5
        deg      = (embedded - baseline) / baseline
        result   = PPLResult("t", baseline, embedded, deg, 100)
        assert abs(result.degradation - 0.05) < 1e-6

    def test_zero_degradation(self):
        result = PPLResult("t", 10.0, 10.0, 0.0, 100)
        assert result.passed
        assert result.degradation == 0.0


if __name__ == "__main__":
    import sys
    classes = [
        TestPerplexityExperiment(),
        TestTaskAccuracyExperiment(),
        TestPPLResultMath(),
    ]
    passed = failed = 0
    for obj in classes:
        cls = type(obj).__name__
        for method in [m for m in dir(obj) if m.startswith("test_")]:
            try:
                getattr(obj, method)()
                print(f"  ✅ {cls}.{method}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {cls}.{method}: {e}")
                failed += 1
    print(f"\n{'='*55}\n  {passed} passed, {failed} failed\n{'='*55}")
    sys.exit(0 if failed == 0 else 1)