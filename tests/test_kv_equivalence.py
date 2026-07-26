import unittest

import torch

from src.kv_equivalence import (
    dtype_from_name,
    max_logit_deviation,
    measure_prompt_equivalence,
    run_study,
    summarize_prompt_results,
)
from src.model import GPT, GPTConfig


class KVEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.config = GPTConfig(
            vocab_size=23,
            context_len=12,
            embedding_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        self.model = GPT(self.config).eval()

    def test_dtype_from_name_accepts_expected_aliases(self) -> None:
        self.assertIs(dtype_from_name("fp32"), torch.float32)
        self.assertIs(dtype_from_name("bf16"), torch.bfloat16)
        self.assertIs(dtype_from_name("float16"), torch.float16)

    def test_max_logit_deviation_uses_symmetric_scale(self) -> None:
        cached = torch.tensor([[2.0, 0.0]])
        uncached = torch.tensor([[1.0, 1e-7]])
        max_abs, max_relative = max_logit_deviation(
            cached,
            uncached,
            relative_epsilon=1e-6,
        )
        self.assertEqual(max_abs, 1.0)
        self.assertAlmostEqual(max_relative, 0.5)

    def test_measure_prompt_equivalence_matches_in_float32(self) -> None:
        result = measure_prompt_equivalence(
            self.model,
            [1, 2, 3],
            horizon=4,
            relative_epsilon=1e-6,
        )

        self.assertEqual(len(result.steps), 4)
        self.assertEqual(len(result.generated_new_tokens), 4)
        self.assertIsNone(result.first_flip_generation_position)
        self.assertLessEqual(result.max_abs_logit_deviation, 1e-5)
        self.assertTrue(
            all(
                step.uncached_next_token == step.cached_next_token
                for step in result.steps
            )
        )

    def test_measure_prompt_equivalence_rejects_context_overflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "context"):
            measure_prompt_equivalence(self.model, [1, 2, 3], horizon=10)

    def test_summarize_prompt_results_reports_flip_aggregate(self) -> None:
        result = measure_prompt_equivalence(
            self.model,
            [1, 2, 3],
            horizon=2,
            relative_epsilon=1e-6,
        )
        summary = summarize_prompt_results([result])
        self.assertEqual(summary["prompt_count"], 1)
        self.assertEqual(summary["prompts_with_flip"], 0)
        self.assertIsNone(summary["first_flip_generation_position_min"])

    def test_run_study_supports_random_toy_config(self) -> None:
        payload = run_study(
            {
                "device": "cpu",
                "dtypes": ["float32"],
                "horizon": 2,
                "relative_epsilon": 1e-6,
                "seed": 13,
                "prompt_tokens": [[1, 2, 3]],
                "model_config": {
                    "vocab_size": 23,
                    "context_len": 8,
                    "embedding_dim": 16,
                    "num_layers": 1,
                    "num_heads": 4,
                    "dropout": 0.0,
                },
            }
        )

        self.assertEqual(payload["status"], "measured-local-development")
        self.assertEqual(payload["prompt_count"], 1)
        self.assertIn("float32", payload["summary"])


if __name__ == "__main__":
    unittest.main()
