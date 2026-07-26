import unittest

import torch

from src.model import GPT, GPTConfig, generate


class GPTConfigTests(unittest.TestCase):
    def test_rejects_incompatible_head_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            GPTConfig(vocab_size=32, embedding_dim=14, num_heads=4)

    def test_rejects_invalid_dropout(self) -> None:
        with self.assertRaisesRegex(ValueError, "dropout"):
            GPTConfig(vocab_size=32, dropout=1.0)

    def test_rejects_zero_heads_without_division_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            GPTConfig(vocab_size=32, num_heads=0)

    def test_rejects_odd_embedding_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "even"):
            GPTConfig(vocab_size=32, embedding_dim=15, num_heads=3)


class GPTTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = GPTConfig(
            vocab_size=31,
            context_len=16,
            embedding_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        self.model = GPT(self.config).eval()

    def test_forward_shape(self) -> None:
        tokens = torch.randint(0, self.config.vocab_size, (2, 6))
        logits = self.model(tokens)
        self.assertEqual(logits.shape, (2, 6, self.config.vocab_size))

    def test_incremental_cache_matches_full_forward(self) -> None:
        tokens = torch.randint(0, self.config.vocab_size, (1, 7))
        full_logits = self.model(tokens)

        cached_logits = []
        cache = None
        for position in range(tokens.size(1)):
            step_logits, cache = self.model(
                tokens[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cached_logits.append(step_logits)

        torch.testing.assert_close(
            torch.cat(cached_logits, dim=1),
            full_logits,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_cache_has_one_entry_per_layer(self) -> None:
        tokens = torch.randint(0, self.config.vocab_size, (2, 5))
        _, cache = self.model(tokens, use_cache=True)
        self.assertEqual(len(cache), self.config.num_layers)
        for key, value in cache:
            self.assertEqual(key.shape, (2, 4, 5, 4))
            self.assertEqual(value.shape, key.shape)

    def test_cached_greedy_generation_matches_uncached(self) -> None:
        prompt = [1, 2, 3, 4]
        cached = generate(
            self.model, prompt, 5, use_cache=True, do_sample=False
        )
        uncached = generate(
            self.model, prompt, 5, use_cache=False, do_sample=False
        )
        self.assertEqual(cached, uncached)

    def test_generation_restores_training_mode_after_error(self) -> None:
        self.model.train()
        with self.assertRaises(IndexError):
            generate(
                self.model,
                [self.config.vocab_size],
                1,
                use_cache=True,
                do_sample=False,
            )
        self.assertTrue(self.model.training)

    def test_rejects_context_overflow(self) -> None:
        tokens = torch.randint(
            0, self.config.vocab_size, (1, self.config.context_len + 1)
        )
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            self.model(tokens)

    def test_dropout_is_disabled_in_eval_mode(self) -> None:
        model = GPT(
            GPTConfig(
                vocab_size=31,
                context_len=16,
                embedding_dim=16,
                num_layers=2,
                num_heads=4,
                dropout=0.5,
            )
        ).eval()
        tokens = torch.randint(0, 31, (1, 5))
        torch.testing.assert_close(model(tokens), model(tokens))

    def test_dropout_changes_training_outputs(self) -> None:
        model = GPT(
            GPTConfig(
                vocab_size=31,
                context_len=16,
                embedding_dim=16,
                num_layers=2,
                num_heads=4,
                dropout=0.5,
            )
        ).train()
        tokens = torch.randint(0, 31, (1, 5))
        self.assertFalse(torch.equal(model(tokens), model(tokens)))


if __name__ == "__main__":
    unittest.main()
