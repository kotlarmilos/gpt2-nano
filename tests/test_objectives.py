import unittest

import torch

from src.objectives import categorical_kl, kl_regularized_loss


class KLDivergenceTests(unittest.TestCase):
    def test_identical_distributions_have_zero_kl(self) -> None:
        logits = torch.tensor([[[1.0, 2.0, 3.0]]])
        self.assertAlmostEqual(categorical_kl(logits, logits).item(), 0.0, places=6)

    def test_kl_matches_manual_calculation(self) -> None:
        model_logits = torch.tensor([[[0.0, 1.0]]])
        reference_logits = torch.tensor([[[1.0, 0.0]]])
        model_log = torch.log_softmax(model_logits, dim=-1)
        reference_log = torch.log_softmax(reference_logits, dim=-1)
        expected = (
            model_log.exp() * (model_log - reference_log)
        ).sum()
        torch.testing.assert_close(
            categorical_kl(model_logits, reference_logits), expected
        )

    def test_regularized_loss_reports_components(self) -> None:
        model_logits = torch.randn(2, 3, 5, requires_grad=True)
        reference_logits = torch.randn(2, 3, 5, requires_grad=True)
        targets = torch.randint(0, 5, (2, 3))
        total, task, kl = kl_regularized_loss(
            model_logits, targets, reference_logits, beta=0.25
        )
        torch.testing.assert_close(total, task + 0.25 * kl)
        total.backward()
        self.assertIsNotNone(model_logits.grad)
        self.assertIsNone(reference_logits.grad)

    def test_default_reduction_averages_over_tokens(self) -> None:
        model_logits = torch.tensor(
            [[[0.0, 1.0], [2.0, 0.0]]]
        )
        reference_logits = torch.tensor(
            [[[1.0, 0.0], [0.0, 2.0]]]
        )
        per_token = categorical_kl(
            model_logits, reference_logits, reduction="none"
        )
        torch.testing.assert_close(
            categorical_kl(model_logits, reference_logits),
            per_token.mean(),
        )

    def test_rejects_negative_beta(self) -> None:
        logits = torch.randn(1, 2, 3)
        targets = torch.randint(0, 3, (1, 2))
        with self.assertRaisesRegex(ValueError, "beta"):
            kl_regularized_loss(logits, targets, logits, beta=-0.1)


if __name__ == "__main__":
    unittest.main()
