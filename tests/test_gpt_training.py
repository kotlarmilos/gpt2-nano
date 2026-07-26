import copy
import unittest

import torch

from src.gpt import language_model_objective
from src.model import GPT, GPTConfig


class KLTrainingIntegrationTests(unittest.TestCase):
    def test_kl_objective_updates_model_but_not_reference(self) -> None:
        torch.manual_seed(3)
        config = GPTConfig(
            vocab_size=17,
            context_len=8,
            embedding_dim=16,
            num_layers=1,
            num_heads=2,
        )
        reference = GPT(config).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        model = copy.deepcopy(reference).train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        with torch.no_grad():
            model.lm_head.weight[0, 0].add_(0.5)
        reference_before = {
            name: parameter.detach().clone()
            for name, parameter in reference.named_parameters()
        }
        model_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.randint(0, config.vocab_size, (2, config.context_len))

        total, task, kl = language_model_objective(
            model,
            inputs,
            reference,
            kl_beta=0.1,
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        self.assertGreater(float(task), 0)
        self.assertGreater(float(kl), 0)
        self.assertTrue(
            any(
                not torch.equal(model_before[name], parameter)
                for name, parameter in model.named_parameters()
            )
        )
        self.assertTrue(
            all(
                torch.equal(reference_before[name], parameter)
                for name, parameter in reference.named_parameters()
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in reference.parameters())
        )


if __name__ == "__main__":
    unittest.main()
