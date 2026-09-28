import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from selection_distill.train_loop import run_training_loop


class TinyTokenizer:
    eos_token = "<eos>"

    def __init__(self):
        self.tokens = {"<bos>": 1}

    def encode(self, text, add_special_tokens=True):
        pieces = text.split()
        ids = []
        for piece in pieces:
            if piece not in self.tokens:
                self.tokens[piece] = len(self.tokens) + 1
            ids.append(self.tokens[piece])
        return ([1] + ids) if add_special_tokens else ids


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 8)
        self.lm_head = torch.nn.Linear(8, 128)

    def forward(self, input_ids, labels, use_cache=False):
        logits = self.lm_head(self.embedding(input_ids))
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss)


class TrainingLoopTests(unittest.TestCase):
    def test_training_updates_model_and_consumes_the_exact_arm_token_budget(self):
        model = TinyCausalLM()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        rows = [
            {"id": "a", "question": "add 2 and 3", "solution": "answer is 5", "allocated_tokens": 8, "loss_weight": 1.0},
            {"id": "b", "question": "subtract 1 from 4", "solution": "answer is 3", "allocated_tokens": 9, "loss_weight": 1.0},
        ]

        receipt = run_training_loop(
            model,
            TinyTokenizer(),
            rows,
            device=torch.device("cpu"),
            max_length=32,
            tokens_per_update=1,
            learning_rate=1e-2,
            seed=3,
        )

        self.assertEqual(receipt["training_tokens_seen"], 17)
        self.assertTrue(receipt["completed_token_budget"])
        self.assertGreaterEqual(receipt["optimizer_steps"], 1)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_invalid_loss_weight_is_rejected_before_optimizer_step(self):
        model = TinyCausalLM()
        rows = [{"id": "a", "question": "add 1", "solution": "answer 2", "allocated_tokens": 8, "loss_weight": 0}]

        with self.assertRaisesRegex(ValueError, "loss weight"):
            run_training_loop(
                model, TinyTokenizer(), rows,
                device=torch.device("cpu"), max_length=32,
                tokens_per_update=1, learning_rate=1e-2, seed=3,
            )

    def test_update_cap_marks_partial_training_as_incomplete(self):
        model = TinyCausalLM()
        rows = [
            {"id": "a", "question": "add 2", "solution": "answer 4", "allocated_tokens": 8, "loss_weight": 1.0},
            {"id": "b", "question": "add 3", "solution": "answer 5", "allocated_tokens": 8, "loss_weight": 1.0},
        ]

        receipt = run_training_loop(
            model, TinyTokenizer(), rows,
            device=torch.device("cpu"), max_length=32,
            tokens_per_update=1, learning_rate=1e-2, seed=3, max_updates=1,
        )

        self.assertEqual(receipt["optimizer_steps"], 1)
        self.assertTrue(receipt["early_stopped"])
        self.assertFalse(receipt["completed_token_budget"])


if __name__ == "__main__":
    unittest.main()
