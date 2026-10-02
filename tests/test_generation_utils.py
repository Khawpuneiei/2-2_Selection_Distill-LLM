import unittest

import torch

from selection_distill.generation_utils import (
    generated_lengths,
    left_pad,
    masked_row_means,
    trim_generation,
    truncate_generation_prompt,
)


class GenerationPromptTests(unittest.TestCase):
    def test_truncation_keeps_attention_mask_aligned_with_retained_prompt_tokens(self):
        ids = torch.tensor([[11, 12, 0, 0]])
        mask = torch.tensor([[1, 1, 0, 0]])

        prompt_ids, prompt_mask = truncate_generation_prompt(ids, mask, max_length=3)

        self.assertEqual(prompt_ids.tolist(), [[12, 0, 0]])
        self.assertEqual(prompt_mask.tolist(), [[1, 0, 0]])

    def test_missing_attention_mask_defaults_to_visible_input_tokens(self):
        ids = torch.tensor([[4, 5, 6]])

        prompt_ids, prompt_mask = truncate_generation_prompt(ids, None, max_length=8)

        self.assertIs(prompt_ids, ids)
        self.assertEqual(prompt_mask.tolist(), [[1, 1, 1]])


class BatchedGenerationHelperTests(unittest.TestCase):
    def test_trim_cuts_at_next_problem_header(self):
        self.assertEqual(trim_generation(" 2+2=4\n#### 4\nProblem: next one"), "2+2=4\n#### 4")
        self.assertEqual(trim_generation("no marker"), "no marker")

    def test_left_pad_truncates_and_masks(self):
        ids, mask = left_pad([[1, 2, 3, 4], [5]], 0, max_length=3)

        self.assertEqual(ids.tolist(), [[2, 3, 4], [0, 0, 5]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [0, 0, 1]])

    def test_lengths_and_masked_means_stop_at_first_stop_token(self):
        continuations = torch.tensor([[7, 8, 9, 0], [7, 0, 0, 0]])
        lengths = generated_lengths(continuations, [0])
        steps = [torch.tensor([1.0, 3.0]), torch.tensor([3.0, 9.0]), torch.tensor([5.0, 9.0]), torch.tensor([7.0, 9.0])]

        self.assertEqual(lengths, [4, 2])
        self.assertEqual(masked_row_means(steps, lengths), [4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
