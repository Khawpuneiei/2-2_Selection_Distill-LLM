import unittest

import torch

from selection_distill.generation_utils import truncate_generation_prompt


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


if __name__ == "__main__":
    unittest.main()
