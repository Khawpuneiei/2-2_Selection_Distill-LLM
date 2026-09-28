import math
import unittest

import torch

from selection_distill.metrics import mean_entropy_from_scores


class GenerationEntropyTests(unittest.TestCase):
    def test_entropy_is_computed_from_each_generated_token_distribution(self):
        scores = [torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0, math.log(3.0)]])]

        self.assertAlmostEqual(mean_entropy_from_scores(scores), (math.log(2.0) + 0.5623351446) / 2.0, places=6)

    def test_empty_generation_has_zero_entropy(self):
        self.assertEqual(mean_entropy_from_scores([]), 0.0)


if __name__ == "__main__":
    unittest.main()
