import unittest

from selection_distill.training_utils import encode_supervised_example


class WhitespaceTokenizer:
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=True):
        tokens = text.split()
        return (["<bos>"] + tokens) if add_special_tokens else tokens


class TrainingTokenBudgetTests(unittest.TestCase):
    def test_encoding_masks_the_prompt_and_counts_nonpadding_tokens_exactly(self):
        encoded = encode_supervised_example(
            {"question": "add 2 and 3", "solution": "the answer is 5", "allocated_tokens": 32},
            WhitespaceTokenizer(),
            max_length=32,
        )

        self.assertEqual(len(encoded["input_ids"]), 12)
        self.assertEqual(encoded["input_ids"], ["<bos>", "Problem:", "add", "2", "and", "3", "Solution:", "the", "answer", "is", "5", "<eos>"])
        self.assertEqual(encoded["labels"], [-100, -100, -100, -100, -100, -100, -100, "the", "answer", "is", "5", "<eos>"])

    def test_partial_final_example_keeps_supervised_answer_tokens(self):
        encoded = encode_supervised_example(
            {"question": "compute 2 plus 3", "solution": "answer is 5", "allocated_tokens": 2},
            WhitespaceTokenizer(),
            max_length=32,
        )

        self.assertEqual(encoded["input_ids"], ["5", "<eos>"])
        self.assertEqual(encoded["labels"], ["5", "<eos>"])
        self.assertEqual(encoded["active_tokens"], 2)

    def test_invalid_token_caps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "allocated_tokens"):
            encode_supervised_example(
                {"question": "one", "solution": "two", "allocated_tokens": 0},
                WhitespaceTokenizer(),
                max_length=32,
            )


if __name__ == "__main__":
    unittest.main()
