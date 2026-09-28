import unittest

from selection_distill.data import normalize_example, stable_id


class DatasetNormalizationTests(unittest.TestCase):
    def test_gsm8k_answer_and_source_id_are_normalized_without_inventing_fine_tags(self):
        raw = {
            "question": "A student has 3 apples and gets 2 more. How many?",
            "answer": "Add the apples.\n#### 5",
        }

        row = normalize_example("gsm8k", "train", raw)

        self.assertEqual(row["answer"], "5")
        self.assertEqual(row["solution"], "Add the apples.")
        self.assertEqual(row["concept"], "gsm8k_word_problem")
        self.assertTrue(row["id"].startswith("gsm8k-train-"))
        self.assertEqual(row["id"], normalize_example("gsm8k", "train", raw)["id"])

    def test_math_subject_and_type_are_preserved_as_separate_labels(self):
        row = normalize_example(
            "math",
            "train",
            {
                "problem": "Solve 2x = 6.",
                "solution": "Divide by two.\\boxed{3}",
                "type": "linear_equations",
                "level": "Level 2",
            },
            subject="algebra",
        )

        self.assertEqual(row["concept"], "algebra")
        self.assertEqual(row["subconcept"], "linear_equations")
        self.assertEqual(row["answer"], "3")

    def test_stable_ids_change_when_dataset_split_or_question_changes(self):
        question = "Find the sum."
        self.assertNotEqual(stable_id("gsm8k", "train", question), stable_id("gsm8k", "test", question))
        self.assertNotEqual(stable_id("gsm8k", "train", question), stable_id("gsm8k", "train", "Find the product."))


if __name__ == "__main__":
    unittest.main()
