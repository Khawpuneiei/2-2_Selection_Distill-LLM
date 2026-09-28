import unittest

from selection_distill.reporting import build_ablation_rows


class AblationReportTests(unittest.TestCase):
    def test_results_bind_each_arm_to_its_budget_and_dataset_accuracy(self):
        selection = {
            "arms": {
                "uniform": {"training_tokens": 100, "teacher_tokens_used": 900, "teacher_token_receipt_complete": True},
                "quadrant_prioritized": {"training_tokens": 100, "teacher_tokens_used": None, "teacher_token_receipt_complete": False},
            }
        }
        evaluations = {
            "uniform": {"splits": {"gsm8k_test": {"accuracy": 0.5, "mean_entropy": 0.7}}},
            "quadrant_prioritized": {"splits": {"gsm8k_test": {"accuracy": 0.6, "mean_entropy": 0.6}}},
        }

        rows = build_ablation_rows(selection, evaluations)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["arm"], "uniform")
        self.assertEqual(rows[0]["dataset"], "gsm8k_test")
        self.assertEqual(rows[0]["training_tokens"], 100)
        self.assertEqual(rows[0]["teacher_tokens_used"], 900)
        self.assertEqual(rows[1]["accuracy"], 0.6)
        self.assertIsNone(rows[1]["teacher_tokens_used"])


if __name__ == "__main__":
    unittest.main()
