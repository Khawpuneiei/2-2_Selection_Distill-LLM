import unittest

from selection_distill.apply1 import concept_confidence_rows


def _row(dataset, share, *, subject=None, status="complete", correct=True, tokens=3):
    return {
        "example_id": f"{dataset}:{subject}:{share}",
        "status": status,
        "source": {"dataset": dataset, "subject": subject},
        "greedy": {"token_ids": [0] * tokens, "correct": correct},
        "samples": [{"token_ids": [0] * tokens}, {"token_ids": [0] * tokens}],
        "self_consistency": {"greedy_agreement_share": share, "majority_vote_share": 1.0},
    }


class Apply1ExportTests(unittest.TestCase):
    def test_averages_agreement_per_concept_and_skips_gsm_plus_and_incomplete(self):
        rows = concept_confidence_rows(
            [
                _row("gsm8k", 1.0),
                _row("gsm8k", 0.5, correct=False),
                _row("math", 0.25, subject="Number Theory"),
                _row("gsm_plus", 0.0),
                _row("math", 0.0, subject="geometry", status="failed"),
            ]
        )

        by_concept = {row["concept"]: row for row in rows}
        self.assertEqual(set(by_concept), {"gsm8k_word_problem", "number_theory"})
        self.assertAlmostEqual(by_concept["gsm8k_word_problem"]["confidence"], 0.75)
        self.assertEqual(by_concept["gsm8k_word_problem"]["teacher_tokens"], 18)
        self.assertEqual(by_concept["gsm8k_word_problem"]["apply1_greedy_accuracy"], 0.5)
        self.assertEqual(by_concept["number_theory"]["id"], "")

    def test_math_row_without_subject_fails(self):
        with self.assertRaisesRegex(ValueError, "subject"):
            concept_confidence_rows([_row("math", 1.0)])

    def test_rejects_unknown_signal(self):
        with self.assertRaises(ValueError):
            concept_confidence_rows([_row("gsm8k", 1.0)], signal="entropy")


if __name__ == "__main__":
    unittest.main()
