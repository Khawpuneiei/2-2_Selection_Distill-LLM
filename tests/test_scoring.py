import unittest

from selection_distill.scoring import answers_match, extract_final_answer


class MathAnswerScoringTests(unittest.TestCase):
    def test_extracts_gsm8k_delimiter_and_nested_latex_box(self):
        self.assertEqual(extract_final_answer("work\n#### 42"), "42")
        self.assertEqual(extract_final_answer(r"therefore \boxed{\frac{3}{2}}"), r"\frac{3}{2}")

    def test_compares_equivalent_scalar_forms(self):
        self.assertTrue(answers_match(r"\frac{1}{2}", "0.5"))
        self.assertTrue(answers_match("1,200", "1200"))
        self.assertFalse(answers_match("12", "21"))

    def test_does_not_execute_or_evaluate_arbitrary_answer_text(self):
        self.assertFalse(answers_match("__import__('os').system('whoami')", "1"))


if __name__ == "__main__":
    unittest.main()
