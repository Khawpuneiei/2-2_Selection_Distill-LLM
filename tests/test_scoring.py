import unittest

from selection_distill.scoring import answers_match, extract_final_answer, extract_prediction_answer


class MathAnswerScoringTests(unittest.TestCase):
    def test_extracts_gsm8k_delimiter_and_nested_latex_box(self):
        self.assertEqual(extract_final_answer("work\n#### 42"), "42")
        self.assertEqual(extract_final_answer(r"therefore \boxed{\frac{3}{2}}"), r"\frac{3}{2}")

    def test_compares_equivalent_scalar_forms(self):
        self.assertTrue(answers_match(r"\frac{1}{2}", "0.5"))
        self.assertTrue(answers_match("1,200", "1200"))
        self.assertFalse(answers_match("12", "21"))

    def test_numeric_reference_falls_back_to_last_number_in_answer_line(self):
        self.assertTrue(answers_match("He pays 3 * 6 = $18.\nThe answer is 18.", "18"))
        self.assertFalse(answers_match("The answer is 18 apples, not 20", "18"))
        self.assertFalse(answers_match("The answer is x + 1", r"\frac{1}{2}"))
        huge = "The answer is " + "9" * 400 + ".5 dollars"
        self.assertFalse(answers_match(huge, "18"))

    def test_prediction_uses_first_answer_marker_before_rambling(self):
        text = "48 + 64 = 112. The answer is 112.\nYou are an AI assistant. The answer is 7"
        self.assertEqual(extract_prediction_answer(text), "112")
        self.assertTrue(answers_match(text, "112"))
        self.assertTrue(answers_match("so x = 3\n#### 3\nmore #### 9", "3"))
        self.assertTrue(answers_match(r"Thus \boxed{\frac{1}{2}}. Later \boxed{5}", "0.5"))

    def test_does_not_execute_or_evaluate_arbitrary_answer_text(self):
        self.assertFalse(answers_match("__import__('os').system('whoami')", "1"))


if __name__ == "__main__":
    unittest.main()
