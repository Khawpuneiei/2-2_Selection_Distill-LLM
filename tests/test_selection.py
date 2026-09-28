import unittest

from selection_distill.selection import (
    build_training_arms,
    classify_quadrant,
    summarize_student_metrics,
)


class QuadrantClassificationTests(unittest.TestCase):
    def test_threshold_boundaries_map_to_the_four_named_quadrants(self):
        self.assertEqual(classify_quadrant(0.50, 0.70), "high_priority")
        self.assertEqual(classify_quadrant(0.50, 0.69), "delay")
        self.assertEqual(classify_quadrant(0.51, 0.70), "low_priority")
        self.assertEqual(classify_quadrant(0.51, 0.69), "exclude")


class StudentProfileTests(unittest.TestCase):
    def test_profile_reports_accuracy_loss_entropy_and_questions_failed_every_time(self):
        rows = [
            {"id": "a", "concept": "fractions", "attempt": 1, "correct": False, "entropy": 0.2, "loss": 2.0},
            {"id": "a", "concept": "fractions", "attempt": 2, "correct": False, "entropy": 0.4, "loss": 4.0},
            {"id": "b", "concept": "fractions", "attempt": 1, "correct": True, "entropy": 0.6, "loss": 6.0},
            {"id": "b", "concept": "fractions", "attempt": 2, "correct": False, "entropy": 0.8, "loss": 8.0},
            {"id": "c", "concept": "geometry", "attempt": 1, "correct": True, "entropy": 0.5, "loss": 1.0},
        ]

        profile = summarize_student_metrics(rows)

        self.assertEqual(profile["fractions"]["examples"], 2)
        self.assertEqual(profile["fractions"]["attempts"], 4)
        self.assertEqual(profile["fractions"]["accuracy"], 0.25)
        self.assertEqual(profile["fractions"]["mean_entropy"], 0.5)
        self.assertEqual(profile["fractions"]["mean_loss"], 5.0)
        self.assertEqual(profile["fractions"]["repeated_failures"], 1)


class EqualTokenSelectionTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"id": "a", "concept": "weak_confident", "tokens": 3, "teacher_confidence": 0.9},
            {"id": "b", "concept": "weak_confident", "tokens": 4, "teacher_confidence": 0.8},
            {"id": "c", "concept": "good_confident", "tokens": 5, "teacher_confidence": 1.0},
            {"id": "d", "concept": "weak_uncertain", "tokens": 6, "teacher_confidence": 0.1},
            {"id": "e", "concept": "good_uncertain", "tokens": 2, "teacher_confidence": 0.2},
        ]
        self.quadrants = {
            "weak_confident": "high_priority",
            "weak_uncertain": "delay",
            "good_confident": "low_priority",
            "good_uncertain": "exclude",
        }

    def test_all_three_arms_use_exact_budget_and_prioritized_arm_stays_in_high_priority(self):
        arms = build_training_arms(self.samples, self.quadrants, token_budget=8, seed=23)

        self.assertEqual(set(arms), {"uniform", "quadrant_prioritized", "confidence_weighted"})
        for rows in arms.values():
            self.assertEqual(sum(row["allocated_tokens"] for row in rows), 8)
        prioritized_ids = {row["id"] for row in arms["quadrant_prioritized"]}
        self.assertTrue(prioritized_ids <= {"a", "b"})

    def test_confidence_weighted_arm_uses_same_draw_as_uniform_and_weights_high_confidence_more(self):
        arms = build_training_arms(self.samples, self.quadrants, token_budget=20, seed=23)

        uniform = arms["uniform"]
        weighted = arms["confidence_weighted"]
        self.assertEqual([row["id"] for row in uniform], [row["id"] for row in weighted])
        by_id = {row["id"]: row["loss_weight"] for row in weighted}
        self.assertAlmostEqual(by_id["c"], 1.0 / 0.6)
        self.assertAlmostEqual(by_id["e"], 0.2 / 0.6)

    def test_repeating_and_truncating_examples_is_visible_in_the_allocation(self):
        arms = build_training_arms(self.samples, self.quadrants, token_budget=8, seed=23)
        selected = arms["quadrant_prioritized"]

        self.assertTrue(any(row["truncated"] for row in selected))
        self.assertTrue(any(row["reused"] for row in selected))

    def test_missing_high_priority_concepts_is_reported_instead_of_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, "high_priority"):
            build_training_arms(
                self.samples,
                {concept: "exclude" for concept in self.quadrants},
                token_budget=8,
                seed=23,
            )


if __name__ == "__main__":
    unittest.main()
