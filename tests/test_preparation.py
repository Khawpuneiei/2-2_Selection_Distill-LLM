import unittest

from selection_distill.preparation import attach_teacher_scores, split_profile_rows


class TeacherScoreJoinTests(unittest.TestCase):
    def test_joins_sample_scores_and_aggregates_confidence_by_concept(self):
        samples = [
            {"id": "a", "concept": "algebra"},
            {"id": "b", "concept": "algebra"},
            {"id": "c", "concept": "geometry"},
        ]
        scores = [
            {"id": "a", "concept": "algebra", "confidence": "0.8", "teacher_tokens": "100"},
            {"id": "b", "concept": "algebra", "confidence": "0.6", "teacher_tokens": "80"},
            {"id": "c", "concept": "geometry", "confidence": "0.9", "teacher_tokens": "50"},
        ]

        attached, by_concept = attach_teacher_scores(samples, scores)

        self.assertEqual([row["teacher_confidence"] for row in attached], [0.8, 0.6, 0.9])
        self.assertEqual([row["teacher_tokens"] for row in attached], [100, 80, 50])
        self.assertEqual(by_concept, {"algebra": 0.7, "geometry": 0.9})

    def test_concept_level_scores_can_be_reused_for_each_matching_example(self):
        samples = [{"id": "a", "concept": "algebra"}, {"id": "b", "concept": "algebra"}]
        scores = [{"concept": "algebra", "confidence": "0.75", "teacher_tokens": "210"}]

        attached, by_concept = attach_teacher_scores(samples, scores)

        self.assertEqual([row["teacher_confidence"] for row in attached], [0.75, 0.75])
        self.assertEqual({row["teacher_score_key"] for row in attached}, {"concept:algebra"})
        self.assertEqual(by_concept, {"algebra": 0.75})

    def test_missing_or_conflicting_scores_fail_with_the_source_ids(self):
        with self.assertRaisesRegex(ValueError, "b"):
            attach_teacher_scores(
                [{"id": "a", "concept": "algebra"}, {"id": "b", "concept": "algebra"}],
                [{"id": "a", "confidence": "0.8"}],
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            attach_teacher_scores(
                [{"id": "a", "concept": "algebra"}],
                [{"id": "a", "concept": "geometry", "confidence": "0.8"}],
            )


class StudentProfileSplitTests(unittest.TestCase):
    def test_profile_split_is_stable_disjoint_and_preserves_each_example_once(self):
        rows = [
            {"id": f"{concept}-{index}", "concept": concept}
            for concept in ("algebra", "geometry")
            for index in range(4)
        ]

        profile, training = split_profile_rows(rows, fraction=0.5, seed=11)

        self.assertEqual(len(profile), 4)
        self.assertEqual(len(training), 4)
        self.assertFalse({row["id"] for row in profile} & {row["id"] for row in training})
        self.assertEqual({row["id"] for row in profile + training}, {row["id"] for row in rows})
        self.assertEqual(profile, split_profile_rows(rows, fraction=0.5, seed=11)[0])


if __name__ == "__main__":
    unittest.main()
