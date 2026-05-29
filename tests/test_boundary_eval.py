from __future__ import annotations

import unittest

from sermon_pipeline.boundary_eval import (
    boundary_f1,
    parse_student_output,
    pk_score,
    windowdiff_score,
)


class BoundaryEvalTests(unittest.TestCase):
    def test_parse_no_boundary(self) -> None:
        parsed = parse_student_output("NO_BOUNDARY", valid_local_sids={"S1", "S2"})
        self.assertEqual(parsed.boundaries, [])
        self.assertEqual(parsed.issues, [])

    def test_parse_sparse_boundaries(self) -> None:
        parsed = parse_student_output("S2 topic_shift\nS5 application_start", valid_local_sids={"S1", "S2", "S5"})
        self.assertEqual([(item.local_sid, item.boundary_type) for item in parsed.boundaries], [("S2", "topic_shift"), ("S5", "application_start")])
        self.assertEqual(parsed.issues, [])

    def test_parse_reports_invalid_lines(self) -> None:
        parsed = parse_student_output("S2 none\nS9 topic_shift\nbad", valid_local_sids={"S1", "S2"})
        codes = [issue["code"] for issue in parsed.issues]
        self.assertIn("invalid_label", codes)
        self.assertIn("unknown_local_sid", codes)
        self.assertIn("invalid_line", codes)

    def test_boundary_f1_exact_and_tolerance(self) -> None:
        exact = boundary_f1(gold={2, 7}, predicted={2, 8}, tolerance=0)
        tolerant = boundary_f1(gold={2, 7}, predicted={2, 8}, tolerance=1)

        self.assertEqual(exact["true_positive"], 1)
        self.assertAlmostEqual(exact["f1"], 0.5)
        self.assertEqual(tolerant["true_positive"], 2)
        self.assertAlmostEqual(tolerant["f1"], 1.0)

    def test_pk_and_windowdiff_perfect_zero(self) -> None:
        self.assertEqual(pk_score(sentence_count=6, gold={2, 4}, predicted={2, 4}), 0.0)
        self.assertEqual(windowdiff_score(sentence_count=6, gold={2, 4}, predicted={2, 4}), 0.0)


if __name__ == "__main__":
    unittest.main()
