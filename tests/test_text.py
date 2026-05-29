import unittest

from sermon_pipeline.text import classify_korean_paragraph, is_layout_noise, normalize_ws, script_counts


class TextTests(unittest.TestCase):
    def test_normalize_ws_collapses_spaces_and_nbsp(self):
        self.assertEqual(normalize_ws("  요한\u00a0  일서\n2장\t1절  "), "요한 일서 2장 1절")

    def test_layout_noise_filters_page_numbers_and_marks(self):
        self.assertTrue(is_layout_noise("12"))
        self.assertTrue(is_layout_noise("•"))
        self.assertFalse(is_layout_noise("오늘 본문 말씀은 요한일서 2장 1절입니다."))

    def test_script_counts_detects_mixed_theology_terms(self):
        counts = script_counts("파라클레토스 παράκλητος logos דבר")
        self.assertGreater(counts["hangul"], 0)
        self.assertGreater(counts["greek"], 0)
        self.assertGreater(counts["latin"], 0)
        self.assertGreater(counts["hebrew"], 0)

    def test_classify_korean_paragraph_keeps_korean_and_scripture_reference(self):
        keep, counts, reason = classify_korean_paragraph("(계 19:7, 8)")
        self.assertTrue(keep)
        self.assertEqual(reason, "scripture_reference_or_short_context")
        self.assertEqual(counts["hangul"], 1)

    def test_classify_korean_paragraph_removes_foreign_translation(self):
        keep, counts, reason = classify_korean_paragraph("This paragraph is an English translation with many latin letters.")
        self.assertFalse(keep)
        self.assertEqual(reason, "latin_dominant_foreign")
        self.assertGreater(counts["latin"], 20)


if __name__ == "__main__":
    unittest.main()
