"""Unit tests for text cleaning and short-review filtering."""

from __future__ import annotations

import unittest

from spam_detection.preprocessing import clean_text_basic, filter_short_reviews

try:
    import pandas as pd
except Exception:  # pragma: no cover - depends on the local binary stack
    pd = None


class CleanTextTests(unittest.TestCase):
    def test_removes_html_and_punctuation(self) -> None:
        self.assertEqual(
            clean_text_basic("<b>Great</b> hotel!!! 10/10."),
            "Great hotel 1010",
        )

    def test_preserves_case_by_default(self) -> None:
        self.assertEqual(clean_text_basic("AMAZING Stay"), "AMAZING Stay")
        self.assertEqual(
            clean_text_basic("AMAZING Stay", keep_case=False), "amazing stay"
        )

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(clean_text_basic("a \n\t  b"), "a b")

    def test_missing_values(self) -> None:
        self.assertEqual(clean_text_basic(None), "")
        self.assertEqual(clean_text_basic(""), "")


@unittest.skipIf(pd is None, "pandas is not importable in this environment")
class FilterShortReviewsTests(unittest.TestCase):
    def test_keeps_long_enough_rows(self) -> None:
        frame = pd.DataFrame({"text": ["short", "x" * 31, ""]})
        filtered = filter_short_reviews(frame, min_chars=30)
        self.assertEqual(filtered["text"].tolist(), ["x" * 31])

    def test_handles_missing_values(self) -> None:
        frame = pd.DataFrame({"text": [None, "x" * 30]})
        filtered = filter_short_reviews(frame, min_chars=30)
        self.assertEqual(len(filtered), 1)


if __name__ == "__main__":
    unittest.main()
