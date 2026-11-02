"""Unit tests for dependency-light helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from spam_detection.utils import (
    as_int_label,
    chunked,
    extract_fake_ratio,
    parse_llm_label,
    to_serializable,
)


class LabelNormalisationTests(unittest.TestCase):
    def test_numeric_labels(self) -> None:
        self.assertEqual(as_int_label(0), 0)
        self.assertEqual(as_int_label(1), 1)
        self.assertEqual(as_int_label(0.0), 0)
        self.assertEqual(as_int_label(1.0), 1)

    def test_yelp_negative_label(self) -> None:
        self.assertEqual(as_int_label(-1), 1)

    def test_string_labels(self) -> None:
        self.assertEqual(as_int_label("fake"), 1)
        self.assertEqual(as_int_label("SPAM"), 1)
        self.assertEqual(as_int_label("real"), 0)
        self.assertEqual(as_int_label("Genuine"), 0)
        self.assertEqual(as_int_label("CG"), 1)
        self.assertEqual(as_int_label("OR"), 0)
        self.assertEqual(as_int_label("Y"), 1)
        self.assertEqual(as_int_label("N"), 0)

    def test_unknown_and_missing(self) -> None:
        self.assertIsNone(as_int_label(None))
        self.assertIsNone(as_int_label("maybe"))
        self.assertIsNone(as_int_label(7))

    def test_boolean(self) -> None:
        self.assertEqual(as_int_label(True), 1)
        self.assertEqual(as_int_label(False), 0)


class LLMOutputParsingTests(unittest.TestCase):
    def test_canonical_format(self) -> None:
        self.assertEqual(parse_llm_label("LABEL: 1\nREASONING: ..."), 1)
        self.assertEqual(parse_llm_label("LABEL: 0\nREASONING: ..."), 0)

    def test_tolerant_format(self) -> None:
        self.assertEqual(parse_llm_label("**label = 1**"), 1)
        self.assertEqual(parse_llm_label("  Label : 0  "), 0)
        self.assertEqual(parse_llm_label("LABEL: 1", fallback=0), 1)

    def test_fallback(self) -> None:
        self.assertEqual(parse_llm_label("no label here", fallback=0), 0)
        self.assertEqual(parse_llm_label("no label here", fallback=1), 1)
        self.assertEqual(parse_llm_label(None, fallback=0), 0)  # type: ignore[arg-type]


class FakeRatioTests(unittest.TestCase):
    def test_dict(self) -> None:
        self.assertAlmostEqual(extract_fake_ratio({"fake_ratio": 0.25}), 0.25)

    def test_serialised_dict(self) -> None:
        self.assertAlmostEqual(
            extract_fake_ratio("{'fake_ratio': 0.4, 'top1_score': 0.9}"), 0.4
        )
        self.assertAlmostEqual(
            extract_fake_ratio('{"fake_ratio": 0.6, "top1_score": 0.9}'), 0.6
        )

    def test_missing(self) -> None:
        self.assertEqual(extract_fake_ratio(None), 0.0)
        self.assertEqual(extract_fake_ratio("not a summary"), 0.0)


class ChunkingTests(unittest.TestCase):
    def test_even_chunks(self) -> None:
        self.assertEqual(list(chunked([1, 2, 3, 4], 2)), [[1, 2], [3, 4]])

    def test_remainder(self) -> None:
        self.assertEqual(list(chunked([1, 2, 3], 2)), [[1, 2], [3]])

    def test_empty(self) -> None:
        self.assertEqual(list(chunked([], 3)), [])


class SerialisationTests(unittest.TestCase):
    def test_path(self) -> None:
        self.assertEqual(to_serializable(Path("a/b")), str(Path("a/b")))

    def test_nested(self) -> None:
        self.assertEqual(
            to_serializable({"x": [Path("a"), 1]}),
            {"x": [str(Path("a")), 1]},
        )


if __name__ == "__main__":
    unittest.main()
