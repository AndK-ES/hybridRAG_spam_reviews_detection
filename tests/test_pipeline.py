"""Unit tests for the routing and interpolation helpers."""

from __future__ import annotations

import unittest

try:
    from spam_detection.pipeline import base_probability, route_label, run_ablation_study
except Exception as exc:  # pragma: no cover - depends on the binary stack
    base_probability = None
    route_label = None
    run_ablation_study = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - depends on the binary stack
    pd = None


@unittest.skipIf(base_probability is None, f"pipeline import failed: {IMPORT_ERROR}")
class BaseProbabilityTests(unittest.TestCase):
    def test_classifier_only(self) -> None:
        self.assertAlmostEqual(base_probability(0.8, None), 0.8)
        self.assertAlmostEqual(base_probability(0.8, 0.2, use_interpolation=False), 0.8)

    def test_interpolation(self) -> None:
        # alpha * p_clf + (1 - alpha) * p_knn, alpha default 0.55
        self.assertAlmostEqual(base_probability(0.8, 0.2, use_interpolation=True), 0.53)

    def test_interpolation_without_knn_falls_back(self) -> None:
        self.assertAlmostEqual(base_probability(0.8, None, use_interpolation=True), 0.8)

    def test_custom_alpha(self) -> None:
        self.assertAlmostEqual(
            base_probability(1.0, 0.0, alpha=0.25, use_interpolation=True), 0.25
        )


@unittest.skipIf(route_label is None, f"pipeline import failed: {IMPORT_ERROR}")
class RouteLabelTests(unittest.TestCase):
    def test_confident_real(self) -> None:
        self.assertEqual(route_label(0.1), (0, False))

    def test_confident_fake(self) -> None:
        self.assertEqual(route_label(0.9), (1, False))

    def test_gray_zone(self) -> None:
        self.assertEqual(route_label(0.5), (1, True))
        self.assertEqual(route_label(0.3), (0, True))
        self.assertEqual(route_label(0.7), (1, True))

    def test_custom_thresholds(self) -> None:
        self.assertEqual(route_label(0.2, 0.1, 0.9), (0, True))
        self.assertEqual(route_label(0.05, 0.1, 0.9), (0, False))


class FakeChain:
    """Minimal LLM stand-in that records prompt arguments."""

    def __init__(self, response: str = "LABEL: 1\nREASONING: test") -> None:
        self.response = response
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> str:
        self.calls.append(args)
        return self.response


@unittest.skipIf(
    run_ablation_study is None or pd is None,
    f"pipeline import failed: {IMPORT_ERROR}",
)
class AblationPipelineTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame(
            {"text": ["a review long enough", "another review"], "label": [1, 0]}
        )

    def test_confident_review_skips_llm(self) -> None:
        chain = FakeChain()
        results = run_ablation_study(
            df=self._frame(),
            clf_probas=[0.95, 0.05],
            llm_chain=chain,
            use_llm=True,
            use_retrieval=True,
            use_interpolation=False,
            show_progress=False,
        )
        self.assertEqual(chain.calls, [])
        self.assertEqual(results["llm_label"].tolist(), [1, 0])
        self.assertTrue(
            results["llm_response"].str.startswith("Skipped LLM").all()
        )

    def test_uncertain_review_calls_llm(self) -> None:
        chain = FakeChain("LABEL: 0\nREASONING: looks genuine")
        results = run_ablation_study(
            df=self._frame().iloc[[0]].reset_index(drop=True),
            clf_probas=[0.5],
            llm_chain=chain,
            use_llm=True,
            use_retrieval=True,
            use_interpolation=False,
            show_progress=False,
        )
        self.assertEqual(len(chain.calls), 1)
        self.assertEqual(results.loc[0, "llm_label"], 0)
        self.assertEqual(results.loc[0, "clf_label"], 1)

    def test_interpolation_combines_probabilities(self) -> None:
        summary = {
            "top1_score": 0.8,
            "mean_topk_score": 0.7,
            "std_topk_score": 0.1,
            "score_gap": -0.1,
            "neighbor_count": 20,
            "fake_ratio": 0.2,
            "top3_examples": ["[REAL] example"] * 3,
        }
        results = run_ablation_study(
            df=self._frame().iloc[[0]].reset_index(drop=True),
            clf_probas=[0.8],
            faiss_summaries=[summary],
            use_llm=False,
            use_retrieval=False,
            use_interpolation=True,
            alpha=0.5,
            show_progress=False,
        )
        self.assertAlmostEqual(results.loc[0, "p_base"], 0.5)
        self.assertAlmostEqual(results.loc[0, "p_knn"], 0.2)


if __name__ == "__main__":
    unittest.main()
