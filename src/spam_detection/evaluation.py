"""Metric computation and result-table helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def binary_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_score: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute the metrics reported in the paper.

    Metrics are calculated for the fake class (label ``1``), with macro F1
    added because the external evaluation sets have very different class
    balances. PR-AUC is only returned when scores are supplied.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None:
        metrics["pr_auc"] = float(average_precision_score(y_true, np.asarray(y_score)))
    return metrics


def predictions_from_results(
    results: Any,
    column: str = "llm_label",
    fallback: str = "clf_label",
) -> np.ndarray:
    """Read predictions from a results frame, falling back when needed."""
    series = results[column]
    if series.isna().any():
        series = series.fillna(results[fallback])
    return series.astype(int).to_numpy()


def comparison_table(
    y_true: Sequence[Any],
    predictions: Mapping[str, Sequence[Any]],
    dataset_name: str = "",
    scores: Mapping[str, Sequence[float]] | None = None,
    decimals: int = 4,
) -> Any:
    """Build a metrics table with one row per configuration."""
    import pandas as pd

    rows = []
    for name, predicted in predictions.items():
        metric_scores = scores.get(name) if scores else None
        metrics = binary_metrics(y_true, predicted, metric_scores)
        row = {"Configuration": name}
        row.update(
            {
                "Accuracy": round(metrics["accuracy"], decimals),
                "Precision": round(metrics["precision"], decimals),
                "Recall": round(metrics["recall"], decimals),
                "F1": round(metrics["f1"], decimals),
                "F1-Macro": round(metrics["f1_macro"], decimals),
            }
        )
        if "pr_auc" in metrics:
            row["PR-AUC"] = round(metrics["pr_auc"], decimals)
        if dataset_name:
            row = {"Dataset": dataset_name, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def gray_zone_analysis(
    results: Any,
    lower_thresh: float = 0.30,
    upper_thresh: float = 0.70,
) -> dict[str, float]:
    """Compare classifier and LLM accuracy inside the uncertainty window."""
    gray = results[
        (results["p_base"] >= lower_thresh) & (results["p_base"] <= upper_thresh)
    ].copy()
    if gray.empty:
        return {
            "gray_zone_size": 0,
            "gray_zone_share": 0.0,
            "classifier_accuracy": float("nan"),
            "llm_accuracy": float("nan"),
            "llm_classifier_agreement": float("nan"),
        }

    y_true = gray["true_label"].astype(int)
    return {
        "gray_zone_size": int(len(gray)),
        "gray_zone_share": float(len(gray) / len(results)),
        "classifier_accuracy": float((gray["clf_label"].astype(int) == y_true).mean()),
        "llm_accuracy": float((gray["llm_label"].astype(int) == y_true).mean()),
        "llm_classifier_agreement": float(
            (gray["llm_label"].astype(int) == gray["clf_label"].astype(int)).mean()
        ),
    }


def save_table(table: Any, path: str, index: bool = False) -> None:
    """Write a table to CSV, creating parent directories when needed."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=index)
