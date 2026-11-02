"""Statistical tests, uncertainty analysis and plots."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .config import RANDOM_SEED, TAU_HIGH, TAU_LOW


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------
def mcnemar_test(
    y_true: Sequence[Any],
    y_model_a: Sequence[Any],
    y_model_b: Sequence[Any],
    exact: bool = True,
) -> float:
    """Return the McNemar p-value comparing two sets of predictions."""
    from statsmodels.stats.contingency_tables import mcnemar

    y_true = np.asarray(y_true).astype(int)
    correct_a = np.asarray(y_model_a).astype(int) == y_true
    correct_b = np.asarray(y_model_b).astype(int) == y_true

    table = np.array(
        [
            [np.sum(~correct_a & ~correct_b), np.sum(~correct_a & correct_b)],
            [np.sum(correct_a & ~correct_b), np.sum(correct_a & correct_b)],
        ]
    )
    return float(mcnemar(table, exact=exact).pvalue)


def summarize_table(
    dataset_name: str,
    y_true: Sequence[Any],
    preds: Mapping[str, Sequence[Any]],
    baseline: str | None = None,
    mcnemar_fn: Any | None = None,
    eps_mcnemar: bool = False,
) -> Any:
    """Build a metrics table with optional McNemar p-values versus a baseline."""
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true).astype(int)
    baseline_name = baseline or next(iter(preds))

    rows = []
    for name, predicted in preds.items():
        predicted = np.asarray(predicted).astype(int)
        row = {
            "Dataset / Setup": f"{dataset_name} - {name}",
            "Accuracy": round(float(accuracy_score(y_true, predicted)), 4),
            "Precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
            "F1 Score": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
            "Macro F1 Score": round(
                float(f1_score(y_true, predicted, average="macro", zero_division=0)), 4
            ),
        }
        if mcnemar_fn is not None:
            if name == baseline_name:
                row["p-value (McNemar's test)"] = "- baseline"
            else:
                value = mcnemar_fn(
                    y_true,
                    np.asarray(preds[baseline_name]).astype(int),
                    predicted,
                )
                row["p-value (McNemar's test)"] = (
                    f"{value:.3e}" if eps_mcnemar else round(float(value), 4)
                )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Uncertainty-window gain analysis
# ---------------------------------------------------------------------------
def bin_gain_analysis(
    y: Sequence[Any],
    p: Sequence[float],
    pred_base: Sequence[Any],
    pred_rag: Sequence[Any],
    edges: Sequence[float] = (0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0),
    name_base: str = "Classifier",
    name_rag: str = "RAG",
    plot: bool = True,
    figsize: tuple[float, float] = (7.5, 3.8),
    graph_title: str | None = None,
    save_path: str | None = None,
) -> Any:
    """Count rescues and harms per classifier-probability bin.

    A *rescue* is a sample the baseline misclassified and the RAG system
    classified correctly; a *harm* is the opposite.
    """
    import pandas as pd

    y, p, base, rag = map(np.asarray, (y, p, pred_base, pred_rag))
    y = y.astype(int)
    base_correct = base.astype(int) == y
    rag_correct = rag.astype(int) == y

    bins = pd.cut(p, bins=list(edges), include_lowest=True)
    rows = []
    for interval in bins.categories:
        mask = np.asarray(bins == interval)
        if mask.sum() == 0:
            continue
        rescue = int((~base_correct & rag_correct & mask).sum())
        harm = int((base_correct & ~rag_correct & mask).sum())
        left = 0.0 if abs(interval.left) < 1e-6 else interval.left
        rows.append(
            {
                "bin": f"{left:.1f}-{interval.right:.1f}",
                "n": int(mask.sum()),
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
            }
        )
    frame = pd.DataFrame(rows)

    if plot and not frame.empty:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.family": "serif", "font.size": 9.5})
        x = np.arange(len(frame))
        _, ax = plt.subplots(figsize=figsize, dpi=300)

        ax.bar(x, frame["rescue"], color="#2a9d8f", alpha=0.9, label=f"Rescued by {name_rag}")
        ax.bar(x, -frame["harm"], color="#e76f51", alpha=0.9, label=f"Broken by {name_rag}")
        ax.plot(x, frame["net"], color="#264653", marker="D", markersize=5, label="Net gain")
        ax.axhline(0, color="k", lw=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{row.bin}\n(n={row.n})" for row in frame.itertuples()]
        )
        ax.set_xlabel(f"{name_base} probability bin")
        ax.set_ylabel("Number of samples")
        if graph_title:
            ax.set_title(graph_title)
        else:
            ax.set_title(f"Gain / loss from {name_rag} (net = {frame['net'].sum():+d})")

        ax.legend(frameon=True, framealpha=0.9)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    return frame


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------
def bootstrap_confidence_test(
    y_true: Sequence[Any],
    y_pred_baseline: Sequence[Any],
    y_pred_rag: Sequence[Any],
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> Any:
    """Paired bootstrap test of the RAG-minus-baseline metric difference.

    Defaults match the paper: 2,000 resamples, 95% confidence, seed 42. The
    paper prints ``p=.055`` for the in-domain F1 row while the prose elsewhere
    says 0.547; either way the F1 change is not significant at 0.05, while the
    precision gain and recall loss are.
    """
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    baseline = np.asarray(y_pred_baseline).astype(int)
    rag = np.asarray(y_pred_rag).astype(int)
    n_samples = len(y_true)

    def metrics(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
        return (
            precision_score(truth, pred, zero_division=0),
            recall_score(truth, pred, zero_division=0),
            f1_score(truth, pred, zero_division=0),
        )

    base_metrics = metrics(y_true, baseline)
    rag_metrics = metrics(y_true, rag)
    observed = {
        name: rag_value - base_value
        for name, rag_value, base_value in zip(
            ("precision", "recall", "f1"), rag_metrics, base_metrics
        )
    }

    differences: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n_samples, size=n_samples)
        sample_truth = y_true[idx]
        base_sample = metrics(sample_truth, baseline[idx])
        rag_sample = metrics(sample_truth, rag[idx])
        for position, name in enumerate(("precision", "recall", "f1")):
            differences[name].append(rag_sample[position] - base_sample[position])

    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)
    ci_label = f"{int((1 - alpha) * 100)}% CI"

    results = {}
    for name in ("precision", "recall", "f1"):
        values = np.asarray(differences[name])
        ci_lower = float(np.percentile(values, lower_pct))
        ci_upper = float(np.percentile(values, upper_pct))

        if observed[name] > 0:
            p_value = 2 * float(np.mean(values <= 0))
        elif observed[name] < 0:
            p_value = 2 * float(np.mean(values >= 0))
        else:
            p_value = 1.0
        p_value = min(1.0, max(p_value, 1.0 / n_bootstraps))

        results[name] = {
            "Delta (RAG - base)": round(observed[name], 4),
            ci_label: (round(ci_lower, 4), round(ci_upper, 4)),
            "p-value": round(p_value, 4),
            "Significant": bool(p_value < alpha),
        }

    return pd.DataFrame(results).T


def plot_bootstrap_results(df_bootstrap: Any, title: str | None = None) -> None:
    """Forest plot of paired-bootstrap metric differences."""
    import matplotlib.pyplot as plt

    metrics = list(df_bootstrap.index)
    deltas = df_bootstrap["Delta (RAG - base)"].to_numpy(dtype=float)
    ci_column = next(column for column in df_bootstrap.columns if "CI" in column)
    ci_lower = np.array([ci[0] for ci in df_bootstrap[ci_column]], dtype=float)
    ci_upper = np.array([ci[1] for ci in df_bootstrap[ci_column]], dtype=float)
    significant = df_bootstrap["Significant"].to_numpy(dtype=bool)

    xerr = np.vstack([deltas - ci_lower, ci_upper - deltas])
    y_pos = list(range(len(metrics)))[::-1]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
        }
    )
    _, ax = plt.subplots(figsize=(7, 3.2), dpi=300)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)

    for y, delta, error_low, error_high, is_significant in zip(
        y_pos, deltas, xerr[0], xerr[1], significant
    ):
        color = "#1f77b4" if is_significant else "#7f7f7f"
        ax.errorbar(
            delta,
            y,
            xerr=[[error_low], [error_high]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=4.5,
            capthick=1.5,
            markersize=6,
            zorder=3,
        )
        ax.text(
            delta,
            y + 0.07,
            f"{delta:+.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([metric.capitalize() for metric in metrics])
    ax.set_xlabel(r"$\Delta$ metric (Full RAG $-$ baseline)")
    ax.set_title(title or f"Paired bootstrap difference ({ci_column})", pad=12)

    max_bound = max(np.max(np.abs(ci_lower)), np.max(np.abs(ci_upper))) * 1.35
    ax.set_xlim(-max_bound, max_bound)
    ax.grid(axis="x", linestyle=":", alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Interpretability metrics
# ---------------------------------------------------------------------------
def positive_correction_rate(
    results: Any,
    lower_thresh: float = TAU_LOW,
    upper_thresh: float = TAU_HIGH,
) -> float:
    """Share of uncertain baseline errors that the LLM corrects."""
    uncertain = results[
        (results["p_base"] >= lower_thresh) & (results["p_base"] <= upper_thresh)
    ].copy()
    if uncertain.empty:
        return float("nan")

    baseline_wrong = uncertain["clf_label"].astype(int) != uncertain["true_label"].astype(int)
    if baseline_wrong.sum() == 0:
        return 0.0
    corrected = (
        (uncertain["llm_label"].astype(int) != uncertain["clf_label"].astype(int))
        & (uncertain["llm_label"].astype(int) == uncertain["true_label"].astype(int))
    )
    return float(corrected[baseline_wrong].mean())


def positive_correction_rates(
    classic_results: Any,
    light_results: Any,
    lower_thresh: float = TAU_LOW,
    upper_thresh: float = TAU_HIGH,
) -> tuple[float, float]:
    """Return ``(classic, light_prompt)`` positive correction rates."""
    return (
        positive_correction_rate(classic_results, lower_thresh, upper_thresh),
        positive_correction_rate(light_results, lower_thresh, upper_thresh),
    )


def override_rate(
    classic_results: Any,
    light_results: Any,
    lower_thresh: float = TAU_LOW,
    upper_thresh: float = TAU_HIGH,
) -> tuple[float, float]:
    """Backward-compatible alias for :func:`positive_correction_rates`.

    The original notebook assigned this tuple to a single DataFrame column,
    which duplicated the values; callers should unpack it explicitly.
    """
    return positive_correction_rates(
        classic_results, light_results, lower_thresh, upper_thresh
    )
