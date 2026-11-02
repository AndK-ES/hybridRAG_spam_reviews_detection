"""Inference pipeline and ablation configurations.

1. use_llm=False, use_interpolation=False -> RoBERTa-LoRA only
2. use_llm=False, use_interpolation=True  -> classifier + k-NN interpolation
3. use_llm=True,  use_interpolation=False -> full RAG
4. use_llm=True,  use_interpolation=True  -> full RAG + interpolation
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Sequence

import numpy as np

from .config import (
    INTERPOLATION_ALPHA,
    K_NEIGHBORS,
    MAX_CONCURRENCY,
    TAU_HIGH,
    TAU_LOW,
)
from .retrieval import (
    precompute_faiss_summaries,
    summary_from_faiss,
    summary_to_prompt_args,
)
from .utils import as_int_label, parse_llm_label


def _texts_from(df: Any, text_col: str) -> list[str]:
    """Extract review texts from a pandas DataFrame or a HF dataset."""
    if df is None:
        raise ValueError("df is required to read the review texts")
    column = df[text_col]
    if hasattr(column, "tolist"):
        return column.tolist()
    return list(column)


def _labels_from(df: Any, true_labels: Iterable[Any] | None, label_col: str) -> list[Any]:
    """Return normalised true labels, using ``true_labels`` when provided."""
    values = list(true_labels) if true_labels is not None else list(df[label_col])
    normalised = []
    for value in values:
        parsed = as_int_label(value)
        normalised.append(parsed if parsed is not None else value)
    return normalised


def base_probability(
    p_clf: float,
    p_knn: float | None,
    alpha: float = INTERPOLATION_ALPHA,
    use_interpolation: bool = False,
) -> float:
    """Combine classifier and k-NN probabilities when interpolation is on."""
    if use_interpolation and p_knn is not None:
        return alpha * p_clf + (1.0 - alpha) * p_knn
    return p_clf


def route_label(
    p_base: float,
    lower_thresh: float = TAU_LOW,
    upper_thresh: float = TAU_HIGH,
) -> tuple[int, bool]:
    """Return ``(base_label, needs_llm)`` for a base probability."""
    label = int(p_base >= 0.5)
    return label, bool(lower_thresh <= p_base <= upper_thresh)


def _invoke_chain(chain: Any, args: dict[str, Any]) -> tuple[str, int | None]:
    """Call the LLM chain and parse its label, keeping errors as text."""
    try:
        response = chain.invoke(args)
    except Exception as exc:  # keep the pipeline running on provider errors
        return f"LLM Error: {exc}", None

    if not isinstance(response, str):
        response = str(response)
    label = parse_llm_label(response, fallback=-1)
    return response, (label if label in (0, 1) else None)


def run_ablation_study(
    df: Any,
    clf_probas: Sequence[float] | None = None,
    true_labels: Iterable[Any] | None = None,
    llm_chain: Any | None = None,
    faiss_summaries: Sequence[dict[str, Any] | None] | None = None,
    embeddings: Sequence[np.ndarray] | None = None,
    faiss_db: Any | None = None,
    lower_thresh: float = TAU_LOW,
    upper_thresh: float = TAU_HIGH,
    alpha: float = INTERPOLATION_ALPHA,
    k_neighbors: int = K_NEIGHBORS,
    use_interpolation: bool = False,
    use_retrieval: bool = True,
    use_llm: bool = True,
    max_concurrency: int = MAX_CONCURRENCY,
    text_col: str = "text",
    label_col: str = "label",
    show_progress: bool = True,
    **kwargs: Any,
) -> Any:
    """Run one ablation configuration and return a result DataFrame.

    Parameters
    ----------
    faiss_summaries:
        Optional pre-computed retrieval summaries (one per review). When
        omitted and ``embeddings``/``faiss_db`` are supplied, summaries are
        computed on demand.
    """
    import pandas as pd

    if clf_probas is not None and np.asarray(clf_probas).ndim > 1:
        # Original positional order:
        # (df, embeddings, clf_probas, true_labels, faiss_db, llm_chain).
        legacy_chain = embeddings
        embeddings = clf_probas
        clf_probas = true_labels
        true_labels = llm_chain
        faiss_db = faiss_summaries
        llm_chain = legacy_chain
        faiss_summaries = None

    if clf_probas is None and "clf_proba" in kwargs:
        clf_probas = kwargs.pop("clf_proba")
    if clf_probas is None and "p_clf" in kwargs:
        clf_probas = kwargs.pop("p_clf")
    if faiss_summaries is None and "faiss_summary" in kwargs:
        faiss_summaries = kwargs.pop("faiss_summary")
    if faiss_db is None and "faiss_df" in kwargs:
        faiss_db = kwargs.pop("faiss_df")

    texts = _texts_from(df, text_col)
    labels = _labels_from(
        df, kwargs.pop("y_true", true_labels), label_col
    )
    probabilities = [float(value) for value in clf_probas]

    if not (len(texts) == len(labels) == len(probabilities)):
        raise ValueError(
            "df, clf_probas and true_labels must have the same length "
            f"({len(texts)}, {len(probabilities)}, {len(labels)})"
        )

    need_retrieval = use_interpolation or (use_llm and use_retrieval)

    if faiss_summaries is None:
        if need_retrieval and faiss_db is not None and embeddings is not None:
            faiss_summaries = precompute_faiss_summaries(
                embeddings,
                faiss_db,
                k_neighbors=k_neighbors,
                show_progress=show_progress,
            )
        else:
            faiss_summaries = [None] * len(texts)

    results: list[dict[str, Any]] = []
    llm_tasks: list[dict[str, Any]] = []

    for text, p_clf, true_label, summary in zip(
        texts, probabilities, labels, faiss_summaries
    ):
        p_knn = None
        active_summary = None
        if need_retrieval and summary is not None:
            p_knn = float(summary.get("fake_ratio", 0.0))
            active_summary = summary

        p_base = base_probability(p_clf, p_knn, alpha, use_interpolation)
        clf_label, uncertain = route_label(p_base, lower_thresh, upper_thresh)

        row: dict[str, Any] = {
            "original_review": text,
            "p_clf": round(p_clf, 4),
            "p_knn": round(p_knn, 4) if p_knn is not None else None,
            "p_base": round(p_base, 4),
            "clf_label": clf_label,
            "faiss_summary": active_summary,
            "llm_label": None,
            "llm_response": None,
            "true_label": true_label,
        }

        if not use_llm:
            if use_interpolation:
                row["llm_response"] = (
                    f"Interpolation (alpha={alpha}): p_clf={p_clf:.3f}"
                )
            else:
                row["llm_response"] = (
                    "Only RoBERTa-LoRA (LLM & Interpolation disabled)"
                )
            results.append(row)
            continue

        if not uncertain:
            row["llm_label"] = clf_label
            row["llm_response"] = (
                "Skipped LLM: High Classifier Confidence "
                f"({'Real' if clf_label == 0 else 'Fake'}) [p_base={p_base:.3f}]"
            )
            results.append(row)
            continue

        invoke_args = {"p_clf": round(p_base, 3), "clf_label": clf_label, "text": text}
        if use_retrieval and active_summary is not None:
            invoke_args.update(summary_to_prompt_args(active_summary))
        else:
            invoke_args.update(summary_to_prompt_args(None))

        llm_tasks.append(
            {"result_idx": len(results), "args": invoke_args, "fallback": clf_label}
        )
        results.append(row)

    if llm_tasks:
        if llm_chain is None:
            import pandas as pd

            raise ValueError("llm_chain is required when use_llm=True")

        iterator: Iterable[dict[str, Any]] = llm_tasks
        if show_progress:
            from tqdm.auto import tqdm

            iterator = tqdm(llm_tasks, desc="LLM arbitration")

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            outputs = list(
                executor.map(lambda task: _invoke_chain(llm_chain, task["args"]), iterator)
            )

        for task, (response, label) in zip(llm_tasks, outputs):
            results[task["result_idx"]]["llm_response"] = response
            results[task["result_idx"]]["llm_label"] = (
                label if label is not None else task["fallback"]
            )

    return pd.DataFrame(results)


def run_hybrid_pipeline(
    df: Any,
    clf_probas: Sequence[float],
    true_labels: Iterable[Any] | None = None,
    llm_chain: Any | None = None,
    faiss_summaries: Sequence[dict[str, Any] | None] | None = None,
    embeddings: Sequence[np.ndarray] | None = None,
    faiss_db: Any | None = None,
    lower_thresh: float = TAU_LOW,
    upper_thresh: float = TAU_HIGH,
    k_neighbors: int = K_NEIGHBORS,
    **kwargs: Any,
) -> Any:
    """Run the full RAG configuration (retrieval + LLM arbitration)."""
    return run_ablation_study(
        df=df,
        clf_probas=clf_probas,
        true_labels=true_labels,
        llm_chain=llm_chain,
        faiss_summaries=faiss_summaries,
        embeddings=embeddings,
        faiss_db=faiss_db,
        lower_thresh=lower_thresh,
        upper_thresh=upper_thresh,
        k_neighbors=k_neighbors,
        use_interpolation=False,
        use_retrieval=True,
        use_llm=True,
        **kwargs,
    )


def build_retrieval_summary(
    faiss_db: Any,
    embedding: np.ndarray,
    k_neighbors: int = K_NEIGHBORS,
) -> dict[str, Any]:
    """Convenience wrapper around :func:`summary_from_faiss`."""
    return summary_from_faiss(faiss_db, embedding, k=k_neighbors)
