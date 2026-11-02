"""FAISS retrieval over the labelled Yelp knowledge base."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .config import K_NEIGHBORS
from .embeddings import normalize_l2
from .utils import as_int_label


def build_faiss_index(
    embeddings: np.ndarray,
    normalize: bool = True,
    index_path: str | None = None,
) -> Any:
    """Build an exact inner-product FAISS index from embeddings.

    Returns the ``faiss.Index`` instance and optionally writes it to
    ``index_path``.
    """
    import faiss

    vectors = normalize_l2(embeddings) if normalize else np.ascontiguousarray(
        embeddings, dtype=np.float32
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    if index_path:
        faiss.write_index(index, str(index_path))
    return index


def load_faiss_index(index_path: str) -> Any:
    """Read a FAISS index from disk."""
    import faiss

    return faiss.read_index(str(index_path))


def load_labeled_faiss_db(
    parquet_path: str,
    index_path: str,
    columns: Sequence[str] = ("review_text", "label"),
    index_name: str = "embeddings",
) -> Any:
    """Load the labelled review corpus and attach the FAISS index.

    The result is a Hugging Face ``datasets.Dataset`` supporting
    ``get_nearest_examples``.
    """
    from datasets import Dataset

    dataset = Dataset.from_parquet(str(parquet_path))
    dataset = dataset.select_columns(list(columns))
    dataset.load_faiss_index(index_name, str(index_path))
    return dataset


def build_faiss_knowledge_base(
    frame: Any,
    embeddings: np.ndarray,
    parquet_path: str,
    index_path: str,
    normalize: bool = True,
) -> Any:
    """Persist the review corpus as parquet and build/write the FAISS index."""
    from pathlib import Path

    parquet = Path(parquet_path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet, index=False)
    return build_faiss_index(embeddings, normalize=normalize, index_path=index_path)


def _as_fake_label(value: Any) -> int:
    """Return 1 for fake/spam, 0 for real; unknown labels count as real."""
    parsed = as_int_label(value)
    if parsed is not None:
        return parsed
    token = str(value).strip().lower()
    return 1 if token in {"fake", "spam", "deceptive", "cg"} else 0


def summary_from_faiss(
    faiss_ds: Any,
    embedding: np.ndarray,
    k: int = K_NEIGHBORS,
    text_col: str = "review_text",
    label_col: str = "label",
    index_name: str = "embeddings",
    max_example_chars: int = 200,
) -> dict[str, Any]:
    """Summarise the neighbourhood of one review in the knowledge base.

    ``top1_score``/``mean_topk_score`` are cosine similarities (higher means
    more similar) because the released index is an inner-product index over
    L2-normalised vectors.
    """
    query = np.asarray(embedding, dtype=np.float32).reshape(-1)
    scores, samples = faiss_ds.get_nearest_examples(index_name, query, k=k)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    if scores.size == 0:
        return {
            "top1_score": None,
            "mean_topk_score": None,
            "std_topk_score": None,
            "score_gap": None,
            "neighbor_count": 0,
            "fake_ratio": 0.0,
            "top3_examples": ["(no example available)"] * 3,
        }

    labels = list(samples.get(label_col, [0] * len(scores)))
    binary_labels = [_as_fake_label(label) for label in labels]
    fake_ratio = float(np.mean(binary_labels)) if binary_labels else 0.0

    top1_score = float(scores[0])
    mean_score = float(scores.mean())
    std_score = float(scores.std())
    score_gap = float(mean_score - top1_score)

    texts = list(samples.get(text_col, [""] * len(scores)))[:3]
    example_labels = binary_labels[:3]
    examples = []
    for text, label in zip(texts, example_labels):
        short = str(text)
        if len(short) > max_example_chars:
            short = short[:max_example_chars] + "..."
        examples.append(f"[{'FAKE' if label else 'REAL'}] {short}")
    while len(examples) < 3:
        examples.append("(no example available)")

    return {
        "top1_score": round(top1_score, 4),
        "mean_topk_score": round(mean_score, 4),
        "std_topk_score": round(std_score, 4),
        "score_gap": round(score_gap, 4),
        "neighbor_count": int(scores.size),
        "fake_ratio": round(fake_ratio, 4),
        "top3_examples": examples,
    }


def precompute_faiss_summaries(
    embeddings: Iterable[np.ndarray],
    faiss_db: Any,
    k_neighbors: int = K_NEIGHBORS,
    text_col: str = "review_text",
    label_col: str = "label",
    index_name: str = "embeddings",
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Compute retrieval summaries for a batch of embeddings."""
    embeddings = list(embeddings)
    iterator: Iterable[np.ndarray] = embeddings
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(embeddings, desc="Retrieving FAISS neighbours")

    return [
        summary_from_faiss(
            faiss_db,
            embedding,
            k=k_neighbors,
            text_col=text_col,
            label_col=label_col,
            index_name=index_name,
        )
        for embedding in iterator
    ]


def summary_to_prompt_args(summary: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten a retrieval summary into the fields used by prompt templates."""
    if summary is None:
        return {
            "top1_score": "N/A",
            "mean_topk_score": "N/A",
            "std_topk_score": "N/A",
            "score_gap": "N/A",
            "neighbor_count": 0,
            "fake_ratio": "N/A",
            "retrieval_example_1": "retrieval disabled",
            "retrieval_example_2": "retrieval disabled",
            "retrieval_example_3": "retrieval disabled",
        }

    examples = list(summary.get("top3_examples", []))[:3]
    while len(examples) < 3:
        examples.append("(no example available)")

    return {
        "top1_score": summary.get("top1_score"),
        "mean_topk_score": summary.get("mean_topk_score"),
        "std_topk_score": summary.get("std_topk_score"),
        "score_gap": summary.get("score_gap"),
        "neighbor_count": summary.get("neighbor_count", 0),
        "fake_ratio": summary.get("fake_ratio"),
        "retrieval_example_1": examples[0],
        "retrieval_example_2": examples[1],
        "retrieval_example_3": examples[2],
    }
