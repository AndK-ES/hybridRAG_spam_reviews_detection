"""Small, dependency-light helpers shared across the project.

Covers label normalisation across heterogeneous CSV encodings (numeric 0/1,
``-1/+1`` used by YelpNYC and some Yelp datasets, ``N/Y`` used by the Kaggle
Yelp sample, and ``CG/OR`` used by the cross-domain corpus), tolerant LLM
output parsing for the required ``LABEL:``/``REASONING:`` format, and
reproducibility seeding.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

from .config import FAKE_TOKENS, RANDOM_SEED, REAL_TOKENS


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and PyTorch (when available) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # local import keeps this helper lightweight
    except ImportError:
        return
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_int_label(value: Any) -> int | None:
    """Normalise a label from several dataset encodings to ``0`` or ``1``.

    Supports numeric labels, strings such as ``"fake"``/``"real"``, the
    abbreviations used by the cross-domain dataset (``CG`` = computer
    generated, ``OR`` = original review) and YelpNYC's ``-1`` fake label.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return int(value)
        # Several Yelp-derived datasets encode fake reviews as -1 and
        # genuine reviews as +1 (YelpNYC, Labelled Yelp Dataset).
        if value == -1:
            return 1
    token = str(value).strip().lower()
    if token in FAKE_TOKENS:
        return 1
    if token in REAL_TOKENS:
        return 0
    return None


def normalize_label_series(series: Any, missing: str = "raise") -> Any:
    """Vectorised version of :func:`as_int_label` for a pandas Series."""
    mapped = series.map(as_int_label)
    if missing == "raise" and mapped.isna().any():
        unknown = series[mapped.isna()].unique().tolist()
        raise ValueError(f"Unknown label values: {unknown}")
    if missing == "drop":
        mapped = mapped.dropna()
    return mapped.astype("Int64")


def parse_llm_label(response: str, fallback: int = 0) -> int:
    """Extract ``LABEL: 0|1`` from an LLM response.

    The parser is intentionally tolerant: it accepts surrounding markdown,
    extra whitespace and both ``LABEL: 1`` and ``label = 1`` styles.
    """
    if not isinstance(response, str):
        return fallback
    match = re.search(r"LABEL\s*[:=]\s*([01])", response, flags=re.IGNORECASE)
    return int(match.group(1)) if match else fallback


def extract_fake_ratio(value: Any) -> float:
    """Read ``fake_ratio`` from a serialised FAISS summary or dict."""
    if isinstance(value, dict):
        return float(value.get("fake_ratio", 0.0))
    if isinstance(value, str):
        try:
            parsed = json.loads(value.replace("'", '"'))
            if isinstance(parsed, dict) and "fake_ratio" in parsed:
                return float(parsed["fake_ratio"])
        except (ValueError, TypeError):
            pass
        match = re.search(r"['\"]fake_ratio['\"]\s*:\s*([\d.]+)", value)
        if match:
            return float(match.group(1))
    return 0.0


def to_serializable(value: Any) -> Any:
    """Convert NumPy scalars/arrays and paths into JSON-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (ValueError, AttributeError):
            pass
    return value


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (including parents) and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def chunked(iterable: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Yield consecutive lists of at most ``size`` items."""
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
