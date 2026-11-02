"""Text cleaning and dataset preprocessing."""

from __future__ import annotations

import re
from typing import Any

try:  # pandas is imported lazily in the helpers below to keep pure text
    import pandas as pd  # noqa: F401  (used for type hints / runtime calls)
except ImportError:  # pragma: no cover - pandas is a declared dependency
    pd = None  # type: ignore[assignment]

from .utils import normalize_label_series

_HTML_TAG = re.compile(r"<.*?>")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9\s]")
_MULTI_SPACE = re.compile(r"\s+")

UNIFIED_COLUMNS = ["id", "source", "domain", "text", "label"]


def clean_text_basic(text: Any, keep_case: bool = True) -> str:
    """Remove HTML tags, punctuation and redundant whitespace.

    Parameters
    ----------
    text:
        Raw review text. Non-string and missing values become ``""``.
    keep_case:
        When ``True`` (the default used in the paper) capitalisation is kept,
        because all-caps words and punctuation can carry spam signals.
    """
    if text is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(text):
            return ""
    except (ImportError, TypeError, ValueError):
        pass
    if not isinstance(text, str):
        text = str(text)

    text = _HTML_TAG.sub("", text)
    text = _NON_ALNUM.sub("", text)
    if not keep_case:
        text = text.lower()
    return _MULTI_SPACE.sub(" ", text).strip()


def remove_empty_reviews(
    df: Any,
    column: str,
    min_length: int = 3,
    verbose: bool = True,
) -> Any:
    """Drop rows whose review text is null, empty, ``"0"`` or too short.

    The original external-evaluation notebooks call this with ``min_length=3``.
    Do not raise that value to 30 for those experiments: the extra filter
    removes 47 Hotel Reviews rows and changes the published metrics.
    """
    initial_size = len(df)
    series = df[column]

    if verbose:
        null_count = int(series.isna().sum())
        empty_count = int((series.fillna("") == "").sum())
        zero_count = int((series.astype(str).str.strip() == "0").sum())
        short_count = int((series.fillna("").str.len() < min_length).sum())
        print(f"Initial rows: {initial_size}")
        print(f"  null/None: {null_count}")
        print(f"  empty strings: {empty_count}")
        print(f"  literal '0': {zero_count}")
        print(f"  shorter than {min_length} characters: {short_count}")

    mask = (
        series.notna()
        & (series.fillna("").str.strip() != "")
        & (series.astype(str).str.strip() != "0")
        & (series.fillna("").str.len() >= min_length)
    )
    cleaned = df.loc[mask].copy()

    if verbose:
        removed = initial_size - len(cleaned)
        pct = removed / initial_size * 100 if initial_size else 0.0
        print(f"Removed {removed} rows ({pct:.2f}%).")
    return cleaned


def filter_short_reviews(df: Any, min_chars: int = 30, column: str = "text") -> Any:
    """Keep reviews with at least ``min_chars`` characters.

    This 30-character filter belongs to the classifier-corpus recipe. The
    external Hotel Reviews and cross-domain experiments deliberately use only
    ``remove_empty_reviews(..., min_length=3)`` instead.
    """
    return df.loc[df[column].fillna("").astype(str).str.len() >= min_chars].copy()


def prepare_opspam(df: Any) -> Any:
    """Normalise the Deceptive Opinion Spam Corpus v1.4."""
    out = df.copy()
    out["id"] = "opspam_" + out.index.astype(str)
    out["source"] = "op_spam"
    out["domain"] = "hotel"
    out["label"] = normalize_label_series(out["label"], missing="drop")
    out = out.loc[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    return out[UNIFIED_COLUMNS].reset_index(drop=True)


def prepare_yelp_sample(df: Any, text_column: str = "reviewContent") -> Any:
    """Normalise the Kaggle Yelp Fake Review Dataset.

    ``flagged`` uses ``N`` for genuine and ``Y`` for spam; the text is cleaned
    with :func:`clean_text_basic` and the case is preserved.
    """
    out = remove_empty_reviews(df, text_column, min_length=3, verbose=False)
    out = out.copy()
    out["text"] = out[text_column].apply(clean_text_basic)
    out["label"] = normalize_label_series(out["flagged"], missing="drop")
    out = out.loc[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    out["id"] = "yelp_" + out.index.astype(str)
    out["source"] = "yelp"
    out["domain"] = "restaurant"
    return out[UNIFIED_COLUMNS].reset_index(drop=True)


def stratified_split(
    df: Any,
    test_size: float = 0.2,
    seed: int = 42,
    label_col: str = "label",
) -> tuple[Any, Any]:
    """Return a reproducible label-stratified train/test split."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[label_col],
    )


def build_training_corpus(
    opspam_df: Any,
    yelp_df: Any,
    test_size: float = 0.2,
    min_chars: int = 30,
    seed: int = 42,
) -> tuple[Any, Any]:
    """Build the combined RoBERTa-LoRA training and test corpora.

    Both source datasets are normalised, split separately with stratification,
    concatenated and finally filtered to reviews of at least ``min_chars``.
    Splitting per source preserves the contribution of each domain.

    With the released inputs and ``test_size=0.2, min_chars=30`` this returns
    5,231 training rows and 1,309 test rows, matching the shipped CSVs. Two
    duplicate texts remain in training and are kept for reproducibility.
    """
    opspam = prepare_opspam(opspam_df)
    yelp = prepare_yelp_sample(yelp_df)

    opspam_train, opspam_test = stratified_split(opspam, test_size, seed)
    yelp_train, yelp_test = stratified_split(yelp, test_size, seed)

    train = pd.concat([yelp_train, opspam_train], ignore_index=True)
    test = pd.concat([yelp_test, opspam_test], ignore_index=True)

    train = filter_short_reviews(train, min_chars=min_chars).reset_index(drop=True)
    test = filter_short_reviews(test, min_chars=min_chars).reset_index(drop=True)
    return train, test


def prepare_yelp_nyc(reviews_df: Any, metadata_df: Any) -> Any:
    """Merge YelpNYC review text with its rating and fake/real label."""
    reviews = (
        reviews_df[["Review_id", "Review_Date", "Review_Text"]]
        .drop_duplicates(subset="Review_id")
        .copy()
    )
    metadata = (
        metadata_df[["Review_id", "Rating", "Label"]]
        .drop_duplicates(subset="Review_id")
        .copy()
    )
    merged = reviews.merge(metadata, on="Review_id", how="inner").dropna()
    merged = merged.rename(
        columns={
            "Review_Date": "review_date",
            "Review_Text": "review_text",
            "Rating": "rating",
            "Label": "label",
        }
    )
    # YelpNYC encodes fake reviews as -1 and genuine reviews as +1. Mapping
    # explicitly avoids treating the numeric +1 (genuine) as the generic
    # binary label 1 (fake).
    merged["label"] = merged["label"].map({-1: 1, 1: 0})
    merged = merged.loc[merged["label"].notna()].copy()
    merged["label"] = merged["label"].astype(int).map({0: "real", 1: "fake"})
    return merged[["review_date", "review_text", "rating", "label"]]


def prepare_yelp_zip(df: Any) -> Any:
    """Normalise the YelpZIP CSV (tag ``fake``/``real``)."""
    out = df.copy()
    out = out.rename(
        columns={"date": "review_date", "text": "review_text", "tag": "label"}
    )
    out = out[["review_date", "review_text", "rating", "label"]].dropna()
    out["rating"] = out["rating"].astype(int)
    out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce")
    out["label"] = normalize_label_series(out["label"], missing="drop")
    out = out.loc[out["label"].notna()].copy()
    out["label"] = out["label"].map({0: "real", 1: "fake"})
    return out


def build_vector_db_frame(nyc_df: Any, yelpzip_df: Any) -> Any:
    """Combine YelpNYC and YelpZIP into the labelled FAISS knowledge base."""
    combined = pd.concat([nyc_df, yelpzip_df], ignore_index=True)
    combined = combined.reset_index(drop=True)
    combined.insert(0, "id", range(len(combined)))
    combined["review_date"] = pd.to_datetime(
        combined["review_date"], errors="coerce"
    )
    return combined[["id", "review_date", "review_text", "rating", "label"]]


def sample_external_dataset(
    df: Any,
    n_samples: int = 3000,
    seed: int = 42,
    label_column: str = "label",
    text_column: str = "text",
) -> Any:
    """Sample an external evaluation dataset and normalise its labels."""
    out = df.sample(n=min(n_samples, len(df)), random_state=seed, replace=False).copy()
    out["label"] = normalize_label_series(out[label_column], missing="drop")
    out = out.loc[out["label"].notna(), [text_column, "label"]].copy()
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)
