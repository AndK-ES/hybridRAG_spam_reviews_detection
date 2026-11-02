"""RoBERTa embedding extraction."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .config import (
    BASE_MODEL,
    EMBEDDING_MODEL,
    MAX_LENGTH_EMBEDDINGS,
)


def get_device(preferred: str | None = None) -> str:
    """Return ``cuda`` when available, otherwise ``cpu``."""
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(hidden_states: Any, attention_mask: Any) -> Any:
    """Average token representations while ignoring padding tokens."""
    import torch

    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def load_roberta_encoder(
    model_name: str = BASE_MODEL,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Load ``roberta-base`` and its tokenizer for feature extraction."""
    from transformers import AutoModel, AutoTokenizer

    resolved_device = get_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(resolved_device)
    model.eval()
    return model, tokenizer


def get_roberta_embeddings(
    texts: Iterable[str],
    model_name: str = BASE_MODEL,
    batch_size: int = 32,
    max_length: int = MAX_LENGTH_EMBEDDINGS,
    device: str | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Extract 768-dimensional mean-pooled RoBERTa embeddings.

    The model/tokenizer can be supplied to avoid reloading the encoder when
    several batches of text are encoded (for example train and test sets).
    """
    import torch
    from tqdm.auto import tqdm

    texts = list(texts)
    if model is None or tokenizer is None:
        model, tokenizer = load_roberta_encoder(model_name, device)
    resolved_device = next(model.parameters()).device

    embeddings: list[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting RoBERTa embeddings")

    with torch.no_grad():
        for start in iterator:
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(resolved_device)
            outputs = model(**encoded)
            pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu().numpy().astype(np.float32))

    if not embeddings:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack(embeddings)


def encode_texts(
    texts: Iterable[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
    device: str | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Encode texts with a Sentence-Transformers model.

    This is the query encoder used for the external evaluation sets in the
    original experiments (``all-distilroberta-v1``). The FAISS index itself was
    built with ``roberta-base``; the mismatch is intentional for reproduction.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=get_device(device))
    embeddings = model.encode(
        list(texts),
        show_progress_bar=False,
        batch_size=batch_size,
        normalize_embeddings=normalize,
    )
    return np.asarray(embeddings, dtype=np.float32)


def normalize_l2(embeddings: np.ndarray) -> np.ndarray:
    """Return an L2-normalised float32 copy suitable for inner-product search."""
    array = np.ascontiguousarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms
