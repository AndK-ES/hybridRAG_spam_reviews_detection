"""Project-wide configuration and paths."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    """Return an environment-variable path when set, otherwise ``default``."""
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _env_path("SPAM_DATA_DIR", PROJECT_ROOT / "data")
MODELS_DIR = _env_path("SPAM_MODELS_DIR", PROJECT_ROOT / "models")
RESULTS_DIR = _env_path("SPAM_RESULTS_DIR", PROJECT_ROOT / "results")

PREPARED_DIR = DATA_DIR / "prepared"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
VECTOR_DB_DIR = DATA_DIR / "for_vector_db"
FAISS_DIR = DATA_DIR / "faiss"

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# OpSpam v1.4: 1,600 hotel reviews (800 genuine / 800 deceptive), labels 0/1.
# Source: https://myleott.com/op_spam_v1.4.zip
OPSPAM_RAW_PATH = DATA_DIR / "op_spam_v1_4.csv"

# Kaggle Yelp Fake Review Dataset: 5,000-review sample used in the paper.
# ``flagged`` is N=genuine, Y=fake. Source:
# https://www.kaggle.com/datasets/virenraina/yelp-fake-review-dataset
YELP_SAMPLE_RAW_PATH = DATA_DIR / "new_yelp_sample.csv"

# Hotel Reviews sample: 3,000 reviews, 2,707 genuine / 293 fake. The original
# labels are 1=genuine and -1=fake and are normalised to 0/1. Source:
# https://www.kaggle.com/datasets/abidmeeraj/yelp-labelled-dataset
HOTEL_REVIEWS_PATH = DATA_DIR / "external_hotels_reviews_sample.csv"

# Balanced 3,000-review cross-domain sample used by the paper (1,516 real /
# 1,484 computer-generated). ``CROSS_DOMAIN_FULL_PATH`` is the complete 40,432
# review corpus; ``CROSS_DOMAIN_10K_PATH`` is a larger 10,000-review sample for
# extra experiments and is not the paper evaluation set. Source:
# https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset
CROSS_DOMAIN_PATH = DATA_DIR / "external_data.csv"
CROSS_DOMAIN_FULL_PATH = DATA_DIR / "external_data_full.csv"
CROSS_DOMAIN_10K_PATH = DATA_DIR / "cross_domain_10k.csv"

# Released 80/20 classifier split: 5,231 train / 1,309 test reviews. The
# training file contains two duplicated texts; they are intentionally kept so
# the released numbers remain reproducible. See ``build_training_corpus``.
CLASSIFIER_TRAIN_PATH = PREPARED_DIR / "new_detector_train.csv"
CLASSIFIER_TEST_PATH = PREPARED_DIR / "new_detector_test.csv"
CLASSIFIER_TRAIN_EMBEDDINGS = PREPARED_DIR / "new_roberta_embeddings_train.npy"
CLASSIFIER_TEST_EMBEDDINGS = PREPARED_DIR / "new_roberta_embeddings_test.npy"

OPSPAM_PREPROCESSED_PATH = PREPROCESSED_DIR / "opspam_preprocessed.csv"
YELP_PREPROCESSED_PATH = PREPROCESSED_DIR / "new_yelp_preprocessed.csv"

# Knowledge base: 768,683 labelled YelpNYC + YelpZIP reviews. The parquet file
# is about 290 MB; the released exact inner-product index is about 2.36 GB.
# Source: Liu (2024), ScienceDB DOI 10.57760/sciencedb.17086.
YELPNYC_REVIEWS_PATH = VECTOR_DB_DIR / "YelpNYC" / "YelpNYC.parquet"
YELPZIP_PATH = VECTOR_DB_DIR / "YelpZIP.parquet"
VECTOR_DB_PARQUET = FAISS_DIR / "Yelp_NYC_ZIP.parquet"
VECTOR_DB_INDEX = FAISS_DIR / "Yelp_NYC_ZIP_roberta_index.faiss"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# The released adapter is rank-8 PEFT LoRA on roberta-base: r=8, alpha=16,
# dropout=0.1, target modules query/value, task SEQ_CLS.
BASE_MODEL = os.getenv("SPAM_BASE_MODEL", "roberta-base")
LORA_ADAPTER_PATH = _env_path(
    "SPAM_ADAPTER_DIR", MODELS_DIR / "roberta_lora_adapter"
)

# External-evaluation notebooks encode queries with the sentence-transformer
# below, while the released FAISS index was built with roberta-base. This
# mismatch is preserved to reproduce the published numbers. Pass
# ``model_name="roberta-base"`` to ``encode_texts`` for a consistent variant.
EMBEDDING_MODEL = os.getenv(
    "SPAM_EMBEDDING_MODEL", "sentence-transformers/all-distilroberta-v1"
)

# ---------------------------------------------------------------------------
# RAG / inference defaults
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TAU_LOW = 0.30  # LLM is skipped below this base probability
TAU_HIGH = 0.70  # LLM is skipped above this base probability
K_NEIGHBORS = 20  # neighbours retrieved per review
INTERPOLATION_ALPHA = 0.55  # p_base = alpha*p_clf + (1-alpha)*p_knn
LLM_MODEL = os.getenv("SPAM_LLM_MODEL", "llama3.1:8b-instruct-q8_0")
LLM_TEMPERATURE = 0.1  # classic prompt; the light prompt uses 0.0
MAX_CONCURRENCY = 8  # concurrent Ollama requests

# Tokenisation defaults
MAX_LENGTH_CLASSIFIER = 256
MAX_LENGTH_EMBEDDINGS = 512
EMBEDDING_DIM = 768

# Binary label convention used everywhere: fake/spam is the positive class.
LABEL_NAMES = {0: "real", 1: "fake"}
FAKE_TOKENS = {"1", "fake", "spam", "deceptive", "cg", "y", "yes", "true"}
REAL_TOKENS = {"0", "real", "genuine", "or", "n", "no", "false"}


def resolve_path(path: str | os.PathLike[str], base: Path | None = None) -> Path:
    """Resolve ``path`` relative to ``base`` (repository root by default)."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ((base or PROJECT_ROOT) / candidate).resolve()


def ensure_directories() -> None:
    """Create the directories that generated artefacts are written to."""
    for directory in (PREPARED_DIR, PREPROCESSED_DIR, FAISS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
