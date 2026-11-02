"""Reusable components for the hybrid RAG fake-review detection system.

Heavy dependencies (PyTorch, Transformers, FAISS, LangChain) are imported
lazily by the submodules, so ``import spam_detection`` stays lightweight.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.0.0"

# Public names grouped by the module that implements them.
_EXPORTS = {
    # configuration and helpers
    "PROJECT_ROOT": "config",
    "DATA_DIR": "config",
    "MODELS_DIR": "config",
    "RESULTS_DIR": "config",
    "TAU_LOW": "config",
    "TAU_HIGH": "config",
    "K_NEIGHBORS": "config",
    "INTERPOLATION_ALPHA": "config",
    "RANDOM_SEED": "config",
    "resolve_path": "config",
    "ensure_directories": "config",
    "set_seed": "utils",
    "as_int_label": "utils",
    "normalize_label_series": "utils",
    "parse_llm_label": "utils",
    "extract_fake_ratio": "utils",
    "to_serializable": "utils",
    "ensure_dir": "utils",
    "chunked": "utils",
    # preprocessing
    "clean_text_basic": "preprocessing",
    "remove_empty_reviews": "preprocessing",
    "filter_short_reviews": "preprocessing",
    "prepare_opspam": "preprocessing",
    "prepare_yelp_sample": "preprocessing",
    "stratified_split": "preprocessing",
    "build_training_corpus": "preprocessing",
    "prepare_yelp_nyc": "preprocessing",
    "prepare_yelp_zip": "preprocessing",
    "build_vector_db_frame": "preprocessing",
    "sample_external_dataset": "preprocessing",
    # embeddings
    "get_device": "embeddings",
    "mean_pooling": "embeddings",
    "load_roberta_encoder": "embeddings",
    "get_roberta_embeddings": "embeddings",
    "encode_texts": "embeddings",
    "normalize_l2": "embeddings",
    # classifier
    "create_lora_classifier": "classifier",
    "load_roberta_model": "classifier",
    "predict_roberta_proba": "classifier",
    "predict_roberta_labels": "classifier",
    "fine_tune_adapter": "classifier",
    # retrieval
    "build_faiss_index": "retrieval",
    "load_faiss_index": "retrieval",
    "load_labeled_faiss_db": "retrieval",
    "summary_from_faiss": "retrieval",
    "precompute_faiss_summaries": "retrieval",
    "summary_to_prompt_args": "retrieval",
    # pipeline
    "base_probability": "pipeline",
    "route_label": "pipeline",
    "run_ablation_study": "pipeline",
    "run_hybrid_pipeline": "pipeline",
    # evaluation
    "binary_metrics": "evaluation",
    "predictions_from_results": "evaluation",
    "comparison_table": "evaluation",
    "gray_zone_analysis": "evaluation",
    "save_table": "evaluation",
    # statistics
    "mcnemar_test": "statistics",
    "summarize_table": "statistics",
    "bin_gain_analysis": "statistics",
    "bootstrap_confidence_test": "statistics",
    "plot_bootstrap_results": "statistics",
    "positive_correction_rate": "statistics",
    "positive_correction_rates": "statistics",
    "override_rate": "statistics",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Import a public helper from its implementing submodule on demand."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
