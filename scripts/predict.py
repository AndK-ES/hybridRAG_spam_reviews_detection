#!/usr/bin/env python
"""Score reviews with the RoBERTa-LoRA + FAISS + LLM pipeline.

Modes
-----
* default: full RAG (retrieval + LLM arbitration);
* ``--classifier-only``: no retrieval, no LLM (fast smoke test);
* ``--interpolation-only``: retrieval + k-NN interpolation, no LLM;
* ``--light-prompt``: LLM without retrieval evidence.

Examples
--------
Score a single review with the full RAG pipeline::

    python scripts/predict.py --text "Amazing hotel, best ever!!!"

Score a CSV column and write the per-review decisions::

    python scripts/predict.py --csv reviews.csv --column text --output scored.csv

Run only the classifier::

    python scripts/predict.py --csv reviews.csv --classifier-only

Requirements
------------
Retrieval modes need ``data/faiss/Yelp_NYC_ZIP.parquet`` and
``data/faiss/Yelp_NYC_ZIP_roberta_index.faiss`` (built by notebook ``03``).
LLM modes need a running Ollama server and a pulled chat model, for example
``ollama pull llama3.1:8b-instruct-q8_0``. The released adapter is loaded from
``models/roberta_lora_adapter`` unless ``--adapter`` is supplied.

The output keeps ``original_review``, ``p_clf``, ``p_knn``, ``p_base``,
``clf_label``, ``llm_label`` and ``llm_response``. ``p_knn`` is ``None`` when
retrieval is disabled; confident reviews contain a "Skipped LLM" response.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  

from spam_detection.classifier import load_roberta_model, predict_roberta_proba  
from spam_detection.config import (  
    EMBEDDING_MODEL,
    K_NEIGHBORS,
    LLM_MODEL,
    LORA_ADAPTER_PATH,
    VECTOR_DB_INDEX,
    VECTOR_DB_PARQUET,
)
from spam_detection.embeddings import encode_texts  
from spam_detection.pipeline import run_ablation_study, run_hybrid_pipeline  
from spam_detection.prompts import create_hybrid_chain, create_light_chain  
from spam_detection.retrieval import ( 
    load_labeled_faiss_db,
    precompute_faiss_summaries,
)
from spam_detection.utils import set_seed  

OUTPUT_COLUMNS = [
    "original_review",
    "p_clf",
    "p_knn",
    "p_base",
    "clf_label",
    "llm_label",
    "llm_response",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="review text (repeatable)")
    source.add_argument("--csv", type=Path, help="CSV file with a review-text column")
    parser.add_argument("--column", default="text", help="text column for --csv (default: text)")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N CSV rows")
    parser.add_argument("--adapter", default=str(LORA_ADAPTER_PATH), help="PEFT adapter directory")
    parser.add_argument("--device", default=None, help="torch device (cuda / cpu)")
    parser.add_argument("--encoder", default=EMBEDDING_MODEL, help="sentence-transformers query encoder")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS, help="number of FAISS neighbours")
    parser.add_argument("--llm-model", default=LLM_MODEL, help="Ollama model tag")
    parser.add_argument("--classifier-only", action="store_true", help="skip retrieval and LLM")
    parser.add_argument("--interpolation-only", action="store_true", help="k-NN interpolation, no LLM")
    parser.add_argument("--light-prompt", action="store_true", help="LLM without retrieval evidence")
    parser.add_argument("--output", type=Path, default=None, help="write decisions to this CSV")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args(argv)


def load_reviews(args: argparse.Namespace) -> pd.DataFrame:
    if args.text:
        return pd.DataFrame({"text": args.text})
    frame = pd.read_csv(args.csv)
    if args.column not in frame.columns:
        raise SystemExit(f"Column {args.column!r} not found in {args.csv}; columns: {list(frame.columns)}")
    if args.limit is not None:
        frame = frame.head(args.limit)
    return frame.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    frame = load_reviews(args)
    if args.text:
        args.column = "text"
    texts = frame[args.column if args.column in frame.columns else "text"].astype(str).tolist()
    labels = frame["label"].tolist() if "label" in frame.columns else [None] * len(texts)

    print(f"Scoring {len(texts)} review(s) with adapter {args.adapter}")
    model, tokenizer = load_roberta_model(args.adapter, device=args.device)
    probabilities = predict_roberta_proba(texts, model, tokenizer)

    if args.classifier_only:
        results = run_ablation_study(
            df=frame,
            clf_probas=probabilities,
            true_labels=labels,
            use_llm=False,
            use_retrieval=False,
            use_interpolation=False,
            show_progress=False,
            text_col=args.column,
        )
    elif args.light_prompt:
        results = run_ablation_study(
            df=frame,
            clf_probas=probabilities,
            true_labels=labels,
            llm_chain=create_light_chain(model_name=args.llm_model),
            use_llm=True,
            use_retrieval=False,
            use_interpolation=False,
            show_progress=False,
            text_col=args.column,
        )
    else:
        embeddings = encode_texts(
            texts,
            model_name=args.encoder,
            device=args.device,
            normalize=True,
        )
        faiss_db = load_labeled_faiss_db(VECTOR_DB_PARQUET, VECTOR_DB_INDEX)
        summaries = precompute_faiss_summaries(
            embeddings,
            faiss_db,
            k_neighbors=args.k,
            show_progress=False,
        )

        if args.interpolation_only:
            results = run_ablation_study(
                df=frame,
                clf_probas=probabilities,
                true_labels=labels,
                faiss_summaries=summaries,
                use_llm=False,
                use_retrieval=False,
                use_interpolation=True,
                show_progress=False,
                text_col=args.column,
            )
        else:
            results = run_hybrid_pipeline(
                df=frame,
                clf_probas=probabilities,
                true_labels=labels,
                llm_chain=create_hybrid_chain(model_name=args.llm_model),
                faiss_summaries=summaries,
                k_neighbors=args.k,
                show_progress=False,
                text_col=args.column,
            )

    columns = [column for column in OUTPUT_COLUMNS if column in results.columns]
    printable = results[columns].copy()
    printable["original_review"] = printable["original_review"].str.slice(0, 120)
    print(printable.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Wrote {len(results)} row(s) to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
