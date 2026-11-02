"""RoBERTa-LoRA classifier."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .config import (
    BASE_MODEL,
    LORA_ADAPTER_PATH,
    MAX_LENGTH_CLASSIFIER,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------
def get_device(preferred: str | None = None) -> str:
    """Return ``preferred`` or the best available torch device name."""
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_base_classifier(base_model: str) -> Any:
    """Load a two-label RoBERTa head without the expected loading report.

    ``roberta-base`` is a masked-language-model checkpoint, so its classification
    head is absent by design. The report produced while loading it is therefore
    noise: the trained head arrives with the PEFT adapter in
    :func:`load_roberta_model`.
    """
    from transformers import RobertaForSequenceClassification
    from transformers.utils import logging as hf_logging

    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        return RobertaForSequenceClassification.from_pretrained(
            base_model, num_labels=2
        )
    finally:
        hf_logging.set_verbosity(previous_verbosity)


def load_roberta_model(
    model_path: str | Any = LORA_ADAPTER_PATH,
    base_model: str = BASE_MODEL,
    device: str | None = None,
    trainable: bool = False,
    is_trainable: bool | None = None,
) -> tuple[Any, Any]:
    """Load ``roberta-base`` with a LoRA adapter and its tokenizer.

    Parameters
    ----------
    model_path:
        Directory containing the PEFT adapter (``adapter_config.json`` and
        ``adapter_model.safetensors``). The tokenizer is read from the same
        directory, falling back to ``base_model`` when it is absent.
    trainable:
        When ``True`` the adapter is prepared for further fine-tuning: the
        input embeddings require gradients and the KV cache is disabled.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    if is_trainable is not None:
        trainable = is_trainable
    resolved_device = get_device(device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        # Adapters are often distributed without the (large) tokenizer files.
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    base = _load_base_classifier(base_model)
    model = PeftModel.from_pretrained(base, model_path)
    model.to(resolved_device)

    if trainable:
        model.config.use_cache = False
        model.enable_input_require_grads()
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
    else:
        model.eval()

    return model, tokenizer


def create_lora_classifier(
    base_model: str = BASE_MODEL,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Create a fresh ``roberta-base`` model with a trainable LoRA adapter."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    resolved_device = get_device(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = _load_base_classifier(base_model)
    config = LoraConfig(
        task_type="SEQ_CLS",
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )
    model = get_peft_model(model, config)
    model.to(resolved_device)
    return model, tokenizer


def predict_roberta_proba(
    texts: Iterable[str],
    model: Any,
    tokenizer: Any,
    batch_size: int = 16,
    max_length: int = MAX_LENGTH_CLASSIFIER,
) -> np.ndarray:
    """Return the fake-review probability for every input text."""
    import torch

    texts = list(texts)
    if not texts:
        return np.empty((0,), dtype=np.float32)

    device = next(model.parameters()).device
    probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            probabilities.append(probs.detach().cpu().numpy())

    return np.concatenate(probabilities).astype(np.float32)


def get_predictions(
    df: Any,
    model: Any,
    tokenizer: Any,
    batch_size: int = 16,
) -> np.ndarray:
    """Return fake-review probabilities for a dataframe's ``text`` column."""
    return predict_roberta_proba(
        df["text"].tolist(), model, tokenizer, batch_size=batch_size
    )


def predict_roberta_labels(
    texts: Iterable[str],
    model: Any,
    tokenizer: Any,
    threshold: float = 0.5,
    **kwargs: Any,
) -> np.ndarray:
    """Return binary fake/real predictions at ``threshold``."""
    return (predict_roberta_proba(texts, model, tokenizer, **kwargs) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def prepare_tokenized_dataset(
    df: Any,
    tokenizer: Any,
    text_col: str = "text",
    label_col: str = "label",
    max_length: int = MAX_LENGTH_CLASSIFIER,
) -> Any:
    """Tokenise a dataframe into a Hugging Face dataset with ``labels``."""
    from datasets import Dataset

    dataset = Dataset.from_pandas(
        df[[text_col, label_col]].reset_index(drop=True), preserve_index=False
    )
    dataset = dataset.rename_column(label_col, "labels")

    def tokenize(examples: Mapping[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            examples[text_col], truncation=True, max_length=max_length
        )

    tokenized = dataset.map(tokenize, batched=True)
    columns_to_remove = [text_col]
    if "__index_level_0__" in tokenized.column_names:
        columns_to_remove.append("__index_level_0__")
    tokenized = tokenized.remove_columns(columns_to_remove)
    tokenized.set_format("torch")
    return {"train": tokenized}


def prepare_training_datasets(
    train_df: Any,
    eval_df: Any,
    tokenizer: Any,
    max_length: int = MAX_LENGTH_CLASSIFIER,
) -> dict[str, Any]:
    """Tokenise train/evaluation frames into the dict expected by Trainer."""
    return {
        "train": prepare_tokenized_dataset(
            train_df, tokenizer, max_length=max_length
        )["train"],
        "eval": prepare_tokenized_dataset(
            eval_df, tokenizer, max_length=max_length
        )["train"],
    }


def split_train_validation(
    df: Any,
    test_size: float = 0.15,
    seed: int = RANDOM_SEED,
    label_col: str = "label",
) -> tuple[Any, Any]:
    """Create a stratified train/validation split for adapter training."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[label_col],
    )


def compute_class_weights(labels: Iterable[int]) -> Any:
    """Return balanced class weights as a float tensor."""
    import torch
    from sklearn.utils.class_weight import compute_class_weight

    array = np.asarray(list(labels))
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(array), y=array
    )
    return torch.tensor(weights, dtype=torch.float)


def get_weights(hf_dataset: Any) -> Any:
    """Compute class weights from a tokenised Hugging Face dataset."""
    labels = hf_dataset["train"]["labels"]
    return compute_class_weights(labels)


def _custom_loss_trainer_class() -> type:
    """Build the weighted cross-entropy Trainer subclass on first use."""
    import torch.nn as nn
    from transformers import Trainer

    class CustomLossTrainer(Trainer):
        """Trainer that applies class-weighted cross-entropy."""

        def __init__(self, *args: Any, class_weights: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(
            self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any
        ) -> Any:
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            if self.class_weights is not None:
                weights = self.class_weights.to(logits.device)
                loss_fn = nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fn = nn.CrossEntropyLoss()

            loss = loss_fn(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    return CustomLossTrainer


def __getattr__(name: str) -> Any:
    if name == "CustomLossTrainer":
        return _custom_loss_trainer_class()
    raise AttributeError(name)


def compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
    """Accuracy, precision, recall and F1 computed from Trainer predictions."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def prepare_model_training(model: Any) -> Any:
    """Enable gradient flow through the LoRA adapter and set training mode."""
    model.enable_input_require_grads()
    model.config.use_cache = False
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    model.train()
    return model


def fine_tune_adapter(
    train_df: Any,
    eval_df: Any,
    model: Any | None = None,
    tokenizer: Any | None = None,
    output_dir: str = "results/adaptation",
    epochs: int = 5,
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    max_length: int = MAX_LENGTH_CLASSIFIER,
    seed: int = RANDOM_SEED,
    warmup_ratio: float = 0.0,
    metric_for_best_model: str = "f1",
) -> tuple[Any, Any, Any]:
    """Fine-tune the LoRA adapter for lightweight domain adaptation.

    Returns ``(trainer, model, tokenizer)``. The caller decides whether to
    persist the adapted weights with ``trainer.save_model``.
    """
    from transformers import DataCollatorWithPadding, TrainingArguments

    if model is None or tokenizer is None:
        model, tokenizer = load_roberta_model(trainable=True)

    datasets = prepare_training_datasets(
        train_df, eval_df, tokenizer, max_length=max_length
    )
    weights = compute_class_weights(datasets["train"]["labels"])

    arguments = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        warmup_ratio=warmup_ratio,
        seed=seed,
    )

    trainer_class = _custom_loss_trainer_class()
    trainer = trainer_class(
        model=prepare_model_training(model),
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=weights,
    )
    trainer.train()
    return trainer, model, tokenizer
