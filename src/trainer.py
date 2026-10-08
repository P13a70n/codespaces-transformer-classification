from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, Value
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from src.data_loader import ARTIFACTS_DIR, MODEL_DIR, load_or_generate_dataset


def _prepare_dataset_for_training(dataset: DatasetDict, seq_model_name: str = "distilbert-base-uncased") -> Tuple[DatasetDict, Dict[str, int], Dict[int, str]]:
    label_names = sorted({str(item) for item in dataset["train"]["label"]})
    id2label = {index: label for index, label in enumerate(label_names)}
    label2id = {label: index for index, label in id2label.items()}

    def _cast_labels(example):
        example["label"] = int(label2id[str(example["label"])])
        return example

    dataset = dataset.map(_cast_labels)
    dataset = dataset.cast_column("label", Value("int64"))
    tokenizer = AutoTokenizer.from_pretrained(seq_model_name)

    def _tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=64)

    dataset = dataset.map(_tokenize_batch, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
    return dataset, label2id, id2label


class TrainingHistoryLogger(TrainerCallback):
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            record = {"step": state.global_step if state else 0, **logs}
            self.logs.append(record)


def train_model(model_name: str = "distilbert-base-uncased", epochs: int = 3, sample_size: int = 1000) -> Dict[str, Any]:
    dataset = load_or_generate_dataset()
    train_df = dataset["train"].to_pandas()
    if sample_size and len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)
        dataset["train"] = Dataset.from_pandas(train_df)

    prepared, label2id, id2label = _prepare_dataset_for_training(dataset, seq_model_name=model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared["train"],
        eval_dataset=prepared["validation"],
    )
    history_logger = TrainingHistoryLogger()
    trainer.add_callback(history_logger)

    train_result = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(MODEL_DIR))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(MODEL_DIR))

    with open(MODEL_DIR / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump({
            "model_name": model_name,
            "epochs": epochs,
            "sample_size": sample_size,
            "training_metrics": train_result.metrics,
            "evaluation_metrics": metrics,
            "id2label": id2label,
            "label2id": label2id,
        }, file, indent=2, ensure_ascii=False)

    return {
        "model_name": model_name,
        "trained_model_dir": str(MODEL_DIR),
        "epochs": epochs,
        "sample_size": sample_size,
        "train_metrics": train_result.metrics,
        "evaluation_metrics": metrics,
        "history": history_logger.logs,
        "label2id": label2id,
        "id2label": id2label,
    }


def compare_models() -> List[Dict[str, Any]]:
    comparison = []
    for model_name in ["distilbert-base-uncased", "bert-base-uncased"]:
        try:
            result = train_model(model_name=model_name, epochs=1, sample_size=400)
            comparison.append(result)
        except Exception as exc:  # pragma: no cover
            comparison.append({"model_name": model_name, "error": str(exc)})
    return comparison
