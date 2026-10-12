from __future__ import annotations

from typing import Any, Dict, Tuple

from datasets import DatasetDict, Value
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.data_loader import ARTIFACTS_DIR, MODEL_DIR, load_or_generate_dataset


def _prepare_dataset_for_training(dataset: DatasetDict, seq_model_name: str = "distilbert-base-uncased") -> Tuple[DatasetDict, Dict[str, int], Dict[int, str]]:
    label_names = sorted({
        str(label)
        for split in dataset.values()
        for label in split["label"]
    })
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
    model_input_columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in dataset["train"].column_names:
        model_input_columns.append("token_type_ids")
    dataset.set_format(type="torch", columns=[*model_input_columns, "label"])
    return dataset, label2id, id2label


def train_model(model_name: str = "distilbert-base-uncased", epochs: int = 3, sample_size: int = 1000) -> Dict[str, Any]:
    dataset = load_or_generate_dataset()
    if sample_size and len(dataset["train"]) > sample_size:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(sample_size))

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
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(MODEL_DIR))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(MODEL_DIR))

    return {"evaluation_metrics": metrics}
