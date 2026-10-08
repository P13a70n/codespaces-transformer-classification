from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loader import MODEL_DIR, load_or_generate_dataset


def evaluate_model(model_dir: str | Path | None = None, dataset_name: str = "test") -> Dict[str, Any]:
    model_path = Path(model_dir) if model_dir else MODEL_DIR
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    dataset = load_or_generate_dataset()
    test_df = dataset[dataset_name].to_pandas()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    label2id = {str(value): int(key) for key, value in model.config.id2label.items()}
    predictions = []
    true_labels = []
    probabilities = []

    for _, row in test_df.iterrows():
        encoded = tokenizer(row["text"], truncation=True, padding=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoded)
        logits = outputs.logits
        probs = logits.softmax(dim=-1).detach().numpy()[0]
        predicted_id = int(np.argmax(probs))
        predictions.append(predicted_id)
        true_labels.append(label2id.get(str(row["label"]), int(row["label"])))
        probabilities.append(probs)

    id2label = {int(key): value for key, value in model.config.id2label.items()}
    label_names = [id2label[i] for i in sorted(id2label)]

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average="macro")
    weighted_f1 = f1_score(true_labels, predictions, average="weighted")
    macro_precision = precision_score(true_labels, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(true_labels, predictions, average="macro", zero_division=0)

    report = classification_report(
        true_labels,
        predictions,
        target_names=[id2label[i] for i in sorted(id2label)],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    confusion = confusion_matrix(true_labels, predictions, labels=list(sorted(id2label)))

    confusion_df = pd.DataFrame(confusion, index=label_names, columns=label_names)

    top_confusions = []
    for i, true_label in enumerate(sorted(id2label)):
        for j, pred_label in enumerate(sorted(id2label)):
            count = confusion[i, j]
            if i != j and count > 0:
                top_confusions.append({
                    "true_intent": id2label[true_label],
                    "predicted_intent": id2label[pred_label],
                    "count": int(count),
                })

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "classification_report": report,
        "report_df": report_df,
        "confusion_matrix": confusion,
        "confusion_df": confusion_df,
        "top_confusions": sorted(top_confusions, key=lambda item: item["count"], reverse=True)[:10],
        "prediction_ids": predictions,
        "true_labels": true_labels,
        "label_names": label_names,
    }
