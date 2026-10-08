from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Value, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "banking_intent_model"

INTENT_PATTERNS: Dict[str, List[str]] = {
    "cash_withdrawal_issue": [
        "I tried to withdraw ₹{amount} from the ATM, but the machine did not dispense cash.",
        "The ATM failed during my withdrawal and I still need the money.",
        "I was charged for a withdrawal that did not complete.",
    ],
    "card_not_working": [
        "My debit card keeps getting declined even though I have sufficient funds.",
        "The card reader rejects my card every time I try to pay.",
        "My card is not working at all and I cannot complete a purchase.",
    ],
    "transfer_issue": [
        "I sent a transfer but the money is not in the recipient account yet.",
        "My transfer failed again and I need to know why.",
        "I was trying to transfer funds and the app shows an error.",
    ],
    "refund_request": [
        "I want a refund for a duplicate charge on my account.",
        "Can you process a refund for the amount that was billed incorrectly?",
        "My purchase was charged twice and I need my money back.",
    ],
    "balance_inquiry": [
        "What is my current balance after the recent transactions?",
        "Can you tell me how much money is left in my account?",
        "I need to check my balance before making a transfer.",
    ],
    "pin_issue": [
        "I forgot my PIN and need help resetting it.",
        "The system is blocking my PIN and I cannot use my card.",
        "I need to change my card PIN because I no longer remember it.",
    ],
    "account_verification": [
        "I am unable to verify my identity in the app.",
        "The verification step keeps failing for my account login.",
        "I need help with identity verification for my banking profile.",
    ],
    "top_up_failed": [
        "My top-up failed even though the card details were correct.",
        "I tried to add money to my account and the transaction failed.",
        "The mobile top-up is not going through.",
    ],
    "card_lost_or_stolen": [
        "My card was stolen and I need to block it immediately.",
        "I lost my credit card and need to freeze it.",
        "Someone stole my card and I need support to secure my account.",
    ],
    "pending_payment": [
        "A payment is showing as pending but it should already be complete.",
        "My payment is stuck in pending status, what should I do?",
        "I made a payment and it is still pending in the app.",
    ],
    "direct_debit_issue": [
        "A direct debit was charged but I did not authorize it.",
        "Why is there a direct debit on my account from an unknown company?",
        "I need help with a direct debit dispute.",
    ],
    "currency_exchange": [
        "I want to know the exchange rate on my international transfer.",
        "The currency conversion fee seems higher than expected.",
        "Can you explain the exchange charges on my transaction?",
    ],
    "card_arrival": [
        "When will my new physical card arrive?",
        "I ordered a replacement card and need the delivery timeline.",
        "Can you check the status of my card delivery?",
    ],
    "virtual_card": [
        "My virtual card is not working for online payments.",
        "The virtual card keeps being declined in checkout.",
        "I need help with my digital card not accepting payments.",
    ],
    "support_contact": [
        "I need to contact customer support about my bank account.",
        "Who can I speak to about account issues?",
        "I need help and want to reach a support advisor.",
    ],
}


def build_fallback_dataset() -> DatasetDict:
    rows: List[Dict[str, Any]] = []
    amounts = [1000, 5000, 10000, 25000, 50000]

    for intent, templates in INTENT_PATTERNS.items():
        for index, template in enumerate(templates):
            text = template.format(amount=amounts[(index + len(intent)) % len(amounts)])
            rows.append({"text": text, "label": intent})

    df = pd.DataFrame(rows)
    dataset = Dataset.from_pandas(df)
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    valid_test = train_test["test"].train_test_split(test_size=0.5, seed=42)
    return DatasetDict(
        train=train_test["train"],
        validation=valid_test["train"],
        test=valid_test["test"],
    )


def load_or_generate_dataset() -> DatasetDict:
    try:
        dataset = load_dataset("banking77")
        if isinstance(dataset, DatasetDict):
            return dataset
        return DatasetDict(dataset)
    except Exception:
        return build_fallback_dataset()


def get_dataset_overview(dataset: DatasetDict) -> Dict[str, Any]:
    train_df = dataset["train"].to_pandas()
    valid_df = dataset["validation"].to_pandas() if "validation" in dataset else dataset["test"].to_pandas()
    test_df = dataset["test"].to_pandas() if "test" in dataset else valid_df

    label_names = None
    if hasattr(dataset["train"].features["label"], "names"):
        label_names = dataset["train"].features["label"].names

    def normalize_label(value: Any) -> str:
        if label_names is not None and isinstance(value, (int, np.integer)):
            return str(label_names[int(value)])
        return str(value)

    for frame in (train_df, valid_df, test_df):
        frame["intent"] = frame["label"].apply(normalize_label)

    intent_counts = (
        train_df.groupby("intent").size().reset_index(name="count").sort_values("count", ascending=False)
    )
    sample_rows = pd.concat([train_df.head(5), valid_df.head(2), test_df.head(2)], ignore_index=True)

    return {
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
        "intent_counts": intent_counts,
        "sample_rows": sample_rows[["text", "intent"]],
    }


def get_intent_reference(dataset: DatasetDict | None = None) -> pd.DataFrame:
    if dataset is not None:
        label_names = getattr(dataset["train"].features["label"], "names", None)
        if label_names:
            data = [{"intent": intent, "description": "Banking customer support intent"} for intent in label_names]
            return pd.DataFrame(data)

    data = [
        {"intent": intent, "description": "Banking customer support intent"}
        for intent in sorted(INTENT_PATTERNS.keys())
    ]
    return pd.DataFrame(data)


def _compute_accuracy(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float(np.mean(preds == labels))}


def train_demo_model(model_name: str = "distilbert-base-uncased", epochs: int = 2, sample_size: int = 200) -> Dict[str, Any]:
    dataset = load_or_generate_dataset()

    train_split = dataset["train"]
    if sample_size < len(train_split):
        train_split = train_split.select(range(sample_size))

    valid_split = dataset["validation"] if "validation" in dataset else dataset["test"]
    raw_labels = sorted(train_split.unique("label"))
    label_names = getattr(train_split.features["label"], "names", None)

    if label_names is not None and all(isinstance(label, (int, np.integer)) for label in raw_labels):
        label2id = {int(label): int(label) for label in raw_labels}
        id2label = {int(label): str(label_names[int(label)]) for label in raw_labels}
        train_split = train_split.map(lambda x: {"label": int(x["label"])})
        valid_split = valid_split.map(lambda x: {"label": int(x["label"])})
    else:
        label2id = {str(label): idx for idx, label in enumerate(raw_labels)}
        id2label = {idx: str(label) for idx, label in enumerate(raw_labels)}
        train_split = train_split.map(lambda x: {"label": label2id[str(x["label"])]})
        valid_split = valid_split.map(lambda x: {"label": label2id.get(str(x["label"]), 0)})

    train_split = train_split.cast_column("label", Value("int64"))
    valid_split = valid_split.cast_column("label", Value("int64"))
    labels = list(range(len(raw_labels))) if label_names is not None and all(isinstance(label, (int, np.integer)) for label in raw_labels) else list(raw_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_tokenized = train_split.map(tokenize, batched=True)
    valid_tokenized = valid_split.map(tokenize, batched=True)

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(raw_labels),
        id2label=id2label,
        label2id={str(k): v for k, v in label2id.items()},
    )

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        load_best_model_at_end=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        compute_metrics=_compute_accuracy,
    )

    trainer.train()
    evaluation = trainer.evaluate()
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    return {
        "model": model_name,
        "epochs": epochs,
        "sample_size": len(train_split),
        "num_labels": len(labels),
        "accuracy": round(float(evaluation.get("eval_accuracy", 0.0)), 4),
        "output_dir": str(MODEL_DIR),
    }


def predict_intent(text: str, model_name: str = "distilbert-base-uncased") -> List[Dict[str, Any]]:
    if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").exists():
        train_demo_model(model_name=model_name, epochs=2, sample_size=200)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits[0], dim=-1)
    topk = torch.topk(probabilities, k=min(3, len(probabilities)))

    results: List[Dict[str, Any]] = []
    for index, score in zip(topk.indices.tolist(), topk.values.tolist()):
        label = model.config.id2label.get(int(index), str(index))
        results.append({"label": label, "score": round(float(score), 4)})

    return results
