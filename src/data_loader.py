from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

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
    return DatasetDict(train=train_test["train"], validation=valid_test["train"], test=valid_test["test"])


def load_or_generate_dataset() -> DatasetDict:
    try:
        dataset = load_dataset("banking77")
        if isinstance(dataset, DatasetDict):
            return dataset
        return DatasetDict(dataset)
    except Exception:
        return build_fallback_dataset()


def get_intent_reference(dataset: DatasetDict | None = None) -> pd.DataFrame:
    if dataset is not None:
        label_names = getattr(dataset["train"].features["label"], "names", None)
        if label_names:
            records = [{"intent": label, "description": "Banking intent label", "examples": "Example query"} for label in label_names]
            return pd.DataFrame(records)

    records = []
    for intent, templates in INTENT_PATTERNS.items():
        records.append({
            "intent": intent,
            "description": "Banking customer support intent",
            "examples": templates[0],
        })
    return pd.DataFrame(records)


def get_dataset_summary(dataset: DatasetDict) -> Dict[str, Any]:
    train_df = dataset["train"].to_pandas()
    valid_df = dataset["validation"].to_pandas() if "validation" in dataset else dataset["test"].to_pandas()
    test_df = dataset["test"].to_pandas() if "test" in dataset else valid_df

    combined = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    combined["length"] = combined["text"].str.len()

    if all(isinstance(item, str) for item in combined["label"].dropna().tolist()):
        label_counts = combined["label"].value_counts().reset_index()
        label_counts.columns = ["intent", "count"]
    else:
        label_counts = combined["label"].astype(str).value_counts().reset_index()
        label_counts.columns = ["intent", "count"]

    summary = {
        "total_intents": int(combined["label"].nunique()),
        "dataset_size": int(len(combined)),
        "train_examples": int(len(train_df)),
        "validation_examples": int(len(valid_df)),
        "test_examples": int(len(test_df)),
        "avg_query_length": float(combined["length"].mean()),
        "max_query_length": int(combined["length"].max()),
        "min_query_length": int(combined["length"].min()),
        "common_intents": label_counts.head(10),
        "rare_intents": label_counts.tail(10).sort_values("count", ascending=True),
        "sample_preview": combined.head(10).copy(),
        "label_counts": label_counts,
        "query_lengths": combined["length"].tolist(),
    }
    return summary
