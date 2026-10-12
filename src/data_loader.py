from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset, DatasetDict, Value, load_dataset

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
        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict(dataset)
    except Exception:
        dataset = build_fallback_dataset()

    if "train" not in dataset:
        raise ValueError("The dataset must contain a 'train' split.")

    if "test" not in dataset:
        if len(dataset["train"]) < 3:
            raise ValueError("At least three training examples are required to create validation and test splits.")
        split = dataset["train"].train_test_split(test_size=0.2, seed=42)
        dataset["train"] = split["train"]
        dataset["test"] = split["test"]

    if "validation" not in dataset:
        if len(dataset["train"]) < 2:
            raise ValueError("At least two training examples are required to create a validation split.")
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    label_names = getattr(dataset["train"].features.get("label"), "names", None)
    for split_name, split in dataset.items():
        if "text" not in split.features:
            raise ValueError(f"The dataset split '{split_name}' must contain a 'text' column.")
        if "label" not in split.features:
            raise ValueError(f"The dataset split '{split_name}' must contain a 'label' column.")
        if len(split) == 0:
            raise ValueError(f"The dataset split '{split_name}' is empty.")

        split_label_names = getattr(split.features["label"], "names", None) or label_names
        if split_label_names:
            split = split.cast_column("label", Value("string"))
            split = split.map(
                lambda example, names=split_label_names: {
                    "label": (
                        names[int(example["label"])]
                        if example["label"].isdigit() and int(example["label"]) < len(names)
                        else example["label"]
                    )
                }
            )
        else:
            split = split.cast_column("label", Value("string"))
        dataset[split_name] = split

    return dataset
