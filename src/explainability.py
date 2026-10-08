from __future__ import annotations

import re
from typing import Dict, List

HIGH_IMPORTANCE_TOKENS = {
    "withdraw": 1.4,
    "atm": 1.4,
    "cash": 1.4,
    "transfer": 1.5,
    "refund": 1.6,
    "balance": 1.3,
    "card": 1.5,
    "pin": 1.4,
    "payment": 1.4,
    "debit": 1.3,
    "cancel": 1.3,
    "support": 1.3,
    "verify": 1.2,
    "top": 1.2,
    "declined": 1.4,
    "stolen": 1.5,
    "lost": 1.5,
    "failed": 1.4,
    "pending": 1.3,
}


def explain_prediction(text: str, predicted_intent: str | None = None) -> List[Dict[str, float | str | bool]]:
    normalized = re.findall(r"[A-Za-z]+|\d+|[€₹$]+", text.lower())
    tokens = []
    for token in normalized:
        token_value = token.strip()
        if not token_value:
            continue
        weight = HIGH_IMPORTANCE_TOKENS.get(token_value, 0.5)
        if predicted_intent and token_value in predicted_intent.lower():
            weight += 0.3
        tokens.append({
            "token": token_value,
            "importance": round(weight, 2),
            "is_highlight": weight >= 1.2,
        })
    return tokens[:20]
