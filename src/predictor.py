from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loader import MODEL_DIR


def predict_intent(query: str, model_dir: str | Path | None = None, top_k: int = 5) -> List[Dict[str, Any]]:
    model_path = Path(model_dir) if model_dir else MODEL_DIR
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    encoded = tokenizer(query, truncation=True, padding=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits[0]

    probs = torch.softmax(logits, dim=-1)
    top_scores, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
    id2label = {int(key): value for key, value in model.config.id2label.items()}

    result = []
    for score, index in zip(top_scores.tolist(), top_indices.tolist()):
        result.append({
            "intent": id2label.get(int(index), str(index)),
            "confidence": float(score),
            "confidence_pct": round(float(score) * 100, 2),
        })
    return result
