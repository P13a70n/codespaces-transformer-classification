from __future__ import annotations

from typing import Dict, List

from transformers import AutoTokenizer


def tokenize_sentence(sentence: str, model_name: str = "distilbert-base-uncased") -> Dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(sentence)
    encoded = tokenizer(sentence, return_attention_mask=True, return_token_type_ids=True, return_tensors="np")

    input_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded["attention_mask"][0].tolist()
    token_type_ids = encoded.get("token_type_ids", [[0] * len(input_ids)])[0].tolist()

    return {
        "raw_text": sentence,
        "tokens": tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def explain_tokenization(sentence: str, model_name: str = "distilbert-base-uncased") -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(sentence)
    explanation = [
        "Step 1: The raw customer message is kept as readable text.",
        "Step 2: The sentence is split into smaller token units such as words and subwords.",
        "Step 3: The tokenizer assigns each token a numerical ID that the model reads.",
        "Step 4: The attention mask tells the model which tokens are real and which are padding.",
        "Step 5: Special tokens like [CLS] and [SEP] mark sentence start and sentence end.",
        f"For the example sentence, the tokenizer produced {len(tokens)} tokens: {tokens}",
    ]
    return explanation
