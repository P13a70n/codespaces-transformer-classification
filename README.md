# Banking Customer Intent Classification with Transformers

This project is a beginner-friendly learning lab for understanding how a Transformer model such as BERT or DistilBERT can classify banking customer messages into intents.

The project follows the full learning flow:

BANKING77 -> Tokenization -> BERT/DistilBERT -> Fine-tuning -> Intent Prediction

## Why this is useful

Banks receive thousands of customer messages every day. Instead of manually routing each message, a Transformer model can learn the meaning of each message and predict the intent.

Example:

> “I tried to withdraw ₹10,000, but the ATM didn't give me the cash.”

The model predicts:

- Intent: Cash withdrawal issue

This project is designed as an educational demonstration for:

- learning the Transformer workflow
- understanding tokenization and classification
- experimenting with BERT and DistilBERT
- building a simple but meaningful Streamlit UI

## Project structure

- `app.py` – main Streamlit application
- `src/transformer_lab.py` – dataset loading, training logic, and prediction helpers
- `artifacts/` – model checkpoints saved after training

## Data approach

The app tries to use the real BANKING77 dataset from Hugging Face when available. If the environment is offline or the dataset cannot be pulled, it falls back to a generated demo dataset that preserves the same learning flow and intent-classification pattern.

This keeps the lab reproducible and classroom-friendly.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## What the app teaches

1. How text is tokenized before entering a Transformer.
2. How intent labels are mapped to customer messages.
3. How a pre-trained BERT model is fine-tuned for classification.
4. How to inspect predictions, confidence, and model behavior.
5. How a Streamlit UI can explain the system to non-technical users.

## Example business use case

A bank can automatically route:

- card problems
- transfer issues
- cash withdrawal complaints
- identity verification questions
- top-up and refund inquiries

into the right support queues without manual triage.
