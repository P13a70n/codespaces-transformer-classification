# Banking Message Classifier

A small Streamlit app that fine-tunes a DistilBERT transformer to classify banking support messages by intent.

## Run the app

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Use it

1. The app loads BANKING77 from Hugging Face. If it is unavailable, it uses a small built-in example dataset.
2. Select **Train model** to fine-tune DistilBERT. The model and tokenizer are saved under `artifacts/banking_intent_model`.
3. Enter a banking support message and select **Predict intent** to see the predicted intent and confidence.

The first run needs an internet connection to download the dataset and pretrained DistilBERT model. Training can take a few minutes, depending on the machine.
