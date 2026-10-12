import pandas as pd
import streamlit as st

from src.data_loader import MODEL_DIR, load_or_generate_dataset
from src.predictor import predict_intent
from src.trainer import train_model

st.set_page_config(page_title="Banking Message Classifier", page_icon="🏦")
st.title("Banking Message Classifier")
st.write("Train a transformer on banking support messages, then enter a message to predict its intent.")


@st.cache_data
def load_dataset():
    return load_or_generate_dataset()


try:
    dataset = load_dataset()
except Exception as exc:
    st.error(f"Could not load the banking dataset: {exc}")
    st.stop()

train_data = dataset["train"]
labels = train_data.features["label"]
intent_count = len(labels.names) if hasattr(labels, "names") else len(set(train_data["label"]))

st.subheader("1. Dataset")
st.write(
    f"Loaded **{len(train_data):,} training messages** across **{intent_count} intents**."
)
st.dataframe(
    train_data.select(range(min(10, len(train_data)))).to_pandas()[["text", "label"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("2. Train the transformer")
st.caption(
    "The first training run downloads DistilBERT and may take a few minutes. "
    "The trained model is saved locally for later predictions."
)
if st.button("Train model", type="primary"):
    try:
        with st.spinner("Fine-tuning DistilBERT on the banking messages..."):
            result = train_model(model_name="distilbert-base-uncased", epochs=1, sample_size=1000)
        accuracy = result["evaluation_metrics"].get("eval_accuracy")
        if accuracy is not None:
            st.success(f"Training complete. Validation accuracy: {accuracy:.1%}")
        else:
            st.success("Training complete. The model is ready to use.")
    except Exception as exc:
        st.error(f"Training failed: {type(exc).__name__}: {exc}")
        with st.expander("Error details"):
            st.exception(exc)

st.subheader("3. Predict a message intent")
message = st.text_area(
    "Enter a banking support message",
    placeholder="For example: My card was declined when I tried to pay.",
    height=120,
)

if st.button("Predict intent"):
    if not message.strip():
        st.warning("Enter a message before asking for a prediction.")
    elif not MODEL_DIR.exists():
        st.info("Train the model first, then submit your message again.")
    else:
        try:
            with st.spinner("Classifying your message..."):
                predictions = predict_intent(message.strip(), top_k=3)
            best = predictions[0]
            st.success(f"Predicted intent: **{best['intent']}**")
            st.caption(f"Model confidence: {best['confidence_pct']:.1f}%")
            if len(predictions) > 1:
                st.write("Other likely intents")
                st.dataframe(
                    pd.DataFrame(predictions[1:])[["intent", "confidence_pct"]].rename(
                        columns={"confidence_pct": "confidence (%)"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            with st.expander("Error details"):
                st.exception(exc)
