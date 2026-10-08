import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.data_loader import MODEL_DIR, get_dataset_summary, get_intent_reference, load_or_generate_dataset
from src.evaluator import evaluate_model
from src.explainability import explain_prediction
from src.predictor import predict_intent
from src.tokenizer_lab import explain_tokenization, tokenize_sentence
from src.trainer import compare_models, train_model
from src.visualizations import make_confusion_matrix_chart, make_intent_distribution_chart, make_prediction_chart, make_query_length_chart

st.set_page_config(page_title="Banking Transformer Lab", page_icon="🏦", layout="wide")

st.title("🏦 Banking Transformer NLP Lab")
st.caption("Professional Transformer project for banking intent classification using the BANKING77-inspired learning workflow.")


@st.cache_data
def load_dataset_cached():
    return load_or_generate_dataset()


@st.cache_data
def build_dataset_summary_cached(dataset):
    return get_dataset_summary(dataset)


@st.cache_data
def load_reference_table():
    return get_intent_reference(load_dataset_cached())


def ensure_model_ready():
    if not MODEL_DIR.exists():
        with st.spinner("No saved model found. Training the default banking intent model..."):
            result = train_model(model_name="distilbert-base-uncased", epochs=2, sample_size=600)
        return result
    return {"trained_model_dir": str(MODEL_DIR)}


def render_metric_cards(metrics: dict):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    with col2:
        st.metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")
    with col3:
        st.metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.3f}")
    with col4:
        st.metric("Macro Recall", f"{metrics.get('macro_recall', 0):.3f}")


def build_pdf_report(summary_payload: dict, eval_payload: dict | None = None):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Banking Intent Classification Report")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(72, 760, "Banking Intent Classification Report")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(72, 735, f"Dataset size: {summary_payload.get('dataset_size', 0)}")
    pdf.drawString(72, 720, f"Total intents: {summary_payload.get('total_intents', 0)}")
    pdf.drawString(72, 705, f"Average query length: {summary_payload.get('avg_query_length', 0):.1f} chars")

    if eval_payload:
        pdf.drawString(72, 685, f"Accuracy: {eval_payload.get('accuracy', 0):.2%}")
        pdf.drawString(72, 670, f"Macro F1: {eval_payload.get('macro_f1', 0):.3f}")
        pdf.drawString(72, 655, f"Weighted F1: {eval_payload.get('weighted_f1', 0):.3f}")

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(72, 620, "Top 5 predicted intents")
    pdf.setFont("Helvetica", 10)
    for idx, item in enumerate((summary_payload.get("label_counts", [])[:5]).to_dict("records") if hasattr(summary_payload.get("label_counts", []), "to_dict") else [], start=1):
        pdf.drawString(72, 600 - (idx - 1) * 18, f"{idx}. {item.get('intent', '')}: {item.get('count', 0)} samples")

    pdf.save()
    buffer.seek(0)
    return buffer


sidebar = st.sidebar
sidebar.title("Navigation")
nav = sidebar.radio(
    "Choose a project section",
    ["Overview", "Dataset Lab", "Tokenizer Lab", "Architecture", "Training Lab", "Evaluation Lab", "Prediction Demo", "Model Comparison", "Simulation", "Intent Reference", "Export Center"],
)

if nav == "Overview":
    st.subheader("Business context")
    st.markdown(
        """
        A digital bank receives support requests from customers in natural language. The goal is to classify each message into the correct banking intent so the bank can route the request to the correct team, resolve issues faster, and improve customer experience.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📚 Learning objective: understand how a transformer reads the full sentence and learns intent patterns from text.")
    with col2:
        st.info("🧠 Core idea: attention helps the model connect words like 'withdraw', 'ATM', and 'cash' into one complete meaning.")
    with col3:
        st.info("🏦 Real-world impact: support teams can prioritize card, transfer, refund, balance, and fraud-related issues efficiently.")

    st.markdown("---")
    st.subheader("End-to-end transformer workflow")
    st.code("Customer message → Tokenizer → Transformer encoder → Classification head → Intent label")

    demo_example = "I tried to withdraw ₹10,000 from the ATM but the cash was not dispensed."
    st.text_area("Example banking message", demo_example, height=100, disabled=True)
    st.success("Predicted intent: cash_withdrawal_issue")

    st.subheader("Why transformers work well")
    st.markdown(
        """
        - They process the whole sentence rather than isolated words.
        - Attention assigns different weights to the most relevant words.
        - They learn contextual relationships like payment + card + declined.
        - They generalize to new customer phrasing that has not been seen before.
        """
    )

elif nav == "Dataset Lab":
    dataset = load_dataset_cached()
    summary = build_dataset_summary_cached(dataset)

    st.subheader("Dataset overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset size", summary["dataset_size"])
    with col2:
        st.metric("Train examples", summary["train_examples"])
    with col3:
        st.metric("Validation examples", summary["validation_examples"])
    with col4:
        st.metric("Test examples", summary["test_examples"])

    st.plotly_chart(make_intent_distribution_chart(summary["label_counts"]), use_container_width=True)
    st.plotly_chart(make_query_length_chart(summary["query_lengths"]), use_container_width=True)

    st.subheader("Preview of the data")
    st.dataframe(summary["sample_preview"], use_container_width=True)

    st.caption("This project supports the real BANKING77 dataset when available and falls back to a generated, classroom-friendly dataset when the public dataset is unavailable.")

elif nav == "Tokenizer Lab":
    st.subheader("Tokenizer explorer")
    text_input = st.text_area("Enter a customer message", value="I need help with a transfer that failed after I tried to send money to my cousin.", height=120)
    model_choice = st.selectbox("Tokenizer model", ["distilbert-base-uncased", "bert-base-uncased"])

    if text_input:
        token_info = tokenize_sentence(text_input, model_choice)
        token_df = pd.DataFrame(
            {
                "position": list(range(len(token_info["tokens"]))),
                "token": token_info["tokens"],
                "input_id": token_info["input_ids"],
                "attention_mask": token_info["attention_mask"],
            }
        )

        left, right = st.columns(2)
        with left:
            st.write("Token list")
            st.dataframe(token_df, use_container_width=True)
        with right:
            st.write("Tokenization explanation")
            for step in explain_tokenization(text_input, model_choice):
                st.write(f"• {step}")

        st.json({
            "raw_text": token_info["raw_text"],
            "token_count": len(token_info["tokens"]),
            "input_ids": token_info["input_ids"],
            "attention_mask": token_info["attention_mask"],
        })

elif nav == "Architecture":
    st.subheader("Model architecture overview")
    st.markdown(
        """
        1. Input text enters the tokenizer.
        2. Tokens become embeddings.
        3. The transformer encoder learns attention-based context.
        4. A classification head outputs one intent score per class.
        5. The highest-scoring class becomes the predicted banking intent.
        """
    )

    html = """
    <style>
    .box { padding: 14px; border-radius: 12px; text-align: center; margin: 6px; font-weight: 600; color: #0f172a; background: linear-gradient(135deg, #e0f2fe, #dbeafe); border: 1px solid #93c5fd; display: inline-block; }
    .arrow { font-size: 28px; color: #3b82f6; }
    </style>
    <div style='display:flex; align-items:center; flex-wrap:wrap; justify-content:center;'>
        <div class='box' title='Customer support messages are passed as raw text'>Customer message</div>
        <div class='arrow'>→</div>
        <div class='box' title='Words are split into tokens and converted to IDs'>Tokenizer</div>
        <div class='arrow'>→</div>
        <div class='box' title='Each token receives a dense vector representation'>Embeddings</div>
        <div class='arrow'>→</div>
        <div class='box' title='Transformer blocks use self-attention to capture context'>Encoder</div>
        <div class='arrow'>→</div>
        <div class='box' title='A prediction head maps encoder output to one of the support intents'>Classifier</div>
        <div class='arrow'>→</div>
        <div class='box' title='The highest probability becomes the routed intent'>Predicted intent</div>
    </div>
    """
    st.components.v1.html(html, height=220)

elif nav == "Training Lab":
    st.subheader("Training the banking intent classifier")
    model_choice = st.selectbox("Base model", ["distilbert-base-uncased", "bert-base-uncased"])
    epochs = st.slider("Training epochs", 1, 5, 2)
    sample_size = st.slider("Training samples", 100, 1500, 600, 50)

    if st.button("Train model"):
        with st.spinner("Training model in progress..."):
            result = train_model(model_name=model_choice, epochs=epochs, sample_size=sample_size)
        st.success("Training completed!")
        st.json({
            "model": result["model_name"],
            "saved_dir": result["trained_model_dir"],
            "epochs": result["epochs"],
            "sample_size": result["sample_size"],
            "eval_accuracy": result["evaluation_metrics"].get("eval_accuracy"),
        })

        history = pd.DataFrame(result.get("history", []))
        if not history.empty:
            st.line_chart(history[[col for col in history.columns if col.startswith("eval_") or col == "loss"]])

    st.markdown(
        """
        Training teaches the model to connect phrase patterns with banking intent labels. For example, the phrases 'ATM', 'cash', and 'withdraw' repeatedly signal a cash withdrawal issue, while 'declined', 'card', and 'payment' often indicate a card or payment problem.
        """
    )

elif nav == "Evaluation Lab":
    st.subheader("Model evaluation")
    model_ready = MODEL_DIR.exists()
    if not model_ready:
        st.warning("No trained model is available yet. Train a model first in the Training Lab.")
        ensure_model_ready()

    if MODEL_DIR.exists():
        eval_payload = evaluate_model(MODEL_DIR)
        render_metric_cards({
            "accuracy": eval_payload["accuracy"],
            "macro_f1": eval_payload["macro_f1"],
            "weighted_f1": eval_payload["weighted_f1"],
            "macro_recall": eval_payload["macro_recall"],
        })

        st.subheader("Classification report")
        st.dataframe(eval_payload["report_df"], use_container_width=True)

        st.subheader("Confusion matrix")
        st.plotly_chart(make_confusion_matrix_chart(eval_payload["confusion_df"]), use_container_width=True)

        st.subheader("Most frequent confusion pairs")
        if eval_payload["top_confusions"]:
            st.dataframe(pd.DataFrame(eval_payload["top_confusions"]), use_container_width=True)
        else:
            st.info("No notable confusion pairs were found in this small validation set.")

elif nav == "Prediction Demo":
    st.subheader("Live intent prediction")
    default_query = "My card was declined while paying for groceries in the supermarket and I need help."
    query = st.text_area("Type a customer support message", value=default_query, height=140)

    if st.button("Predict intent"):
        try:
            result = predict_intent(query, MODEL_DIR)
            st.success(f"Top predicted intent: {result[0]['intent']}")
            st.plotly_chart(make_prediction_chart(result), use_container_width=True)

            top_result = result[0]
            st.metric("Confidence", f"{top_result['confidence_pct']}%")

            st.subheader("Explainability view")
            explanation = explain_prediction(query, top_result["intent"])
            explanation_df = pd.DataFrame(explanation)
            if not explanation_df.empty:
                st.dataframe(explanation_df, use_container_width=True)
                st.bar_chart(explanation_df.set_index("token")["importance"])
        except FileNotFoundError:
            st.warning("Model not found. Train one from the Training Lab, then retry the prediction.")

elif nav == "Model Comparison":
    st.subheader("Model comparison")
    if st.button("Run comparison"):
        with st.spinner("Comparing DistilBERT and BERT on a lightweight training setup..."):
            comparison = compare_models()
        comparison_df = pd.DataFrame([
            {
                "model_name": entry.get("model_name"),
                "epochs": entry.get("epochs"),
                "sample_size": entry.get("sample_size"),
                "accuracy": entry.get("evaluation_metrics", {}).get("eval_accuracy"),
                "macro_f1": entry.get("evaluation_metrics", {}).get("eval_f1"),
                "loss": entry.get("evaluation_metrics", {}).get("eval_loss"),
                "error": entry.get("error"),
            }
            for entry in comparison
        ])
        st.dataframe(comparison_df, use_container_width=True)

elif nav == "Simulation":
    st.subheader("Support queue simulator")
    messages = st.text_area("Add one message per line", value="I need a refund for a duplicate charge.\nMy transfer failed but the money went out of my account.\nMy ATM did not dispense cash.\nI lost my card and need it blocked immediately.", height=180)

    if st.button("Run simulation"):
        lines = [line.strip() for line in messages.splitlines() if line.strip()]
        rows = []
        for text in lines:
            pred = predict_intent(text, MODEL_DIR)
            rows.append({"message": text, "predicted_intent": pred[0]["intent"], "confidence_pct": pred[0]["confidence_pct"]})

        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True)
        st.bar_chart(result_df.set_index("predicted_intent")["confidence_pct"])

elif nav == "Intent Reference":
    st.subheader("Banking intent reference")
    reference_df = load_reference_table()
    st.dataframe(reference_df, use_container_width=True)

elif nav == "Export Center":
    st.subheader("Export project artifacts")
    dataset = load_dataset_cached()
    summary = build_dataset_summary_cached(dataset)

    report_json = json.dumps({
        "dataset_summary": summary,
        "model_dir": str(MODEL_DIR),
    }, indent=2, ensure_ascii=False)
    st.download_button("Download JSON summary", report_json, file_name="banking_intent_summary.json", mime="application/json")

    csv_buffer = io.StringIO()
    pd.DataFrame(summary["label_counts"]).to_csv(csv_buffer, index=False)
    st.download_button("Download intent counts CSV", csv_buffer.getvalue(), file_name="intent_counts.csv", mime="text/csv")

    if MODEL_DIR.exists():
        try:
            evaluation = evaluate_model(MODEL_DIR)
            pdf_buffer = build_pdf_report(summary, evaluation)
            st.download_button("Download PDF report", pdf_buffer.getvalue(), file_name="banking_intent_report.pdf", mime="application/pdf")
        except Exception:
            st.info("Train a model first so a PDF report can be generated.")
    else:
        st.info("Train a model first so a PDF report can be generated.")

sidebar.markdown("---")
sidebar.write("Project highlights")
sidebar.write("• Real BANKING77-compatible workflow")
sidebar.write("• Transformer training and evaluation")
sidebar.write("• Explainability and intent simulation")
sidebar.write("• Educational project layout for portfolio/demo use")

ensure_model_ready()
