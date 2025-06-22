import gradio as gr
from tensorflow.keras.models import load_model
from datetime import datetime
import pickle
import csv
import os
import numpy as np

# === CONFIGURATION ===
MODEL_PATH = "models/jobscout_pipeline_v1.keras"
VECTORIZER_PATH = "models/jobscout_vectorizer.pkl"
LOG_PATH = "jobscout_logs_final.csv"
THRESHOLD = 0.83

# === LOAD ARTIFACTS ===
pipeline = load_model(MODEL_PATH)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# === LOGGING FUNCTION ===
def log_prediction(prob, label, title, preview, log_path=LOG_PATH):
    log_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not log_exists or os.stat(log_path).st_size == 0:
            writer.writerow(["timestamp", "probability", "label", "title_preview", "text_snippet"])
        writer.writerow([
            datetime.now().isoformat(),
            round(prob, 4),
            label,
            title,
            preview
        ])

# === PREDICTION FUNCTION ===
def predict_fraud(job_text):
    try:
        # Vectorize input
        vec = vectorizer.transform([job_text]).toarray()
        prob = pipeline.predict(vec)[0][0]

        # Risk classification
        if prob >= THRESHOLD:
            label = "🚩 High Fraud Risk"
            disclaimer = (
                "This job posting is flagged as potentially fraudulent based on its content. "
                "We strongly recommend verifying its legitimacy before proceeding. "
                "Contact the company directly through their official website or phone number."
            )
        elif prob >= 0.30:
            label = "🟡 Caution Advised"
            disclaimer = (
                "This posting shows some traits commonly found in deceptive job ads. "
                "Please confirm key details with the company before applying."
            )
        else:
            label = "✅ Low Risk — Appears Legitimate"
            disclaimer = (
                "This job posting appears consistent with legitimate listings based on known patterns."
            )

        # Result string
        result = (
            f"{label}\n\n"
            f"Estimated Fraud Probability: {prob:.2%}\n\n"
            f"{disclaimer}\n\n"
            f"(Model flags job ads as fraud when probability ≥ {int(THRESHOLD * 100)}%)"
        )

        # Clean logging
        clean_text = job_text.replace("\n", " ").replace("\r", " ").strip()
        preview = clean_text[:120]
        title_guess = clean_text.split()[0:10]
        title = " ".join(title_guess) if title_guess else "Untitled"

        log_prediction(prob, label, title, preview)

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"

# === GRADIO APP ===
demo = gr.Interface(
    fn=predict_fraud,
    inputs=gr.Textbox(lines=6, label="Paste a job posting"),
    outputs=gr.Text(label="Fraud Risk Assessment"),
    title="JobScout AI",
    description=(
        "Paste a job ad below to assess its fraud risk.\n\n"
        "🔍 This version uses only the job text.\n"
        "🚩 Posts scoring ≥ 83% are flagged as potentially fraudulent.\n"
        "📝 All inputs are logged for internal analysis."
    )
)

# === LAUNCH ===
demo.launch()
