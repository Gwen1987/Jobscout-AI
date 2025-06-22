import gradio as gr
from tensorflow.keras.models import load_model
import tensorflow as tf
import csv
from datetime import datetime
import os

# Load the raw-text model (with TextVectorization baked in)
pipeline = load_model("models/archived/jobscout_pipeline_v1.keras")
THRESHOLD = 0.83
LOG_FILE = "jobscout_logs_old_model.csv"

def predict_fraud(job_text):
    try:
        # Input as string wrapped in tensor
        input_tensor = tf.constant([job_text])
        prob = pipeline.predict(input_tensor)[0][0]

        # Classification
        if prob >= THRESHOLD:
            label = "🚩 High Fraud Risk"
        elif prob >= 0.30:
            label = "🟡 Caution Advised"
        else:
            label = "✅ Low Risk — Appears Legitimate"

        result = (
            f"{label}\n\n"
            f"Estimated Fraud Probability: {prob:.2%}\n\n"
            f"(Model flags job ads as fraud when probability ≥ {int(THRESHOLD * 100)}%)"
        )

        # Logging with headers if file doesn't exist
        log_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            if not log_exists:
                writer.writerow(["timestamp", "probability", "label", "preview"])
            preview = job_text.replace("\n", " ").replace("\r", " ")[:120]
            writer.writerow([datetime.now().isoformat(), round(prob, 4), label, preview])

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio UI
demo = gr.Interface(
    fn=predict_fraud,
    inputs=gr.Textbox(lines=6, label="Paste a job posting"),
    outputs=gr.Text(label="Fraud Risk Assessment"),
    title="JobScout AI — (Original Model)",
    description=(
        "This version uses the original model trained on raw text only.\n"
        "No structured features or external vectorizer are required.\n\n"
        "🚩 Posts scoring ≥ 83% are flagged as suspicious.\n"
        "📝 Inputs are logged to compare with new model performance."
    )
)

# Launch
demo.launch()
