import gradio as gr
from tensorflow.keras.models import load_model
from datetime import datetime
import numpy as np
import pickle
import csv
import os
from scipy.sparse import csr_matrix, hstack

# === CONFIGURATION ===
MODEL_PATH = "models/jobscout_tfidf_smoteenn.keras"
VECTORIZER_PATH = "models/jobscout_tfidf_vectorizer.pkl"
LOG_PATH = "logs/jobscout_logs_smoteenn.csv"
THRESHOLD = 0.85

# === RED FLAG BOOSTER ===
RED_FLAGS = [
    "wire transfer", "bitcoin", "crypto", "gift card", "telegram", "whatsapp only",
    "no interview", "scan id", "daily payout", "pre-employment fee", "send money",
    "zelle", "cashapp", "google hangouts", "training fee"
]

def red_flag_score(text: str) -> float:
    text = text.lower()
    hits = sum(flag in text for flag in RED_FLAGS)
    return min(hits / len(RED_FLAGS), 1.0)

# === LOAD MODEL AND VECTORIZER ===
model = load_model(MODEL_PATH)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# === LOGGING FUNCTION ===
def log_prediction(prob, label, title, preview, log_path=LOG_PATH):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline='', encoding="utf-8") as f:
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
        clean_text = job_text.replace("\n", " ").replace("\r", " ").strip()

        # Vectorize text
        vec = vectorizer.transform([clean_text])  # (1, 10000)
        booster_val = red_flag_score(clean_text)
        booster = csr_matrix([[booster_val]])     # (1, 1)

        # Combine features
        full_input = hstack([vec, booster]).toarray()  # (1, 10001)

        # Predict
        prob = model.predict(full_input)[0][0]

        # Classify
        if prob >= 0.83:
            label = "🚩 High Fraud Risk"
            disclaimer = (
                "This job posting is flagged as potentially fraudulent based on its content. "
                "Please verify before applying."
            )
        elif prob >= THRESHOLD:
            label = "🟡 Caution Advised"
            disclaimer = (
                "This posting shows some traits of deceptive listings. Confirm details with the employer."
            )
        else:
            label = "✅ Low Risk — Appears Legitimate"
            disclaimer = "This job ad appears consistent with legitimate listings."

        result = (
            f"{label}\n\n"
            f"Estimated Fraud Probability: {prob:.2%}\n\n"
            f"{disclaimer}\n\n"
            f"(Model flags job ads as fraud when probability ≥ {int(THRESHOLD * 100)}%)"
        )

        # Logging
        preview = clean_text[:120]
        title_guess = clean_text.split()[0:10]
        title = " ".join(title_guess) if title_guess else "Untitled"
        log_prediction(prob, label, title, preview)

        # Debug info (optional)
        print("🧾 Booster Score:", booster_val)
        print("🧾 Input Shape:", full_input.shape)
        print("🔮 Predicted Probability:", prob)

        return result

    except Exception as e:
        return f"❌ Error during prediction:\n{str(e)}"

# === GRADIO INTERFACE ===
demo = gr.Interface(
    fn=predict_fraud,
    inputs=gr.Textbox(lines=6, label="Paste a job posting"),
    outputs=gr.Text(label="Fraud Risk Assessment"),
    title="JobScout AI — TF-IDF + Red Flag Booster",
    description=(
        "Paste a job ad below to assess its fraud risk.\n\n"
        "🚩 Posts scoring ≥ 83% are flagged as high risk.\n"
        "🟡 Posts scoring ≥ 85% are flagged for caution.\n"
        "📝 All entries are logged for ongoing analysis and improvements."
    )
)

if __name__ == "__main__":
    demo.launch()
