import csv
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import classification_report
import pandas as pd
from scipy.sparse import hstack
import numpy as np

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

# === Load model and vectorizer ===
model = load_model("models/jobscout_tfidf_smoteenn.keras")
with open("models/jobscout_tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# === Evaluation Threshold ===
THRESHOLD = 0.83

# === Prepare output CSV ===
with open("data/your_eval_results.csv", "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["text", "y_true", "prediction", "probability"])

    with open("data/validation_data.csv", newline='', encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            text = row["text"]
            y_true = int(row["fraudulent"])

            # Transform + booster
            vec = vectorizer.transform([text])
            booster = np.array([[red_flag_score(text)]])
            full_input = hstack([vec, booster])

            # Predict
            prob = model.predict(full_input)[0][0]
            pred = 1 if prob > THRESHOLD else 0

            writer.writerow([text[:1000], y_true, pred, round(prob, 4)])

# === Print final metrics ===
df = pd.read_csv("data/your_eval_results.csv")
print(classification_report(df["y_true"], df["prediction"], digits=4))
