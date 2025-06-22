import csv
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load model and vectorizer
model = load_model("models/jobscout_pipeline.keras")
with open("models/jobscout_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Open output CSV
with open("data/your_eval_results.csv", "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["text", "y_true", "prediction", "probability"])

    with open("data/validation_data.csv", newline='', encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            text = row["text"]
            y_true = int(row["fraudulent"])

            # Transform and predict
            vec = vectorizer.transform([text]).toarray()
            prob = model.predict(vec)[0][0]
            pred = 1 if prob > 0.5 else 0

            writer.writerow([text[:1000], y_true, pred, round(prob, 4)])
