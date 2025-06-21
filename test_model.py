from tensorflow.keras.models import load_model

# === Load the saved model pipeline ===
pipeline = load_model("jobscout_pipeline.keras")

# === Sample job postings to test ===
samples = [
    "Remote data entry job — no experience needed, start today!",
    "We are hiring an experienced CPA to lead audits for public companies.",
    "Earn money fast from home. Limited spots, click now!",
    "Full-time developer role in Toronto, hybrid schedule, salary disclosed."
]

# === Run predictions ===
predictions = pipeline.predict(samples)

# === Print results ===
for text, pred in zip(samples, predictions):
    label = "FRAUD" if pred[0] > 0.5 else "Legit"
    print("\nText:", text)
    print(f"Probability: {pred[0]:.4f} → {label}")
