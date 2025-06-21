import tensorflow as tf
import numpy as np
import json

# Load vectorizer config + weights
with open("vectorizer_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

vectorizer = tf.keras.layers.TextVectorization.from_config(config)

weights = np.load("vectorizer_weights.npz")
vectorizer.set_weights([weights[key] for key in weights])

# Load model
model = tf.keras.models.load_model("jobscout_model.keras")

# Define prediction function
def predict_fraud(title: str, description: str) -> dict:
    text = title + " " + description
    tokenized = vectorizer(tf.constant([text]))
    prediction = model(tokenized, training=False)[0][0].numpy()
    return {
        "fraudulent": bool(prediction > 0.5),
        "confidence": float(prediction)
    }
