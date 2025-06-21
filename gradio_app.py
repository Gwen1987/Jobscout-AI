import gradio as gr
from tensorflow.keras.models import load_model

# Load the trained pipeline (vectorizer + model)
pipeline = load_model("jobscout_pipeline.keras")

# Prediction function
import tensorflow as tf

def predict_fraud(job_text):
    try:
        input_tensor = tf.constant([job_text])  # ✅ wrap input in a tensor
        prob = pipeline.predict(input_tensor)[0][0]
        label = "🚩 Fraud" if prob > 0.5 else "✅ Legit"
        return f"{label} — Probability: {prob:.2%}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Gradio interface
demo = gr.Interface(
    fn=predict_fraud,
    inputs=gr.Textbox(lines=6, label="Paste a job posting"),
    outputs=gr.Text(label="Prediction"),
    title="JobScout AI — Fraud Detection",
    description="Paste a job ad below to detect if it's potentially fraudulent."
)

# Launch the app
demo.launch()
