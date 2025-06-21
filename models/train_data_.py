import sys
import os
import contextlib
import unicodedata
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight

# --- Force UTF-8 encoding behavior ---
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

# === Load and preprocess data ===
with open("data/preprocessed_train_data.csv", "r", encoding="utf-8", errors="replace") as f:
    df = pd.read_csv(f)
X = df["text"].fillna("").values
y = df["fraudulent"].values

# === Create and adapt the TextVectorization layer ===
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=300
)
vectorizer.adapt(X)

# === Clean and deduplicate vectorizer vocabulary to prevent Unicode save crash ===
vocab = vectorizer.get_vocabulary()
cleaned_vocab = [
    unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode("ascii")
    for word in vocab
]

# Deduplicate while preserving order
seen = set()
unique_cleaned_vocab = []
for word in cleaned_vocab:
    if word not in seen:
        seen.add(word)
        unique_cleaned_vocab.append(word)

vectorizer.set_vocabulary(unique_cleaned_vocab)

# === Compute class weights ===
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = {int(i): float(w) for i, w in enumerate(weights)}

# === Build the model ===
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=300),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# === Vectorize input for training ===
X_vec = vectorizer(X).numpy()

# === Train the model ===
model.fit(X_vec, y, epochs=5, class_weight=class_weights)

# === Combine vectorizer and model into full pipeline ===
pipeline = tf.keras.Sequential([vectorizer, model])

# === Call the pipeline once to build it ===
_ = pipeline(tf.constant(["trigger shape build"]))

# === Safe save: redirect stdout + stderr to avoid Unicode crashes ===
def safe_save_pipeline(model, filename="jobscout_pipeline.keras"):
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        model.save(filename)

safe_save_pipeline(pipeline)
