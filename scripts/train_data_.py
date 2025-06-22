import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pickle

# === CONFIGURATION ===
USE_CLASS_WEIGHT = True
MODEL_OUTPUT = "models/jobscout_pipeline_v1.keras"
VECTORIZER_OUTPUT = "models/jobscout_vectorizer.pkl"

# === LOAD DATA ===
df = pd.read_csv("data/preprocessed_train_data_v3.csv")
X_text_raw = df["text"]
y = df["fraudulent"]

# === TEXT VECTORIZATION ===
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_text = vectorizer.fit_transform(X_text_raw)

# === SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# === CLASS WEIGHTING ===
class_weight = None
if USE_CLASS_WEIGHT:
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(weights))

# === MODEL ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === TRAIN ===
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(
    X_train.toarray(), y_train,
    validation_data=(X_val.toarray(), y_val),
    epochs=10,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[early_stop]
)

# === SAVE ARTIFACTS ===
model.save(MODEL_OUTPUT)

with open(VECTORIZER_OUTPUT, "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Training complete. Artifacts saved:")
print("   -", MODEL_OUTPUT)
print("   -", VECTORIZER_OUTPUT)
