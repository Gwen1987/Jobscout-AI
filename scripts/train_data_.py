import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTEENN
from scipy.sparse import hstack
import numpy as np
import pickle
import os

# === CONFIGURATION ===
USE_CLASS_WEIGHT = True
MODEL_OUTPUT = "models/jobscout_tfidf_smoteenn.keras"
VECTORIZER_OUTPUT = "models/jobscout_tfidf_vectorizer.pkl"

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

# === LOAD DATA ===
df_main = pd.read_csv("data/preprocessed_train_data_v3.csv")
df_patch = pd.read_csv("data/synthetic_patch_jobs.csv")

# Deduplicate and combine
df_patch = df_patch[~df_patch["description"].isin(df_main["description"])]
df = pd.concat([df_main, df_patch], ignore_index=True)

# Generate text column and booster
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
df["text"] = df[text_cols].fillna("").agg(" ".join, axis=1)
df["booster"] = df["text"].apply(red_flag_score)

# === SPLIT ===
X_text_raw = df["text"]
booster = df["booster"].values.reshape(-1, 1)
y = df["fraudulent"]

X_text_train, X_text_val, booster_train, booster_val, y_train, y_val = train_test_split(
    X_text_raw, booster, y, test_size=0.2, random_state=42, stratify=y
)

# === TF-IDF VECTORIZATION ===
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_vec_train = vectorizer.fit_transform(X_text_train)
X_vec_val = vectorizer.transform(X_text_val)

# Save fitted vectorizer
os.makedirs("models", exist_ok=True)
with open(VECTORIZER_OUTPUT, "wb") as f:
    pickle.dump(vectorizer, f)

# Stack with booster
X_train = hstack([X_vec_train, booster_train])
X_val = hstack([X_vec_val, booster_val])

# Save validation set
val_df = pd.DataFrame({
    "text": X_text_val.values,
    "fraudulent": y_val
})
val_df.to_csv("data/validation_data.csv", index=False)

# === RESAMPLE ===
smote = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# === CLASS WEIGHT ===
class_weight = None
if USE_CLASS_WEIGHT:
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(weights))

# === MODEL BUILD ===
input_dim = X_resampled.shape[1]
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === TRAIN ===
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(
    X_resampled.toarray(), y_resampled,
    validation_data=(X_val.toarray(), y_val),
    epochs=10,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# === SAVE MODEL ===
model.save(MODEL_OUTPUT)

print("✅ Training complete. Artifacts saved:")
print("   -", MODEL_OUTPUT)
print("   -", VECTORIZER_OUTPUT)
print("   - data/validation_data.csv")
