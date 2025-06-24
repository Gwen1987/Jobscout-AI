import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

from tensorflow.keras.layers import (
    TextVectorization, Input, Embedding, LSTM, Dense,
    Dropout, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === Load Data ===
df = pd.read_csv("data/preprocessed_train_data_v3.csv")
df["text_len"] = df["text"].str.len()

X_text_raw = df["text"]
y = df["fraudulent"]
X_len = df["text_len"].values.reshape(-1, 1)

# === Normalize length values ===
scaler = MinMaxScaler()
X_len_scaled = scaler.fit_transform(X_len)

# === Train/Test Split ===
X_text_train, X_text_val, X_len_train, X_len_val, y_train, y_val = train_test_split(
    X_text_raw, X_len_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# === Vectorization ===
MAX_TOKENS = 10000
MAX_LEN = 300

vectorizer = TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LEN
)
vectorizer.adapt(X_text_train)

# === Model Definition ===
text_input = Input(shape=(MAX_LEN,), name="text_input")
embedding = Embedding(input_dim=MAX_TOKENS, output_dim=64)(text_input)
lstm = LSTM(64)(embedding)
dropout_1 = Dropout(0.2)(lstm)

len_input = Input(shape=(1,), name="len_input")
concat = Concatenate()([dropout_1, len_input])
dense = Dense(64, activation="relu")(concat)
dropout_2 = Dropout(0.2)(dense)
output = Dense(1, activation="sigmoid")(dropout_2)

model = Model(inputs=[text_input, len_input], outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# === Vectorize Text ===
X_text_train_vec = vectorizer(np.array(X_text_train)).numpy()
X_text_val_vec = vectorizer(np.array(X_text_val)).numpy()

# === Class Weights for Imbalanced Data ===
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# === Callbacks ===
os.makedirs("models/checkpoints", exist_ok=True)
checkpoint = ModelCheckpoint("models/checkpoints/textlen_best.keras", save_best_only=True)
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# === Train ===
history = model.fit(
    {"text_input": X_text_train_vec, "len_input": X_len_train},
    y_train,
    validation_data=({"text_input": X_text_val_vec, "len_input": X_len_val}, y_val),
    epochs=15,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

# === Save Model + Artifacts ===
model.save("models/jobscout_textlen_model.keras")

with open("models/jobscout_vectorizer_config.pkl", "wb") as f:
    pickle.dump(vectorizer.get_config(), f)

with open("models/jobscout_vectorizer_vocab.txt", "w", encoding="utf-8") as f:
    vocab = vectorizer.get_vocabulary()
    unique_vocab = list(dict.fromkeys(vocab))  # deduplicate and preserve order
    f.write("\n".join([v for v in unique_vocab if v]))

with open("models/jobscout_len_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and vectorizer artifacts saved successfully.")
