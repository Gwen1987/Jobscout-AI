import pandas as pd
from collections import Counter
import re

# === LOAD DATA ===
df = pd.read_csv("data/preprocessed_train_data_v3.csv")  # adjust path as needed

# === SPLIT ===
df_fraud = df[df["fraudulent"] == 1]
df_legit = df[df["fraudulent"] == 0]

# === TOKENIZE ===
def tokenize(text):
    return re.findall(r"\b[a-z]{3,}\b", text.lower())

fraud_tokens = Counter()
legit_tokens = Counter()

for text in df_fraud["text"].dropna():
    fraud_tokens.update(tokenize(text))
for text in df_legit["text"].dropna():
    legit_tokens.update(tokenize(text))

# === COMBINE & RANK ===
df_tokens = pd.DataFrame.from_dict(fraud_tokens, orient="index", columns=["fraud_count"])
df_tokens["legit_count"] = df_tokens.index.map(lambda w: legit_tokens.get(w, 0))
df_tokens["fraud_ratio"] = df_tokens["fraud_count"] / (df_tokens["legit_count"] + 1)
df_tokens["total_count"] = df_tokens["fraud_count"] + df_tokens["legit_count"]
df_tokens = df_tokens[df_tokens["fraud_count"] >= 3]  # keep meaningful terms

# === CANDIDATES FOR BOOSTER ===
booster_terms = df_tokens[
    (df_tokens["fraud_ratio"] >= 3.0) &
    (df_tokens["fraud_count"] >= 5)
].sort_values(by="fraud_ratio", ascending=False).head(20)

# === HIGH-NOISE WORDS ===
noise_terms = df_tokens[
    (df_tokens["fraud_ratio"] < 1.5) &
    (df_tokens["total_count"] > 100)
].sort_values(by="total_count", ascending=False).head(20)

# === SHOW RESULTS ===
print("=== 🔥 Top 20 Booster Candidates ===")
print(booster_terms[["fraud_count", "legit_count", "fraud_ratio"]])

print("\n=== ⚠️ Top 20 High-Noise Terms ===")
print(noise_terms[["fraud_count", "legit_count", "fraud_ratio"]])


booster_terms.to_csv("data/booster_red_flags.csv")
noise_terms.to_csv("data/high_noise_terms.csv")
