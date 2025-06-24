import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np
import os

# === Load eval results ===
df = pd.read_csv("data/your_eval_results.csv")  # Use path from generate_eval_results.py

y_true = df["y_true"].values
y_probs = df["probability"].values

# === Calculate Precision, Recall, and F1 across thresholds ===
precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)

# Best threshold
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]

# === Plot 1: Threshold Optimization ===
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], label="Precision", color="blue")
plt.plot(thresholds, recalls[:-1], label="Recall", color="orange")
plt.plot(thresholds, f1_scores[:-1], label="F1 Score", color="green")
plt.axvline(best_threshold, color="red", linestyle="--", label=f"Best Threshold = {best_threshold:.2f}")
plt.title("Threshold Optimization")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()

os.makedirs("images", exist_ok=True)
plt.savefig("images/threshold_optimization_smoteenn.png", dpi=300)
plt.show()

print(f"✅ Best Threshold: {best_threshold:.4f} with F1 Score: {best_f1:.4f}")

# === Analyze Probability Bins ===
bins = np.arange(0, 1.05, 0.1)  # 0.0 to 1.0 in 0.1 steps
labels = [f"{b:.1f}-{b+0.1:.1f}" for b in bins[:-1]]
df["prob_bin"] = pd.cut(df["probability"], bins=bins, labels=labels, include_lowest=True)

fraud_rate_df = (
    df.groupby("prob_bin")
    .agg(
        total=("probability", "count"),
        num_fraud=("y_true", "sum")
    )
    .assign(fraud_rate=lambda d: d["num_fraud"] / d["total"])
    .reset_index()
)

print("\n📊 Fraud Rate by Predicted Probability Bin:\n")
print(fraud_rate_df)

os.makedirs("logs", exist_ok=True)
fraud_rate_df.to_csv("logs/probability_bin_fraud_rates.csv", index=False)

# === Plot 2: Fraud Rate by Bin ===
plt.figure(figsize=(10, 5))
plt.bar(fraud_rate_df["prob_bin"], fraud_rate_df["fraud_rate"], color="orange")
plt.title("Fraud Rate by Predicted Probability Bin")
plt.xlabel("Predicted Fraud Probability")
plt.ylabel("Actual Fraud Rate")
plt.ylim(0, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("images/fraud_rate_by_bin.png", dpi=300)
plt.show()
