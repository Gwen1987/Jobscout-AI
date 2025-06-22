import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load predictions
df = pd.read_csv("data/your_eval_results.csv")
y_true = df["y_true"]
y_probs = df["probability"]

# Compute metrics
precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)

# Best threshold by F1
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
print(f"✅ Best threshold: {best_threshold:.2f} (F1 = {f1_scores[best_index]:.2f})")

# Create output dirs
os.makedirs("images", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Plot threshold optimization
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.plot(thresholds, f1_scores[:-1], label="F1 Score")
plt.axvline(best_threshold, color='red', linestyle='--', label=f"Best Threshold = {best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("images/threshold_optimization.png")
plt.close()

# Plot PR curve
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("images/precision_recall_curve.png")
plt.close()

# Save best threshold info
with open("logs/eval_summary.md", "w") as f:
    f.write("# Evaluation Summary\n\n")
    f.write(f"**Best Threshold (F1 Max):** {best_threshold:.4f}\n")
    f.write(f"**F1 Score:** {f1_scores[best_index]:.4f}\n")
