import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
import matplotlib.pyplot as plt

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

# Plot
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
plt.show()
