from sklearn.metrics import classification_report,  confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import os
import matplotlib.pyplot as plt



# Load results and print classification report
df = pd.read_csv("data/your_eval_results.csv")
print(classification_report(df["y_true"], df["prediction"], digits=4))
# Save classification report to a text file
with open("logs/classification_report.txt", "w") as f:
    f.write(classification_report(df["y_true"], df["prediction"], digits=4))


#create confusion matrix and save it as an image
# Load predictions
y_true = df["y_true"]
y_pred = df["prediction"]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])

# Plot and save
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()

# Ensure directory exists
os.makedirs("images", exist_ok=True)
plt.savefig("images/confusion_matrix.png")
plt.close()
