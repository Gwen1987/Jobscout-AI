Training Log: JobScout AI

This file documents all major simulation runs of the TensorFlow model for JobScout AI, including configuration details, training outcomes, and post-hoc analysis.

✅ Simulation 1 — Baseline (Unweighted, Shallow NN)

Date: [insert date]Tokenizer: TextVectorization (200 tokens)Architecture:

Dense(16, relu) → Dense(16, relu) → Dense(1, sigmoid)

No embedding, no dropout

Epochs: 5
Class Weights: None

Validation Results:

Accuracy: 97.71%

Precision: 95.05%

Recall: 55.49%

Loss: 0.0846

Test Results:

Accuracy: 98.24%

Precision: 96.88%

Recall: 61.42%

Loss: 0.0680

Summary:
Simulation 1 showed strong general accuracy and precision but moderate recall — the model missed nearly 40% of fraudulent jobs.

✅ Simulation 2 — Class-Weighted + Dropout (v2)

Date: [insert date]Tokenizer: TextVectorization (300 tokens)Architecture:

TextVectorization → Embedding(128)

Dropout(0.4) → GlobalAveragePooling1D

Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dense(1, sigmoid)

Epochs: 5
Class Weights: Balanced using sklearn class_weight

Validation Results:

Accuracy: 97.34%

Precision: 83.62%

Recall: 56.07%

Loss: 0.0889

Training Epoch Summary:

E1 → acc: 61.98%, precision: 5.82%, recall: 45.51%

E5 → acc: 88.89%, precision: 30.28%, recall: 92.13%

Summary:
Simulation 2 aimed to address recall and overfitting using dropout and class weighting. Early epochs showed poor precision but gradually improved. Final validation recall was slightly higher than v1, but at the cost of precision and interpretability.

📌 Conclusion

Simulation 1 slightly outperformed v2 in real-world balance of precision/recall. While v2 had more robust architecture and better recall in late epochs, its lower precision and higher complexity may reduce generalizability.

🧪 Next Steps for Simulation 3

To further reduce false negatives (i.e., improve recall) without losing too much precision:

✅ Add Bidirectional LSTM or GRU layer after Embedding

✅ Try pre-trained word embeddings (e.g. GloVe)

✅ Evaluate with AUC-ROC, PR-AUC metrics

✅ Add EarlyStopping + ModelCheckpoint

🚫 Avoid overtraining on minority class — test more epochs with lower learning rate

Feel free to log new experiments by appending below:

✅ Simulation 3 — LSTM-Enhanced Architecture

Date: [insert date]Tokenizer: TextVectorization (300 tokens)Architecture:

TextVectorization → Embedding(128) → Bidirectional(LSTM(64))

Dropout(0.3) → Dense(64, relu) → Dense(1, sigmoid)

Epochs: 5Class Weights: Balanced using sklearn class_weight

Validation Results:

Accuracy: 95.47%

Precision: 51.96%

Recall: 84.39%

Loss: 0.1464

Training Epoch Summary:

E1 → acc: 87.55%, precision: 18.67%, recall: 51.12%

E2 → acc: 93.65%, precision: 42.90%, recall: 92.13%

E3 → acc: 96.50%, precision: 59.13%, recall: 97.44%

E4 → acc: 98.50%, precision: 76.82%, recall: 99.61%

E5 → acc: 98.44%, precision: 76.87%, recall: 97.73%

Summary:
Simulation 3 significantly improved recall, achieving over 97% by the final epoch. While precision remained moderate, it was notably better than Simulation 2. This LSTM-enhanced architecture offers a more balanced fraud detection model and may serve as a foundation for GloVe or GRU-based enhancements in future experiments.

