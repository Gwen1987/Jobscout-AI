# 🕵️ JobScout AI

JobScout AI is a neural network-based binary classification project built to detect fraudulent job postings. Using natural language processing (NLP) and TensorFlow, it analyzes job descriptions and titles to predict the likelihood of fraud, aiming to support job seekers by filtering out deceptive listings.

---

## 🌐 Live Demo on Hugging Face

Want to test JobScout AI instantly? Try it now on [Hugging Face Spaces](https://huggingface.co/spaces/gwen-s/jobscout-ai)!



**Usage:**

1. Paste a full job posting (title + description) into the textbox.
2. Receive a real-time fraud risk assessment.
3. All submissions are logged internally for future model improvements.

⚠️ *This version uses only the job text and flags jobs with a fraud probability ≥ 85%.*

---

## 🗓️ Threshold Optimization

JobScout AI optimizes its decision boundary based on precision-recall tradeoffs, selecting a fraud classification threshold of **0.85**. This decision balances strong recall with meaningful precision while minimizing false positives.

| Threshold | Precision  | Recall     | F1 Score   |
| --------- | ---------- | ---------- | ---------- |
| 0.50      | 0.4646     | 0.8678     | 0.6052     |
| 0.60      | 0.4935     | 0.8678     | 0.6292     |
| 0.70      | 0.5261     | 0.8678     | 0.6551     |
| **0.85**  | **0.5781** | **0.8506** | **0.6884** |
| 0.90      | 0.61       | 0.8448     | 0.7084     |

> ✨ **Note:** Thresholds >0.90 had slightly better precision but sacrificed recall. Thresholds <0.35 began introducing false positives in low-risk bins.

---

## 📊 Fraud Risk Binning Justification

The model assigns risk tiers based on predicted fraud probability:

- 🔵 Legit: < 0.35
- 🟡 Caution Advised: 0.35–0.84
- 🔴 High Risk: ≥ 0.85

Bin analysis shows that fraud rates are negligible under 0.35 and begin to rise sharply near the 0.85 mark:

| Probability Bin | Fraud Rate |
| --------------- | ---------- |
| 0.0–0.1         | 0.57%      |
| 0.1–0.2         | 8.70%      |
| 0.2–0.3         | 0.00%      |
| 0.3–0.4         | 0.00%      |
| 0.8–0.9         | 16.00%     |
| 0.9–1.0         | 60.99%     |

📄 This confirms **0.35** as an appropriate threshold to initiate caution.

---

## 📦 Project Structure

```
JobScout-AI/
├── .gitignore                    # Ignored files for version control
├── EDA.ipynb                     # Exploratory Data Analysis notebook
├── gradio_app.py                # Main Gradio app entry point
├── LICENSE                      # Project license
├── README.md                    # Project overview and usage
├── requirements.txt             # Core project dependencies
├── data/                        # Processed and raw datasets - files excluded from github due to size
├── images/                      # Visualizations and plots
   ├── confusion_matrix.png
   ├── threshold_optimization.png

├── logs/                        # Evaluation logs and final reports
    ├── classification_report.txt
    ├── eval_summary.md
    ├── jobscout_logs_final.csv
├── models/                      # Model + vectorizer artifacts
   ├── archived/                # Archived models
   ├── jobscout_tfidf_smoteenn.keras
   ├── jobscout_tfidf_vectorizer.pkl
├── notebook/                    # Metric visualizations (Jupyter)
   ├── precision_recall_graph.ipynb
├── real_world_examples/         # Real job ads for evaluation
├── scripts/                     # Training and evaluation scripts
   ├── analyze_threshold.py
   ├── classification_report.py
   ├── generate_eval_results.py
   ├── merge_training_data.py
   ├── split_train_validation.py
   ├── train_data_.py

```

---

## 🔍 Features

- 🤖 **Binary Fraud Classifier** — Real vs Fake job posting prediction
- 🧠 **Dense Neural Network** — Optimized with class weights and dropout layers
- ⚖️ **SMOTEENN Resampling** — Hybrid oversampling + undersampling strategy
- 🔹 **Red Flag Booster** — Flags phrases often associated with scams
- 📊 **Metrics Tracking** — Accuracy, Precision, Recall, F1, Threshold Tuning

---

## 🧪 Model Testing Summary

| Model Version | Architecture                         | Precision (%) | Recall (%) | Val Accuracy (%) | Val Loss   |
| ------------- | ------------------------------------ | ------------- | ---------- | ---------------- | ---------- |
| v1            | Dense (16x2)                         | 95.05         | 55.49      | 97.71            | 0.0846     |
| v2            | Embedding + Dropout                  | 83.62         | 56.07      | 97.34            | 0.0889     |
| v3            | LSTM + Dropout                       | 51.96         | 84.39      | 95.47            | 0.1464     |
| **v4**        | Dense + Dropout + SMOTEENN + Booster | **57.81**     | **85.06**  | **95.93**        | **0.1394** |

---

## 🛠️ Tools & Technologies

- Python, Pandas, NumPy
- TensorFlow, Keras (Dense models)
- Scikit-learn, Imbalanced-learn (SMOTEENN)
- JupyterLab, Matplotlib, Seaborn
- Gradio for web app interface

---

## 📁 Data Source

Dataset: `fake_job_postings.csv` from Kaggle [fake-job-posting-prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

---

## ▶️ How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Gradio app locally:
   ```bash
   python gradio_app.py
   ```

---

## 🚀 Future Enhancements

- Add Transformer-based NLP model (e.g., BERT or DistilBERT) for richer contextual understanding

---

## 🧠 Author

Gwen Seymour — [LinkedIn](https://www.linkedin.com/in/gwen-seymour) | [GitHub](https://github.com/Gwen1987)

> JobScout AI was built to sharpen TensorFlow skills and explore real-world NLP challenges that affect job seekers worldwide.

