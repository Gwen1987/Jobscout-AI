# 🕵️‍♀️ JobScout AI

JobScout AI is a neural network-based binary classification project built to detect fraudulent job postings. Using natural language processing (NLP) and TensorFlow, it analyzes job descriptions and titles to predict the likelihood of fraud, aiming to support job seekers by filtering out deceptive listings.

---

## 📦 Project Structure

```
JobScout-AI/
├── data/                        # CSV dataset of real vs fake job posts
├── src/                         # Training + preprocessing scripts
├── notebooks/                   # Optional EDA and training logs
├── images/                      # Visualizations and plots
├── logs/                        # Training logs + metrics
├── README.md                    # Project overview and usage
└── requirements.txt             # Project dependencies
```

---

## 🔍 Features

* 🤖 **Binary Fraud Classifier** — Real vs Fake job posting prediction
* 🧠 **LSTM Neural Network** — Advanced architecture for text classification
* ⚖️ **Class Weighting** — Addressing dataset imbalance
* 📊 **Validation Metrics Tracking** — Accuracy, Precision, Recall, Loss
* 📈 **Visualizations** — Metrics comparison across 3 model versions

---

## 🧪 Model Testing Summary

| Simulation | Architecture        | Precision (%) | Recall (%) | Val Accuracy (%) | Val Loss |
| ---------- | ------------------- | ------------- | ---------- | ---------------- | -------- |
| Sim 1      | Dense (16x2)        | 95.05         | 55.49      | 97.71            | 0.0846   |
| Sim 2      | Embedding + Dropout | 83.62         | 56.07      | 97.34            | 0.0889   |
| Sim 3      | LSTM + Dropout      | 51.96         | 84.39      | 95.47            | 0.1464   |

🔗 [View full training logs](logs/training_log.md)

---

## 🧰 Tools & Technologies

* Python, Pandas, NumPy
* TensorFlow, Keras
* Scikit-learn
* JupyterLab (for EDA and prototyping)
* Matplotlib (visualizations)

---

## 📊 Visual Summary

![Precision vs Recall](images/precision_recall_comparison.png)

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
3. Navigate to `src/` and run:

   ```bash
   python train_data.py
   ```

---

## 🚀 Future Enhancements

* Add GloVe or FastText embeddings
* Build web form for testing job posts
* ROC & PR-AUC visual dashboards
* Deploy with FastAPI

---

## 🧠 Author

Gwen Seymour — [LinkedIn](https://www.linkedin.com/in/gwen-seymour) | [GitHub](https://github.com/Gwen1987)

> JobScout AI was built to sharpen TensorFlow skills and explore real-world NLP challenges that affect job seekers worldwide.
