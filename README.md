# 🚢 Titanic Survival Prediction — LightGBM vs XGBoost

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-9ACD32?style=for-the-badge&logo=leaf&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

*A comparative study of two leading gradient boosting algorithms on the classic Titanic survival dataset.*

</div>

---

## 📌 Project Overview

This project benchmarks **LightGBM (LGBM)** and **XGBoost (XGBM)** — two of the most powerful gradient boosting frameworks — on the Titanic dataset. The goal is to predict passenger survival based on features like age, gender, passenger class, and fare, while rigorously evaluating and comparing both models using standard classification metrics.

---

## 🎯 Objectives

- Perform **Exploratory Data Analysis (EDA)** on the Titanic dataset
- Preprocess and clean training and test data consistently
- Build and train **LightGBM** and **XGBoost** classifiers
- Compare models using **Accuracy, Precision, Recall, F1-Score, and AUC-ROC**
- Visualize model performance via **Confusion Matrices, ROC Curves, and Feature Importance plots**

---

## 📂 Project Structure

```
LGBM-XGBoost-Assessment-Titanic-Survival-Prediction/
│
├── LGBM & XGBM Assessment.ipynb   # Main Jupyter Notebook
├── Titanic_train.csv               # Training dataset
├── Titanic_test.csv                # Test dataset
└── README.md
```

---

## 🔍 Exploratory Data Analysis

- Inspected missing values across all features
- Visualized distributions of numerical features using **histograms**
- Analyzed survival rates across **passenger class (Pclass)** and **gender (Sex)** using bar plots

---

## 🛠️ Data Preprocessing

Both training and test datasets underwent the same preprocessing pipeline to ensure consistency:

| Step | Detail |
|------|--------|
| **Column Removal** | Dropped `PassengerId`, `Name`, `Ticket`, `Cabin` (non-predictive) |
| **Missing Values** | `Age` → filled with **mean**; `Embarked` → filled with **mode** |
| **Encoding** | One-hot encoded `Sex` and `Embarked` using `pd.get_dummies()` with `drop_first=True` |

---

## 🤖 Models

### ⚡ LightGBM (`lgb.LGBMClassifier`)
- Leaf-wise tree growth strategy
- Faster training, especially on larger datasets
- Histogram-based algorithm for memory efficiency

### 🌳 XGBoost (`xgb.XGBClassifier`)
- Level-wise tree growth with regularization
- `eval_metric='logloss'`, `use_label_encoder=False`
- Robust to overfitting with built-in L1/L2 regularization

---

## 📊 Evaluation Metrics

Both models were evaluated using:

- **Accuracy** — overall correct predictions
- **Precision** — true positive rate among predicted positives
- **Recall** — true positive rate among actual positives
- **F1-Score** — harmonic mean of precision and recall
- **AUC-ROC** — area under the receiver operating characteristic curve

### Confusion Matrices
Side-by-side confusion matrices plotted for both models to visualize true vs. predicted survival outcomes.

### ROC Curves
Overlapping ROC curves for LightGBM and XGBoost with respective AUC scores displayed.

---

## 📈 Feature Importance

Feature importance was extracted and plotted for both models:

- **LightGBM** — importance derived from `split` count via `booster_.feature_importance()`
- **XGBoost** — importance from `feature_importances_` attribute

Key features influencing predictions include **Sex**, **Pclass**, **Age**, and **Fare**.

---

## 📦 Requirements

```bash
pip install lightgbm xgboost scikit-learn pandas seaborn matplotlib
```

Or install all at once:

```bash
pip install -r requirements.txt
```

**Core dependencies:**

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Train/test split, metrics, encoding |
| `lightgbm` | LightGBM classifier |
| `xgboost` | XGBoost classifier |
| `seaborn` | Statistical visualizations |
| `matplotlib` | Plotting |

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/syedaltamash-analytics/LGBM-XGBoost-Assessment-Titanic-Survival-Prediction.git
   cd LGBM-XGBoost-Assessment-Titanic-Survival-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install lightgbm xgboost scikit-learn pandas seaborn matplotlib
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook "LGBM & XGBM Assessment.ipynb"
   ```

4. Update the dataset paths in the notebook to point to your local `Titanic_train.csv` and `Titanic_test.csv` files.

---

## 💡 Key Takeaways

- **LightGBM** offers faster training and is well-suited for larger datasets due to its histogram-based approach
- **XGBoost** provides strong regularization which can reduce overfitting on smaller datasets
- Both models achieve competitive performance on the Titanic dataset with strong precision, recall, and F1-scores
- Feature importance plots reveal that **gender and passenger class** are the most influential survival predictors — consistent with historical records

---

## 👤 Author

<div align="center">

### Syed Mohd Altamash
*Data Science & Machine Learning Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-syedaltamash--analytics-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/syedaltamash-analytics)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Syed%20Mohd%20Altamash-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/syedaltamash-analytics)

</div>

---

<div align="center">
⭐ If you found this project useful, consider giving it a star!
</div>
