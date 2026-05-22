[README.md](https://github.com/user-attachments/files/28160726/README.md)
# 🩺 Diabetes Prediction Using Machine Learning — CBIO313

This project explores how clinical features such as **HbA1c level**, **blood glucose**, and **BMI** can be used to predict diabetes using various **machine learning models**. The goal is to identify which model performs best on a real-world diabetes dataset.

> 🔍 **Problem Statement**
> Can we reliably predict whether a person has diabetes using HbA1c levels and other health metrics?

---

## 📂 Table of Contents

- [Overview](#-overview)
- [Dataset](#️-dataset)
- [Preprocessing](#-preprocessing)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering & Selection](#-feature-engineering--selection)
- [Modeling & Evaluation](#-modeling--evaluation)
- [Performance Comparison](#-performance-comparison)
- [Installation](#-installation)
- [Folder Structure](#-folder-structure)

---

## 🧠 Overview

This project investigates how well different ML algorithms can classify patients into diabetic or non-diabetic groups, with a focus on **HbA1c level** — a standard long-term indicator of blood sugar control.

We trained and compared models ranging from simple baselines like Logistic Regression and KNN, to powerful ensemble techniques including **Random Forest**, **XGBoost**, **Gradient Boosting**, **Bagging**, **Voting**, and **Stacking**.

---

## 🗂️ Dataset

- **File**: `diabetes_dataset.csv`
- **Size**: 100,000 rows
- **Task**: Binary classification (Diabetes: 0 = No, 1 = Yes)

### ⚙️ Features Used

| Feature | Description |
|---|---|
| `age` | Patient age |
| `gender` | Male / Female |
| `bmi` | Body Mass Index |
| `hbA1c_level` | Glycated hemoglobin — key predictor |
| `blood_glucose_level` | Blood glucose level |
| `smoking_history` | Smoking history category |
| `hypertension` | Hypertension status (0/1) |
| `heart_disease` | Heart disease status (0/1) |
| `diabetes` | **Target** (0 = No, 1 = Yes) |

> The `location` column was dropped as it had no predictive value.

---

## 🧼 Preprocessing

- Checked for and confirmed **no missing values or duplicates**
- Dropped the irrelevant `location` column
- Encoded categorical variables:
  - `gender`: Male → 0, Female → 1
  - `smoking_history`: ordinal mapping (No Info=0, never=1, former=2, current=3, not current=4, ever=5)
- Applied **StandardScaler** to: `age`, `bmi`, `hbA1c_level`, `blood_glucose_level`
- Split data: **80% training / 20% testing** (`random_state=42`)

---

## 📊 Exploratory Data Analysis

Visualizations included:

- Target variable distribution (class balance check)
- Age distribution histogram
- BMI vs. Diabetes status — box plot
- Blood glucose & HbA1c by diabetes status — violin plots
- **Correlation heatmap** across all features

> 🔎 **HbA1c level** and **blood glucose** emerged as the strongest predictors of diabetes.

---

## 🔧 Feature Engineering & Selection

- Created a new binary feature: `high_glucose` (blood glucose > 140)
- Used **Random Forest feature importances** to rank all features
- Selected the **top 5 most important features** for final model training
- Verified feature-target linearity using logistic regression trend plots

---

## 🤖 Modeling & Evaluation

We trained the following models:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Bagging Classifier
- Hard Voting Classifier
- Soft Voting Classifier
- Stacking Classifier

All models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve & AUC Score

---

## 📈 Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Random Forest | 0.97 | 0.97 | 0.68 | 0.80 |
| XGBoost | 0.97 | 0.96 | 0.69 | 0.81 |
| Gradient Boosting | 0.97 | 0.99 | 0.68 | 0.81 |
| AdaBoost | 0.97 | 0.97 | 0.69 | 0.81 |
| Bagging | 0.97 | 0.94 | 0.70 | 0.80 |
| Stacking | 0.97 | 0.91 | 0.67 | 0.77 |
| Logistic Regression | 0.96 | 0.87 | 0.62 | 0.73 |
| KNN | 0.96 | 0.90 | 0.57 | 0.70 |
| Hard Voting | 0.9652 | 0.93 | 0.64 | 0.76 |
| Soft Voting | 0.9649 | 0.89 | 0.67 | 0.77 |
| Decision Tree | 0.95 | 0.70 | 0.73 | 0.72 |
| Naive Bayes | 0.91 | 0.46 | 0.66 | 0.54 |
| SVM | 0.91 | 0.00 | 0.00 | 0.00 |

<details>
<summary><b>🔽 Click to expand performance comparison code</b></summary>

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_summary = pd.DataFrame({
    'Model': [
        'Bagging', 'Stacking', 'Hard Voting', 'Soft Voting', 'XGBoost',
        'Gradient Boosting', 'AdaBoost', 'Naive Bayes', 'SVM',
        'Decision Tree', 'Logistic Regression', 'KNN', 'Random Forest'
    ],
    'Accuracy': [
        0.97, 0.97, 0.9652, 0.9649, 0.97,
        0.97, 0.97, 0.91, 0.91, 0.95, 0.96, 0.96, 0.97
    ],
    'Precision': [
        0.94, 0.91, 0.93, 0.89, 0.96,
        0.99, 0.97, 0.46, 0.00, 0.70, 0.87, 0.90, 0.97
    ],
    'Recall': [
        0.70, 0.67, 0.64, 0.67, 0.69,
        0.68, 0.69, 0.66, 0.00, 0.73, 0.62, 0.57, 0.68
    ],
    'F1 Score': [
        0.80, 0.77, 0.76, 0.77, 0.81,
        0.81, 0.81, 0.54, 0.00, 0.72, 0.73, 0.70, 0.80
    ]
})

df_melted = df_summary.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(15, 7))
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric')
plt.title('Comparison of Machine Learning Model Performance Metrics')
plt.xticks(rotation=30, ha='right')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
```

</details>

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Abdalmajed8/CBIO313-Data-Mining-and-Machine-Learning-Project.git
   cd CBIO313-Data-Mining-and-Machine-Learning-Project
   ```

2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels joblib
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook "ML Full NoteBook Final.ipynb"
   ```

---

## 📁 Folder Structure

```
├── ML Full NoteBook Final.ipynb   # Full ML pipeline notebook
├── diabetes_dataset.csv           # Dataset
└── README.md
```

---

## 👤 Author

**Abdalmajed** — CBIO313 Course Project, Nile University
