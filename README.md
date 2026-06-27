# 🩺 Diabetes Prediction using Machine Learning

A complete end-to-end machine learning pipeline for predicting diabetes probability using demographic, lifestyle, and behavioral survey features. This project was developed as a solution for the Kaggle **Playground Series - Season 5, Episode 12** competition and demonstrates the complete data science workflow from exploratory data analysis to model interpretation.

---

## 📌 Project Overview

Early detection of diabetes can significantly improve patient outcomes. In this project, several machine learning models were trained and compared to predict the probability of diabetes based on lifestyle and health indicator data — not clinical lab tests, but behavioral and demographic survey features similar to those collected by the CDC's Behavioral Risk Factor Surveillance System (BRFSS).

The workflow includes:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Feature engineering
* Categorical feature encoding
* Model training and evaluation
* Bayesian hyperparameter tuning with Optuna
* Cross-validation
* Model explainability using SHAP
* Final prediction generation for Kaggle submission

---

## 📊 Dataset

**Source:** Kaggle Playground Series - Season 5, Episode 12

The dataset was synthetically generated from a deep learning model trained on the Diabetes Health Indicators Dataset (CDC BRFSS). It contains demographic, lifestyle, and behavioral survey information — no clinical lab values such as glucose or HbA1c.

Features include:

* Age, Gender, Ethnicity
* BMI, Waist-to-Hip Ratio
* Education Level, Income Level, Employment Status
* Smoking Status, Alcohol Consumption per Week
* Physical Activity Minutes per Week
* Sleep Hours per Day, Screen Time per Day
* Diet Score
* Hypertension History, Cardiovascular History
* Family History of Diabetes
* Cholesterol (Total, HDL, LDL), Triglycerides
* Systolic/Diastolic Blood Pressure, Heart Rate

**Target Variable:** `diagnosed_diabetes` (probability, evaluated with ROC-AUC)

---

## 🛠 Project Pipeline

### 1. Exploratory Data Analysis

* Class balance analysis (synthetic data is ~62% positive vs ~14% in original CDC data — rebalanced by the generative model)
* Distribution plots for continuous features
* Countplots for binary/ordinal features
* Correlation heatmap (top 10 features)
* Normalized rate analysis to test feature hypotheses correctly (raw counts vs actual diabetes rates per group)
* Comparison of synthetic vs original dataset distributions

### 2. Data Preprocessing

* One-Hot Encoding for nominal categorical variables (gender, ethnicity, smoking status, employment status)
* Ordinal Encoding for ordered categories (education level, income level) with meaningful order preserved
* Encoding performed jointly on train and test datasets to prevent column mismatch

### 3. Feature Engineering

Two domain-informed features were created based on SHAP analysis:

**Physical Activity BMI Ratio**
```
BMI / (Physical Activity Minutes per Week + 1)
```
Captures the balance between a key risk factor (BMI) and a protective factor (activity). The +1 prevents division by zero for sedentary patients.

**Family History × Age**
```
Family History of Diabetes × Age
```
Encodes the idea that genetic predisposition that has had more time to manifest (older age) represents higher cumulative risk.

### 4. Machine Learning Models

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.6943 |
| Random Forest | 0.6928 |
| LightGBM (default) | 0.7259 |
| LightGBM (tuned) | 0.7264 |

Final model: **LightGBM** with Bayesian hyperparameter tuning via Optuna.

### 5. Hyperparameter Tuning

Used **Optuna** (Bayesian optimization) over 50 trials with Stratified 5-Fold CV to find optimal parameters:

```python
num_leaves: 61
learning_rate: 0.0615
min_child_samples: 91
subsample: 0.7087
```

---

## 📈 Results

| Metric | Score |
|---|---|
| CV ROC-AUC (5-Fold) | 0.7264 |
| Public Leaderboard AUC | 0.6954 |

---

## 🔍 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was applied to understand both global feature importance and individual prediction drivers.

**Top 5 most important features (global):**
1. Physical Activity per Minute
2. Family History of Diabetes
3. Age
4. Triglycerides
5. BMI

**Key clinical insights from the beeswarm plot:**
- **Physical activity** is a *protective* factor — patients with high activity have lower predicted diabetes risk (high values push SHAP left)
- **Family history** is a *risk amplifier* — having a family history consistently pushes predictions toward diabetes (high values push SHAP right)
- **Triglycerides** appearing in the top 5 aligns with the known clinical role of dyslipidemia in insulin resistance — the model learned this from behavioral survey data alone, without any lab measurements

---

## 📁 Project Structure

```
├── diabetes-prediction.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
└── README.md
```

---

## 🚀 Libraries Used

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
lightgbm
shap
optuna
```

---

## 🎯 Future Improvements

* Ensemble learning (Stacking/Blending with XGBoost and CatBoost)
* Probability calibration (Platt scaling or isotonic regression) — important in medical contexts where calibrated probabilities matter clinically
* Advanced feature engineering based on deeper clinical domain knowledge
* Experiment tracking with MLflow
* Investigating the train/test distribution gap to reduce the CV-to-leaderboard score drop

---

## 📄 License

This project is intended for educational and portfolio demonstration purposes.
