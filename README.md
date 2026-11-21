# Loan Default Classifier

An end-to-end supervised machine learning pipeline for predicting credit default risk using the Credit Risk Dataset. This project covers exploratory data analysis (EDA), leak-free preprocessing, domain-specific feature engineering, multi-model benchmarking, hyperparameter tuning, threshold optimization, and model serialization.

---

## Problem Statement

Financial institutions face asymmetric risk when evaluating loan applications: approving a borrower who defaults (False Negative) is significantly more costly than declining a safe applicant (False Positive). Furthermore, credit risk datasets are naturally imbalanced, making standard accuracy a misleading metric.

This project builds a calibrated, leak-free classification pipeline to accurately rank and identify high-risk loan applicants using **ROC-AUC**, **PR-AUC**, and **Class-1 Recall / F1-score**.

---

## Dataset Overview

* **Source:** `credit_risk_dataset.csv`
* **Raw Size:** 32,581 rows × 12 columns
* **Cleaned Size:** 32,416 rows (165 duplicate records removed)
* **Missing Values:** `person_emp_length` (895 missing), `loan_int_rate` (3,116 missing)
* **Target Variable (`loan_status`):**
  * `0` (Non-Default): **78.13%**
  * `1` (Default): **21.87%**

---

## Feature Engineering & Leak-Free Preprocessing

To prevent data leakage, all transformations are fit strictly on the training split and applied to validation and test splits via `.transform()`:

1. **Stratified Split (60% / 20% / 20%):**
   * Training set: `19,449` rows
   * Validation set: `6,483` rows
   * Test set: `6,484` rows
2. **Imputation:** Missing values in `person_emp_length` and `loan_int_rate` are imputed using `SimpleImputer(strategy='median')` fit only on training data.
3. **Engineered Features (computed post-imputation):**
   * `interest_burden = loan_amnt * (loan_int_rate / 100)` — captures total annual interest pressure (MI Score: `0.0601`).
   * `income_buffer = person_income - loan_amnt` — captures remaining financial cushion after loan principal (MI Score: `0.0828`).
4. **Feature Selection & Encoding:**
   * Dropped low-signal features (`person_age`, `cb_person_cred_hist_length`).
   * Scaled numerical features with `StandardScaler` and encoded categorical features with `DictVectorizer(sparse=False)`.

---

## Model Benchmarking (Validation Set)

Four model families were evaluated on the untouched validation set (`6,483` rows) using class-balanced weights to handle the 78/22 target imbalance:

| Model | Train AUC | Val ROC-AUC | Val PR-AUC | Accuracy | Recall (Default) | F1 (Default) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 1.0000 | **0.9294** | **0.8812** | **0.9224** | 0.7518 | **0.8091** |
| Decision Tree (`max_depth=6`) | 0.8986 | 0.8960 | 0.8270 | 0.9021 | 0.7179 | 0.7623 |
| Logistic Regression | 0.8700 | 0.8725 | 0.7248 | 0.8075 | **0.7800** | 0.6393 |
| Dummy Classifier (Baseline) | 0.5000 | 0.5000 | 0.2187 | 0.7813 | 0.0000 | 0.0000 |

---

## Hyperparameter Tuning & Threshold Optimization

The baseline Random Forest showed signs of overfitting (`Train AUC = 1.0000` vs. `Val AUC = 0.9294`). Using `RandomizedSearchCV` (5-fold CV, `n_iter=25`, `scoring='roc_auc'`), the optimal regularized configuration was found:

* **`n_estimators`:** `300`
* **`max_depth`:** `14`
* **`min_samples_leaf`:** `1`
* **`max_features`:** `'sqrt'`
* **`class_weight`:** `'balanced'`

This improved validation ROC-AUC to **0.9303** while reducing overfitting (`Train AUC = 0.9925`).

### Decision Threshold Tuning (Validation Set)

| Threshold | Recall (Default) | Precision (Default) | F1-Score (Default) |
| :---: | :---: | :---: | :---: |
| `0.50` | 0.7546 | 0.8756 | 0.8106 |
| **`0.45` (Selected)** | **0.7779** | **0.8244** | **0.8004** |
| `0.40` | 0.7969 | 0.7589 | 0.7774 |
| `0.35` | 0.8181 | 0.6701 | 0.7367 |

A threshold of **`0.45`** was selected to increase default detection recall while maintaining over 82% precision.

---

## Final Evaluation (Untouched Test Set)

The tuned Random Forest was retrained on the combined training + validation data (`25,932` rows) and evaluated once on the held-out test set (`6,484` rows) at threshold `0.45`:

* **Test ROC-AUC:** `0.9386`
* **Test PR-AUC:** `0.8901`
* **Overall Accuracy:** `91%`
* **Default Class (`1`) Performance:** Precision `0.81` | Recall `0.77` | F1-Score `0.79`

**Confusion Matrix:**

| | Predicted Non-Default (`0`) | Predicted Default (`1`) |
| :--- | :---: | :---: |
| **Actual Non-Default (`0`)** | 4,811 (TN) | 255 (FP) |
| **Actual Default (`1`)** | 324 (FN) | 1,094 (TP) |

---

## Top Feature Importances

The model's predictions are strongly aligned with financial domain intuition—with both engineered features ranking in the top 6 predictors:

1. `loan_percent_income`
2. `income_buffer` *(engineered)*
3. `loan_int_rate`
4. `person_income`
5. `loan_grade=D`
6. `interest_burden` *(engineered)*
7. `person_home_ownership=RENT`
8. `loan_amnt`

---

## Project Structure

```text
loan-default-classifier/
├── Notebook.ipynb                  # Full EDA, preprocessing, training, tuning & evaluation
├── credit_risk_dataset.csv         # Raw dataset
├── credit_risk_rf_model.joblib     # Serialized model bundle (imputer, scaler, vectorizer, model, threshold)
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation