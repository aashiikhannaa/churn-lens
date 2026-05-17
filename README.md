# 🔭 Churn Lens — Customer Churn Prediction

> An end-to-end machine learning project that predicts customer churn for a telecom company, with a live explainable AI web app built in Streamlit.

**[🚀 Live App](https://churn-lens-app.streamlit.app)** · **[📓 Notebook](telcoCustomerChurn.ipynb)**

---

## Overview

Customer churn — when a customer cancels their subscription — is one of the most costly problems in subscription-based businesses. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a machine learning pipeline that predicts which customers are likely to churn, explains *why* using SHAP, and delivers those insights through a deployed interactive web application.

The app allows a non-technical business user to enter a customer's details and instantly receive:
- A **churn probability score**
- A **risk level** (High / Medium / Low) with a recommended action
- A **SHAP feature explanation** showing exactly which factors drove the prediction

---

## Live Demo

🌐 **[churn-lens-app.streamlit.app](https://churn-lens-app.streamlit.app)**

The app has three sections:
- **Predict** — enter customer details and get a real-time churn prediction with SHAP explanation
- **Insights** — EDA and model evaluation plots with plain-language business interpretations
- **Learn** — theory and intuition behind every decision in the project, written for non-technical readers

---

## Dataset

**Telco Customer Churn** — publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Rows | 7,032 customers |
| Features | 20 raw → 23 after encoding |
| Target | `Churn` (Yes / No) |
| Class balance | 26.5% churned, 73.5% retained |

---

## Project Structure

```
churn-lens/
├── app.py                        # Streamlit web application
├── telcoCustomerChurn.ipynb      # Full analysis notebook
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version pin (3.11)
│
├── model.pkl                     # Trained Logistic Regression model
├── scaler.pkl                    # Fitted StandardScaler
├── columns.pkl                   # Training column order
│
├── shap_bar.png                  # Global SHAP feature importance
├── shap_beeswarm.png             # SHAP beeswarm plot
├── confusion_matrix.png          # Confusion matrix
├── roc_curve.png                 # ROC curve
├── precision_recall_curve.png    # Precision-Recall curve
├── churn_by_contract.png         # Churn rate by contract type
├── churn_by_tenure.png           # Churn rate by tenure group
├── monthly_charges_dist.png      # Monthly charges distribution
│
└── .streamlit/
    └── config.toml               # Streamlit dark theme config
```

---

## ML Pipeline

### 1. Data Cleaning
- Converted `TotalCharges` from object to numeric (`pd.to_numeric`, `errors='coerce'`)
- Dropped 11 rows with null `TotalCharges` (0.15% of data — negligible)
- Dropped `customerID` — unique identifier, zero predictive signal

### 2. Encoding
- **Label encoded** binary columns (Yes/No, Male/Female) → 0/1
- **Collapsed** "No internet service" / "No phone service" → "No" across 7 service columns
- **One-hot encoded** multi-category columns: `Contract`, `InternetService`, `PaymentMethod` (with `drop_first=True` to avoid multicollinearity)

### 3. Train-Test Split
- 80/20 split with `stratify=y` — preserves the 26.5%/73.5% churn ratio in both halves
- Split performed **before** scaling to prevent data leakage

### 4. Scaling
- `StandardScaler` fitted on `X_train` only, applied to both `X_train` and `X_test`
- Applied to continuous columns: `tenure`, `MonthlyCharges`, `TotalCharges`
- Scaler saved as `scaler.pkl` for consistent transformation in the deployed app

### 5. Class Imbalance — SMOTE
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to training data only
- Balanced training set to 50/50 churn ratio
- Test set left untouched to reflect real-world distribution

---

## Model Comparison

Four models were trained and evaluated. The primary metrics are **AUC-ROC** and **Recall** — chosen because class imbalance makes accuracy misleading, and missing a churner is far more costly than a false alarm.

| Model | Accuracy | Precision | Recall | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.7456 | 0.5148 | **0.7433** | **0.8277** |
| Random Forest | 0.7711 | 0.5602 | 0.6471 | 0.8155 |
| XGBoost (default) | 0.7655 | 0.5478 | 0.6738 | 0.8094 |
| XGBoost (tuned) | 0.7470 | 0.5181 | 0.6872 | 0.8212 |

**✅ Logistic Regression selected as the final model** — it outperformed all others on both key metrics. The dataset is small (7,032 rows) and the relationship between features and churn is largely linear, which is precisely where Logistic Regression excels. More complex models did not have enough data complexity to exploit their advantage — a clear demonstration that model selection should be evidence-driven, not assumption-driven.

> XGBoost was tuned via `GridSearchCV` (32 combinations, 3-fold CV, `scoring='roc_auc'`) before this conclusion was drawn.

---

## Why Not Neural Networks?

Neural networks require large amounts of data and are black boxes — you cannot explain why they made a specific prediction. For a business problem where a retention manager needs to act on the output (call the customer, offer a discount), **explainability is not optional**. XGBoost with SHAP gives us clear, feature-level explanations. Neural networks dominate in image recognition, NLP, and audio — not structured tabular data with ~7,000 rows.

---

## Key Findings

### SHAP Feature Importance
`MonthlyCharges` dominates every other feature by a wide margin — its average SHAP impact is more than twice that of the second most important feature (`InternetService_Fiber optic`). Contract type, despite strong EDA signal, ranks lower than expected — the financial signal overrides the commitment signal.

### Business Insights

| Finding | Implication |
|---|---|
| High monthly charges → highest churn signal | Targeted pricing intervention for high-bill, short-tenure customers has the greatest retention ROI |
| 0–12 month customers churn at the highest rate | An onboarding retention programme in the first year is the most impactful initiative |
| Fiber optic customers churn significantly more | Loyalty pricing for fiber customers directly addresses the highest-risk internet segment |
| Month-to-month contracts → ~43% churn vs ~3% for two-year | Incentivising contract upgrades — even with a discount — dramatically reduces churn exposure |
| Gender and Partner have near-zero SHAP impact | Demographic targeting is ineffective; financial and service factors drive churn |

---

## Metrics Explained

**Why not accuracy?** With 73.5% non-churners, a model predicting "no churn" every time scores 73.5% accuracy — and is useless.

**AUC-ROC (0.828)** — measures the model's ability to separate churners from non-churners across all thresholds. 0.5 = random, 1.0 = perfect.

**Recall (0.743)** — of all actual churners, 74.3% are caught. Prioritised because missing a churner (lost revenue) costs far more than a false alarm (unnecessary retention call).

**Precision (0.515)** — of all customers flagged as churners, 51.5% actually churn. Acceptable given the business cost asymmetry.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn, xgboost, imbalanced-learn |
| Explainability | shap |
| Visualisation | matplotlib |
| Web app | Streamlit |
| Deployment | Streamlit Cloud |
| Version control | Git, GitHub |

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/aashiikhannaa/churn-lens.git
cd churn-lens

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The dataset is not included in this repo. Download it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) to reproduce the notebook.

---

## Author

**Aashi Khanna** · 2026
