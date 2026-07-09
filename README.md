# Yes Bank Stock Price Prediction

An end-to-end machine learning regression framework built to predict monthly equity closing prices for Yes Bank. This project implements advanced feature transformations to overcome severe multi-feature correlation and delivers an overfit-protected, production-ready predictive pipeline.

## 📌 Project Overview
The financial performance of Yes Bank over the past decade reflects extreme volatility, driven by structural shifts and regulatory events. This project develops a highly stable machine learning workflow that extracts clean asset pricing trends while actively minimizing variance and mathematical weight explosion.

* **Project Type:** Machine Learning Regression
* **Core Task:** Continuous Numerical Forecasting (`Close` Price)
* **Dataset Source:** Historical Monthly Yes Bank Stock Metrics (Open, High, Low, Close)

---

## 🛠️ Tech Stack & Dependencies
The notebook is built to be entirely self-contained and executable in a single run.
* **Language:** Python 3
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## ⚙️ Core Pipeline Architecture

### 1. Exploratory Data Analysis & Feature Transformation
* **Distribution Balancing:** Applied a natural log transformation ($y = \ln(x)$) across all absolute pricing features to clean up severe right-skewness and normalize outlier weights.
* **Multicollinearity Fix:** Raw features (`Open`, `High`, `Low`) exhibit a near-perfect linear correlation of $1.00$. To eliminate structural redundancy, a custom relative volatility metric (`Log_Spread`) was engineered:
  $$\text{Log\_Spread} = \ln(\text{High}) - \ln(\text{Low})$$

### 2. Regularized Modeling & Hyperparameter Tuning
To handle the highly collinear feature space safely, three cross-validated regularized architectures were deployed using `GridSearchCV` (5-Fold CV):
* **Ridge Regression (L2)**
* **Lasso Regression (L1)**
* **ElasticNet Regression (Hybrid)**

---

## 📊 Model Evaluation Summary

| Model Architecture | Tuned Parameters | Test $R^2$ Score | Final MAE | Final RMSE |
| :--- | :--- | :--- | :--- | :--- |
| Ridge Regression | $\alpha = 0.1$ | $0.95947$ | $0.13685$ | $0.17926$ |
| **Lasso Regression (Champion)** | $\alpha = 0.01$ | **$0.96538$** | **$0.12935$** | **$0.16568$** |
| ElasticNet Regression | $\alpha = 0.01, \text{l1\_ratio} = 0.8$ | $0.96457$ | $0.13070$ | $0.16760$ |

### Key Takeaway
**Tuned Lasso Regression** was selected as the champion model. By enforcing L1 regularization, it safely dropped unhelpful noisy dimensions down to zero, capturing **96.5%** of true market variance while maintaining highly predictable, tightly bounded error ranges.

---

## 🚀 Execution Instructions
This repository features deployment-ready, production-grade code. The entire workflow can be executed sequentially without error:

1. Clone this repository locally:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Yes-Bank-Stock-Price-Prediction.git](https://github.com/YOUR_USERNAME/Yes-Bank-Stock-Price-Prediction.git)
