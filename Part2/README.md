# Part 2 – Supervised Machine Learning: Build, Train and Evaluate





---

# Objective

The objective of this part is to build and evaluate both regression and classification machine learning models using the cleaned insurance dataset obtained from Part 1.

The tasks performed include:

- Data preprocessing
- Feature encoding
- Train-test split
- Feature scaling
- Linear Regression
- Ridge Regression
- Logistic Regression
- ROC-AUC Evaluation
- Threshold Analysis
- Logistic Regression Regularization
- Bootstrap Confidence Interval

---

# Dataset

The cleaned dataset generated in Part 1 (`cleaned_data.csv`) was used.

## Regression Target

**charges**

Medical insurance charges were selected as the continuous target variable.

## Classification Target

A binary classification target was created using the median insurance charge.

```python
y_clf = (charges > charges.median()).astype(int)
```

Class Labels

- 0 → Low Insurance Charges
- 1 → High Insurance Charges

---

# Feature Encoding

The dataset contained three categorical variables:

- sex
- smoker
- region

Since none of these variables have any natural ordering, One-Hot Encoding was applied.

```python
pd.get_dummies(drop_first=True)
```

### Why One-Hot Encoding?

One-Hot Encoding prevents the model from assuming an artificial numerical relationship between categories.

For example,

```
Male ≠ Female
```

Neither category is greater than the other.

Using Label Encoding would incorrectly introduce ordinal information.

---

# Train-Test Split

The dataset was divided into

- 80% Training Data
- 20% Testing Data

using

```python
train_test_split(test_size=0.2, random_state=42)
```

---

# Feature Scaling

StandardScaler was applied.

The scaler was fitted **only on the training dataset** and then applied to the testing dataset.

### Why?

Fitting the scaler on the complete dataset would introduce **data leakage**, since statistics from the test set would influence the training process and produce overly optimistic performance estimates.

---

# Regression Models

## 1. Linear Regression

The Linear Regression model was trained using the scaled training data.

### Results

| Metric | Value |
|--------|------:|
| Mean Squared Error (MSE) | **35,478,020.68** |
| R² Score | **0.8069** |

### Interpretation

The model explains approximately **80.69%** of the variation in insurance charges.

---

## Feature Importance

Top three features based on absolute coefficient magnitude:

| Rank | Feature | Coefficient |
|------|----------|------------:|
| 1 | smoker_yes | 9234.34 |
| 2 | age | 3472.98 |
| 3 | bmi | 1927.83 |

### Interpretation

A positive coefficient indicates that increasing the feature increases predicted insurance charges.

A negative coefficient indicates that increasing the feature decreases predicted insurance charges.

The large positive coefficient for **smoker_yes** indicates that smoking status has the strongest influence on insurance cost.

---

## 2. Ridge Regression

Ridge Regression was trained using

```python
Ridge(alpha=1.0)
```

### Comparison

| Model | MSE | R² Score |
|------|------:|------:|
| Linear Regression | *(Your value)* | *(Your value)* |
| Ridge Regression | *(Your Ridge MSE)* | *(Your Ridge R²)* |

### Why Ridge Regression?

Ridge Regression applies **L2 regularization**, which penalizes excessively large coefficient values.

The parameter **alpha** controls the strength of regularization.

Higher alpha values produce stronger coefficient shrinkage and reduce overfitting.

---

# Classification Model

## Logistic Regression

The Logistic Regression classifier was trained using

```python
LogisticRegression(max_iter=1000)
```

---

# Class Imbalance

Training class distribution:

| Class | Percentage |
|------|-----------:|
| Low Charges | 50.05% |
| High Charges | 49.95% |

The classes were nearly perfectly balanced.

Therefore, no balancing technique such as SMOTE was required.

---

# Classification Performance

| Metric | Value |
|--------|------:|
| Accuracy | **90.67%** |
| Precision | **89.78%** |
| Recall | **91.79%** |
| F1 Score | **90.77%** |

The classifier demonstrates strong predictive performance across both classes.

---

# Precision and Recall

Precision

```
TP / (TP + FP)
```

Precision measures how many predicted positive samples are actually positive.

Recall

```
TP / (TP + FN)
```

Recall measures how many actual positive samples are correctly identified.

For this insurance classification problem, Recall is important because failing to identify genuinely high-cost customers could affect future insurance pricing decisions.

---

# ROC Curve

The Receiver Operating Characteristic (ROC) curve was generated.

### AUC Score

**0.9529**

### Interpretation

An AUC of approximately **0.95** indicates excellent class separation capability.

The classifier is highly effective at distinguishing between low-cost and high-cost insurance customers.

---

# Decision Threshold Analysis

Thresholds between 0.30 and 0.70 were evaluated.

| Threshold | Precision | Recall | F1 Score |
|-----------|----------:|-------:|---------:|
| 0.30 | 0.779 | 0.948 | 0.855 |
| 0.40 | 0.855 | 0.925 | 0.889 |
| 0.50 | 0.898 | 0.918 | 0.908 |
| 0.60 | 0.960 | 0.903 | **0.931** |
| 0.70 | 1.000 | 0.843 | 0.915 |

### Best Threshold

**0.60**

This threshold produced the highest F1-score.

Increasing the decision threshold improves Precision but reduces Recall.

Lowering the threshold improves Recall but increases False Positives.

---

# Logistic Regression Regularization

Two Logistic Regression models were compared.

| Model | Precision | Recall | AUC |
|-------|----------:|-------:|----:|
| Baseline (C=1.0) | 0.8978 | 0.9179 | 0.9529 |
| Regularized (C=0.01) | 0.8841 | 0.9104 | 0.9511 |

### Interpretation

The baseline model slightly outperformed the strongly regularized model.

Reducing C increases regularization strength and shrinks the coefficients.

For this dataset, stronger regularization resulted in a slight decrease in performance.

---

# Bootstrap Confidence Interval

500 bootstrap samples were generated.

### Results

Mean AUC Difference

```
0.00173
```

95% Confidence Interval

```
Lower = -0.00070

Upper = 0.00470
```

### Interpretation

Since the confidence interval includes zero, the observed performance difference between the baseline and regularized models is not statistically reliable.

---

# Files Included

- Part2.ipynb
- cleaned_data.csv
- README.md
- requirements.txt

---

# Conclusion

Both regression and classification models performed well on the insurance dataset.

Key observations include:

- Linear Regression explained over 80% of the variance in insurance charges.
- Smoking status was the strongest predictor of insurance cost.
- Logistic Regression achieved an accuracy of approximately 91%.
- The classifier achieved an excellent ROC-AUC score of 0.95.
- A threshold of 0.60 produced the highest F1-score.
- Strong regularization did not improve performance.
- Bootstrap analysis indicated that the difference between the baseline and regularized models was not statistically significant.

The trained models provide a strong foundation for the ensemble learning techniques implemented in Part 3.
