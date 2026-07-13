# Part 3 – Advanced Modeling: Ensemble Learning, Hyperparameter Tuning and Model Pipeline

## Objective

The objective of this part is to compare multiple machine learning models, improve model performance through ensemble learning and hyperparameter tuning, and build a reusable machine learning pipeline.

---

# Dataset

The cleaned insurance dataset generated in Part 1 (cleaned_data.csv) was used.

Target Variable:

Binary Classification

```
High Charges = 1
Low Charges = 0
```

---

# Models Implemented

## 1. Decision Tree (Baseline)

A Decision Tree classifier was trained using default parameters.

### Results

Training Accuracy

100%

Testing Accuracy

90.67%

### Interpretation

The baseline Decision Tree achieved perfect training accuracy but lower testing accuracy, indicating overfitting. Decision Trees are high-variance models because they continue splitting until pure leaf nodes are formed.

---

## 2. Controlled Decision Tree

Parameters

```
max_depth=5
min_samples_split=20
```

### Results

Training Accuracy

92.89%

Testing Accuracy

93.66%

### Interpretation

Limiting tree depth reduced overfitting and improved testing accuracy.

---

# Gini vs Entropy

Two Decision Trees were compared.

| Criterion | Test Accuracy |
|-----------|--------------:|
| Gini | 94.03% |
| Entropy | 92.54% |

### Gini Formula

```
1 − Σ(pi²)
```

### Entropy Formula

```
− Σ pi log₂(pi)
```

A Gini value of zero indicates a completely pure node.

---

# Random Forest

Parameters

```
n_estimators = 100

max_depth = 10

random_state = 42
```

### Results

Training Accuracy

96.35%

Testing Accuracy

94.78%

ROC-AUC

0.9487

---

# Top Five Feature Importances

| Feature | Importance |
|----------|-----------:|
| age | 0.5161 |
| smoker_yes | 0.2938 |
| bmi | 0.1106 |
| children | 0.0412 |
| sex_male | 0.0139 |

Random Forest computes feature importance using the average reduction in Gini impurity across all trees.

Unlike linear regression coefficients, feature importance does not indicate direction; it indicates contribution to prediction.

---

# Bagging

Random Forest uses Bootstrap Aggregating (Bagging).

Each tree is trained using a bootstrap sample.

Each split considers a random subset of features.

Final predictions are obtained by combining predictions from all trees.

This reduces variance and improves generalization.

---

# Gradient Boosting

Parameters

```
n_estimators = 100

learning_rate = 0.1

max_depth = 3
```

### Results

Training Accuracy

95.60%

Testing Accuracy

93.28%

ROC-AUC

0.9502

Gradient Boosting builds trees sequentially, where each tree corrects the mistakes of the previous trees.

---

# Feature Ablation Study

The five least important features were removed and another Random Forest model was trained.

| Model | ROC-AUC |
|------|---------:|
| Full Random Forest | 0.9487 |
| Reduced Random Forest | 0.9396 |

Removing the least important features slightly reduced ROC-AUC, indicating that these features still contributed useful predictive information.

---

# Cross Validation

5-fold Stratified Cross Validation was performed.

| Model | Mean AUC | Std AUC |
|------|---------:|--------:|
| Logistic Regression | 0.9473 | 0.0149 |
| Controlled Decision Tree | 0.9342 | 0.0116 |
| Random Forest | 0.9495 | 0.0136 |
| Gradient Boosting | 0.9509 | 0.0105 |

Gradient Boosting achieved the highest average ROC-AUC.

Cross-validation provides a more reliable estimate of generalization performance because every sample is used for both training and validation.

---

# Hyperparameter Tuning

GridSearchCV was applied to the Random Forest pipeline.

Best Parameters

```
max_depth = 10

min_samples_leaf = 1

n_estimators = 200
```

Best Cross Validation ROC-AUC

```
0.9553
```

Total Parameter Combinations

18

Total Models Evaluated

90

Grid Search evaluates every parameter combination and finds the best performing model.

---

# Manual Learning Curve

The tuned pipeline was trained using 20%, 40%, 60%, 80% and 100% of the training data.

Training and testing ROC-AUC values were recorded.

The testing ROC-AUC improved as more training data became available, indicating that additional training data can further improve model performance.

---

# Model Serialization

The best pipeline was saved using

```python
joblib.dump(best_pipeline,"best_model.pkl")
```

The saved model was successfully reloaded using

```python
joblib.load("best_model.pkl")
```

Predictions on new samples were successfully generated.

---

# Files Included

- Part3.ipynb
- best_model.pkl
- cleaned_data.csv
- README.md
- requirements.txt

---

# Final Recommendation

Among all evaluated models, the tuned Random Forest pipeline and Gradient Boosting classifier achieved the strongest performance.

The tuned Random Forest model was selected as the final production model because of its excellent ROC-AUC, robustness, and ability to be serialized and deployed easily.
