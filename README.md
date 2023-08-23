# credit_card_fraud_detection
# Credit Card Fraud Detection using Machine Learning

**Author:** Malick Hamidou .B

## Introduction

Credit card fraud is a significant concern for both consumers and financial institutions. Detecting fraudulent transactions is crucial to protect customers from unauthorized charges. This project aims to develop a machine learning model that can effectively identify fraudulent credit card transactions using the "Credit Card Fraud Detection" dataset available on Kaggle.

## Dataset Overview

The dataset used for this project is the "Credit Card Fraud Detection" dataset available on Kaggle. It contains credit card transactions made by European cardholders. The dataset consists of 284,807 transactions, out of which 492 are fraudulent. The data contains only numerical input variables resulting from Principal Component Analysis (PCA) transformations due to confidentiality issues. The features include 'Time', 'Amount', and 'V1' through 'V28', as well as the 'Class' variable, which is the target variable indicating whether the transaction is fraudulent (1) or not (0).

## Data Understanding and Preprocessing

### Importing Libraries and Dataset

```python
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# Importing the dataset
dataset = pd.read_csv('/path/to/creditcard.csv')
```

### Data Overview

```python
print('Dataset Shape:', dataset.shape)
print(dataset.head())
print(dataset.info())
print(dataset.nunique())
print(dataset.isnull().sum())
print(dataset[['Time', 'Amount']].describe())
```

### Data Visualization

Visualizing the distribution of the 'Time' and 'Amount' features, understanding transaction patterns, and handling the class imbalance.

### Feature Engineering

Handling correlations and standardizing features using StandardScaler.

## Modeling

### Baseline Model - Logistic Regression with SMOTE

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Separate features and target
X = dataset.drop(['Class', 'Amount_log'], axis=1)
y = dataset['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Baseline: Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_resampled, y_train_resampled)
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_lr))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_lr))
```

### Advanced Machine Learning Models

Exploring advanced models like Random Forest, Gradient Boosting, Support Vector Machines, and Neural Networks.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_resampled, y_train_resampled)

# Support Vector Machine
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_resampled, y_train_resampled)

# Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
nn_model.fit(X_train_resampled, y_train_resampled)
```

### Model Evaluation and Interpretation

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print("Cross-Validation AUC-ROC Scores:", cv_scores)
    
    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Call the function for each model
evaluate_model(rf_model, X_train_resampled, y_train_resampled, X_test, y_test, "Random Forest")
evaluate_model(gb_model, X_train_resampled, y_train_resampled, X_test, y_test, "Gradient Boosting")
evaluate_model(svm_model, X_train_resampled, y_train_resampled, X_test, y_test

, "Support Vector Machine")
evaluate_model(nn_model, X_train_resampled, y_train_resampled, X_test, y_test, "Neural Network")
```

### Model Comparison

```python
from sklearn.metrics import roc_auc_score

# Calculate AUC-ROC score on test data for each model
test_scores = {
    'Random Forest': roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]),
    'Gradient Boosting': roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1]),
    'Support Vector Machine': roc_auc_score(y_test, svm_model.decision_function(X_test)),
    'Neural Network': roc_auc_score(y_test, nn_model.predict_proba(X_test)[:, 1])
}

print("Test AUC-ROC Scores:")
for model_name, score in test_scores.items():
    print(f"{model_name}: {score}")
```

## Conclusion

This project focused on detecting credit card fraud using machine learning techniques. We explored data preprocessing, visualization, and various advanced models to identify fraudulent transactions. The models were evaluated using cross-validation, confusion matrices, and AUC-ROC scores. The Gradient Boosting model achieved the highest AUC-ROC score, making it a potential choice for deploying a fraud detection system.
