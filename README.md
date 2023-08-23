
# Credit Card Fraud Detection using Machine Learning

## Introduction

Credit card fraud is a significant concern that impacts both consumers and financial institutions. Detecting fraudulent transactions is crucial for protecting customers from unauthorized charges. This project aims to build a machine learning model that can effectively identify fraudulent credit card transactions using the "Credit Card Fraud Detection" dataset available on Kaggle.

## Dataset Overview

The dataset contains credit card transactions made by European cardholders. It consists of 284,807 transactions, out of which 492 are fraudulent. The dataset includes numerical features derived from Principal Component Analysis (PCA) transformations to address confidentiality issues. Key features include 'Time', 'Amount', and transformed features 'V1' through 'V28'. The 'Class' variable indicates whether a transaction is fraudulent (1) or not (0).

## Data Understanding and Preprocessing

- Imported necessary libraries: NumPy, pandas, Matplotlib, Seaborn.
- Loaded the dataset and checked its shape and basic structure.
- Explored key statistics and distributions of 'Time' and 'Amount' features.
- Investigated data correlations and applied standardization using StandardScaler.
- Handled class imbalance using SMOTE technique for oversampling.

## Modeling and Evaluation

- Created a baseline Logistic Regression model with SMOTE to address class imbalance.
- Explored advanced machine learning models: Random Forest, Gradient Boosting, Support Vector Machine, and Neural Network.
- Evaluated models using cross-validation, confusion matrices, and AUC-ROC scores.
- Compared model performances and identified the Gradient Boosting model as the top performer based on AUC-ROC scores.

## Conclusion

This project demonstrated the process of developing a credit card fraud detection system using machine learning. By preprocessing the data, exploring different models, and evaluating their performances, we identified the Gradient Boosting model as the most suitable for this task. The project's findings highlight the potential of machine learning in identifying fraudulent transactions and enhancing security in financial transactions.
