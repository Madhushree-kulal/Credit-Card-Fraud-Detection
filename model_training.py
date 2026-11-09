# ==========================================
# CREDIT CARD FRAUD DETECTION SYSTEM
# PHASE 3: DATA PREPROCESSING
# PHASE 4: MACHINE LEARNING
# ==========================================

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


# Step 2: Load Dataset

df = pd.read_csv("creditcard.csv")

print("\n========== DATASET LOADED ==========")

print("Original Dataset Shape:", df.shape)


# Step 3: Remove Duplicate Records

print("\nDuplicate Records:", df.duplicated().sum())

df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates:", df.shape)


# Step 4: Separate Features and Target

X = df.drop("Class", axis=1)

y = df["Class"]

print("\nFeatures Shape:", X.shape)

print("Target Shape:", y.shape)


# Step 5: Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)

print("Testing Data:", X_test.shape)


# Step 6: Create Machine Learning Pipeline

model = Pipeline([
    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])


# Step 7: Train Model

print("\n========== MODEL TRAINING STARTED ==========")

model.fit(X_train, y_train)

print("Model Training Completed Successfully!")


# Step 8: Make Predictions

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# Step 9: Evaluate Model

print("\n========== MODEL EVALUATION ==========")

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, zero_division=0)

recall = recall_score(y_test, y_pred, zero_division=0)

f1 = f1_score(y_test, y_pred, zero_division=0)

roc_auc = roc_auc_score(y_test, y_prob)

pr_auc = average_precision_score(y_test, y_prob)


print("Accuracy:", round(accuracy * 100, 2), "%")

print("Precision:", round(precision, 4))

print("Recall:", round(recall, 4))

print("F1 Score:", round(f1, 4))

print("ROC-AUC Score:", round(roc_auc, 4))

print("PR-AUC Score:", round(pr_auc, 4))


# Step 10: Classification Report

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# Step 11: Confusion Matrix

print("\nConfusion Matrix:")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# Step 12: Save Trained Model

joblib.dump(model, "fraud_detection_model.pkl")

print("\nModel Saved Successfully!")

print("\n========== PROJECT COMPLETED ==========")
