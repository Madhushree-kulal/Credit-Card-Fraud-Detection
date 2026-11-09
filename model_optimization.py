# ==========================================
# CREDIT CARD FRAUD DETECTION SYSTEM
# PHASE 5: MODEL OPTIMIZATION
# THRESHOLD TUNING
# ==========================================

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

print("\nDataset Loaded Successfully!")


# Step 3: Remove Duplicates

df = df.drop_duplicates()

print("Dataset Shape:", df.shape)


# Step 4: Separate Features and Target

X = df.drop("Class", axis=1)
y = df["Class"]


# Step 5: Split into Training, Validation and Testing

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("\nTraining Data:", X_train.shape)
print("Validation Data:", X_val.shape)
print("Testing Data:", X_test.shape)


# Step 6: Create Random Forest Model

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


# Step 7: Train Model

print("\nTraining Random Forest Model...")

model.fit(X_train, y_train)

print("Training Completed!")


# Step 8: Get Validation Probabilities

y_val_prob = model.predict_proba(X_val)[:, 1]


# Step 9: Test Different Thresholds

print("\n========== THRESHOLD COMPARISON ==========")

thresholds = np.arange(0.10, 0.91, 0.05)

best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:

    y_val_pred = (y_val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1 Score: {f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


# Step 10: Display Best Threshold

print("\n========== BEST THRESHOLD ==========")

print("Best Threshold:", best_threshold)
print("Best Validation F1 Score:", round(best_f1, 4))


# Step 11: Final Evaluation on Test Data

y_test_prob = model.predict_proba(X_test)[:, 1]

y_test_pred = (
    y_test_prob >= best_threshold
).astype(int)


print("\n========== FINAL MODEL EVALUATION ==========")

print(
    "Accuracy:",
    round(accuracy_score(y_test, y_test_pred) * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision_score(y_test, y_test_pred), 4)
)

print(
    "Recall:",
    round(recall_score(y_test, y_test_pred), 4)
)

print(
    "F1 Score:",
    round(f1_score(y_test, y_test_pred), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, y_test_prob), 4)
)

print(
    "PR-AUC:",
    round(average_precision_score(y_test, y_test_prob), 4)
)


# Step 12: Confusion Matrix

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_test_pred))


# Step 13: Classification Report

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )
)


# Step 14: Save Model and Threshold

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold
    },
    "fraud_detection_bundle.pkl"
)

print("\nOptimized Model Saved Successfully!")

print("\n========== OPTIMIZATION COMPLETED ==========")