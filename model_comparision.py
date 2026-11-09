# ==========================================
# CREDIT CARD FRAUD DETECTION
# RANDOM FOREST MODEL
# ==========================================

# Step 1: Import Libraries

import pandas as pd
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


# Step 5: Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


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

print("\nRandom Forest Training Started...")

model.fit(X_train, y_train)

print("Training Completed Successfully!")


# Step 8: Make Predictions

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# Step 9: Model Evaluation

print("\n========== RANDOM FOREST RESULTS ==========")

print("Accuracy:",
      round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("Precision:",
      round(precision_score(y_test, y_pred, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_test, y_pred, zero_division=0), 4))

print("F1 Score:",
      round(f1_score(y_test, y_pred, zero_division=0), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, y_prob), 4))

print("PR-AUC:",
      round(average_precision_score(y_test, y_prob), 4))


# Step 10: Confusion Matrix

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))


# Step 11: Classification Report

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# Step 12: Save Model

joblib.dump(model, "random_forest_model.pkl")

print("\nRandom Forest Model Saved Successfully!")