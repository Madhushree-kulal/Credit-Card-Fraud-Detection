# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load Dataset

df = pd.read_csv("creditcard.csv")

print("\n========== DATASET LOADED SUCCESSFULLY ==========")


# Step 3: Display First 5 Rows

print("\nFirst 5 Rows:")
print(df.head())


# Step 4: Display Dataset Shape

print("\nDataset Shape:")
print(df.shape)

print("Total Transactions:", df.shape[0])
print("Total Columns:", df.shape[1])


# Step 5: Dataset Information

print("\nDataset Information:")
df.info()


# Step 6: Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())


# Step 7: Check Duplicate Values

print("\nDuplicate Records:")
print(df.duplicated().sum())


# Step 8: Transaction Class Distribution

print("\nTransaction Classes:")
print(df["Class"].value_counts())


# Step 9: Transaction Percentage

print("\nTransaction Distribution Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)


# Step 10: Visualize Transaction Distribution

plt.figure(figsize=(7, 5))

sns.countplot(x="Class", data=df)

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.show()


# Step 11: Transaction Amount Statistics

print("\nTransaction Amount Statistics:")
print(df["Amount"].describe())


# Step 12: Transaction Amount Distribution

plt.figure(figsize=(8, 5))

sns.histplot(df["Amount"], bins=50)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# Step 13: Compare Transaction Amounts

plt.figure(figsize=(7, 5))

sns.boxplot(x="Class", y="Amount", data=df)

plt.title("Transaction Amount by Class")
plt.xlabel("Transaction Class")
plt.ylabel("Transaction Amount")

plt.tight_layout()
plt.show()


# Step 14: Correlation Analysis

plt.figure(figsize=(12, 9))

correlation = df.corr()

sns.heatmap(correlation, cmap="coolwarm", annot=False)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()


# Step 15: Fraud vs Legitimate Transaction Amount

print("\nAverage Transaction Amount by Class:")

print(df.groupby("Class")["Amount"].mean())


# Step 16: Final Summary

print("\n========== PROJECT DATA ANALYSIS SUMMARY ==========")

print("Total Transactions:", len(df))

print("Legitimate Transactions:", (df["Class"] == 0).sum())

print("Fraudulent Transactions:", (df["Class"] == 1).sum())

print("Missing Values:", df.isnull().sum().sum())

print("Duplicate Records:", df.duplicated().sum())

print("\nDATA ANALYSIS COMPLETED SUCCESSFULLY!")