# 💳 Credit Card Fraud Detection System

## 📌 Project Overview

The Credit Card Fraud Detection System is a Machine Learning-based application designed to identify potentially fraudulent credit card transactions.

It uses a Random Forest classification algorithm to analyze transaction features and predict whether a transaction is legitimate or potentially fraudulent.

The project also includes an interactive web application developed using Streamlit.

## 🎯 Objectives

- Detect potentially fraudulent credit card transactions.
- Analyze transaction data using Exploratory Data Analysis (EDA).
- Handle imbalanced datasets.
- Train and compare Machine Learning models.
- Optimize the prediction threshold.
- Build an interactive fraud detection application.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## 📂 Dataset

Dataset: Credit Card Fraud Detection

Original dataset contains:
- 284,807 transactions.
- 31 columns.
- 492 fraudulent transactions.
- 284,315 legitimate transactions.

Target column:
- 0 – Legitimate transaction
- 1 – Fraudulent transaction

The dataset contains anonymized PCA-transformed features (V1–V28), along with transaction time and amount.

## ⚙️ Project Workflow

1. Dataset Collection
2. Exploratory Data Analysis
3. Data Cleaning and Preprocessing
4. Handling Class Imbalance
5. Logistic Regression Model Training
6. Random Forest Model Training
7. Model Evaluation and Comparison
8. Threshold Optimization
9. Streamlit Application Development

## 🤖 Machine Learning Model

Random Forest Classifier was selected for the final application.

The decision threshold was optimized using validation data to select a threshold that maximized the F1-score among the tested thresholds.

Selected threshold: 0.55

## 📊 Model Performance

Final optimized model test results:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 99.94% |
| Precision | 94.37% |
| Recall | 70.53% |
| F1-Score | 80.72% |
| ROC-AUC | 95.14% |
| PR-AUC | 80.21% |

These results are from the held-out test dataset.

## 🌐 Streamlit Application

The application allows users to:

- Select a legitimate or fraudulent sample transaction.
- Generate a sample from the dataset.
- View the predicted fraud probability.
- View the model's classification.
- Compare the prediction with the actual dataset label.

Note: The application currently demonstrates predictions using sample transactions from the dataset. It is not connected to a live banking system.

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd Credit_Card_Fraud_Detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Download `creditcard.csv` from Kaggle and place it inside the project folder.

### 5. Run the Application

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── app.py
├── data_analysis.py
├── model_training.py
├── model_comparision.py
├── model_optimization.py
├── requirements.txt
├── fraud_detection_bundle.pkl
├── random_forest_model.pkl
├── fraud_detection_model.pkl
├── creditcard.csv
└── README.md
```

## 🚀 Future Enhancements

- Integrate a database using SQL.
- Develop an interactive Power BI dashboard.
- Improve fraud detection recall.
- Integrate real-time transaction processing.
- Deploy the application online.

## 👩‍💻 Developed By

Madhushree Kulal

BE – Artificial Intelligence and Machine Learning