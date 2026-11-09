import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    bundle = joblib.load("fraud_detection_bundle.pkl")
    return bundle["model"], bundle["threshold"]

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    df = df.drop_duplicates()
    return df

model, threshold = load_model()
df = load_data()

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "An AI-powered application to identify "
    "potentially fraudulent credit card transactions."
)

st.sidebar.title("Project Information")
st.sidebar.write("Machine Learning Model: Random Forest")
st.sidebar.write(f"Decision Threshold: {threshold}")
st.sidebar.write("Dataset: Credit Card Transactions")

st.divider()

st.subheader("Test a Sample Transaction")

transaction_type = st.selectbox(
    "Choose Transaction Type",
    ["Legitimate Transaction", "Fraudulent Transaction"]
)

if st.button("Generate Sample Transaction", type="primary"):

    if transaction_type == "Legitimate Transaction":
        sample = df[df["Class"] == 0].sample(1)
    else:
        sample = df[df["Class"] == 1].sample(1)

    actual_class = int(sample["Class"].iloc[0])

    feature_columns = [
        "Time"
    ] + [
        f"V{i}" for i in range(1, 29)
    ] + [
        "Amount"
    ]

    input_data = sample[feature_columns]

    probability = model.predict_proba(input_data)[0][1]

    prediction = 1 if probability >= threshold else 0

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )

    if prediction == 1:
        st.error("⚠️ Potentially Fraudulent Transaction")
    else:
        st.success("✅ Transaction Classified as Legitimate")

    st.write("Actual Dataset Class:",
             "Fraudulent" if actual_class == 1 else "Legitimate")

    st.write("Predicted Class:",
             "Fraudulent" if prediction == 1 else "Legitimate")

    if prediction == actual_class:
        st.success("Prediction matches the dataset label!")
    else:
        st.warning("Prediction does not match the dataset label.")

    with st.expander("View Transaction Details"):
        st.dataframe(input_data)

st.divider()

st.caption(
    "Developed using Python, Scikit-learn and Streamlit"
)