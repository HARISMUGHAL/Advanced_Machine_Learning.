import streamlit as st
import pandas as pd
import joblib

st.title("Telco Customer Churn Predictor")

# Load the trained model
model = joblib.load("telco_churn_pipeline.pkl")

# Input form for user to provide customer data
gender = st.selectbox("Gender", ['Male', 'Female'])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ['Yes', 'No'])
Dependents = st.selectbox("Dependents", ['Yes', 'No'])
tenure = st.number_input("Tenure (in months)", min_value=0)
PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
PaymentMethod = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")
