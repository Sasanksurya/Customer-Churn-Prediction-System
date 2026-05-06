import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/best_rf_model.pkl")

# App title
st.title("Customer Churn Prediction System")

st.write(
    "Predict whether a customer is likely to churn."
)

# User inputs
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure Months",
    1,
    72
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Create dataframe
input_data = pd.DataFrame({
    "Gender": [gender],
    "Senior Citizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "Tenure Months": [tenure],
    "Monthly Charges": [monthly_charges],
    "Contract": [contract],
    "Internet Service": [internet_service],
    "Payment Method": [payment_method]
})

# Prediction button
if st.button("Predict Churn"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:

        st.error(
            f"Customer is likely to churn.\nProbability: {probability:.2f}"
        )

    else:

        st.success(
            f"Customer is likely to stay.\nProbability: {probability:.2f}"
        )