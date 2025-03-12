import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model_path = "logistic_regression_model.pkl"  # Update with the correct path
with open(model_path, "rb") as file:
    model = pickle.load(file)

st.title("Logistic Regression Live Prediction")

# User Inputs
st.sidebar.header("Enter Passenger Details:")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=30)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)

# Convert input to DataFrame
input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare]],
                          columns=["Pclass", "Age", "SibSp", "Parch", "Fare"])

# Standardization (if required)
# If the model was trained on scaled data, use the same scaler here
# scaler = pickle.load(open("scaler.pkl", "rb"))  # Uncomment if needed
# input_data = scaler.transform(input_data)

# Make Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display Result
    if prediction == 1:
        st.success(f"🟢 Survived! (Probability: {round(probability * 100, 2)}%)")
    else:
        st.error(f"🔴 Did not survive. (Probability: {round(probability * 100, 2)}%)")
