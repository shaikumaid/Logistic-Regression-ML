import streamlit as st
import pandas as pd
import numpy as np
import pickle  # To load the saved model
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title('Titanic Survival Prediction 🚢')

# 🔹 Load Pre-trained Model
model_path = "logistic_regression_model.pkl"  # Update with actual model path
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found! Please train and save the model first.")
    st.stop()

# 🔹 Standard Scaler (Ensures input data is transformed correctly)
scaler = StandardScaler()

# 🔹 User Input Section
st.header("Enter Passenger Details:")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=500.0, value=30.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# 🔹 Encode Categorical Data
sex_encoded = 1 if sex == "male" else 0
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_mapping[embarked]

# 🔹 Prepare Data for Prediction
user_input = np.array([[pclass, age, sibsp, parch, fare, sex_encoded, embarked_encoded]])
user_input_scaled = scaler.fit_transform(user_input)  # Standardization

# 🔹 Make Prediction
if st.button("Predict Survival"):
    prediction = model.predict(user_input_scaled)
    survival_prob = model.predict_proba(user_input_scaled)[0][1]

    if prediction[0] == 1:
        st.success(f"🟢 Survived! (Probability: {round(survival_prob * 100, 2)}%)")
    else:
        st.error(f"🔴 Did not survive. (Probability: {round((1 - survival_prob) * 100, 2)}%)")
