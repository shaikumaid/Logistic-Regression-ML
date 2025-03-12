import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and scaler
model_path = "logistic_regression_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

# Custom CSS for better styling
st.markdown("""
    <style>
        .center-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .stButton>button {
            width: 50%;
            margin: auto;
            display: block;
            font-size: 18px;
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            width: 60% !important;
            margin: auto;
        }
        .prediction-container {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        .center-input {
            text-align: center;
            width: 100%;
        }
        .center-input > div {
            width: 60%;
            margin: auto;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("🚢 Titanic Survival Prediction")

# Centered Input Form
st.markdown("<div class='center-container'>", unsafe_allow_html=True)

pclass = st.selectbox("🎟️ Passenger Class", [1, 2, 3], key='pclass', format_func=lambda x: f"Class {x}")
age = st.number_input("🎂 Age", min_value=1, max_value=100, value=30, key='age')
sibsp = st.number_input("👨‍👩‍👦 Siblings/Spouses Aboard", min_value=0, max_value=10, value=0, key='sibsp')
parch = st.number_input("👶 Parents/Children Aboard", min_value=0, max_value=10, value=0, key='parch')
fare = st.number_input("💰 Fare Amount", min_value=0.0, max_value=500.0, value=50.0, key='fare')
sex = st.selectbox("⚧️ Gender", ["Male", "Female"], key='sex')
embarked = st.selectbox("🚢 Embarked Port", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"], key='embarked')

st.markdown("</div>", unsafe_allow_html=True)

# Encode Categorical Variables
sex_encoded = 1 if sex == "Male" else 0
embarked_mapping = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
embarked_encoded = embarked_mapping[embarked]

# Create Input DataFrame
input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex_encoded, embarked_encoded]],
                          columns=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"])

# Apply StandardScaler
input_data_scaled = scaler.transform(input_data)

# Centering the Predict Button
st.markdown("<div class='center-container'>", unsafe_allow_html=True)
if st.button("🔮 Predict"):
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    # Display Result
    st.markdown("<div class='prediction-container'>", unsafe_allow_html=True)
    if prediction == 1:
        st.success(f"🟢 Survived! (Probability: {round(probability * 100, 2)}%)")
        st.markdown("### 🎉 You would have made it! Stay lucky! 🍀")
    else:
        st.error(f"🔴 Did not survive. (Probability: {round((1 - probability) * 100, 2)}%)")
        st.markdown("### 😢 Unfortunately, you wouldn't have made it. Better luck next time!")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
