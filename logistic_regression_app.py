import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model_path = "logistic_regression_model.pkl"  # Update with actual path
scaler_path = "scaler.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)  # Load the same scaler used in training

# Apply custom CSS to center elements
st.markdown(
    """
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
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚢 Titanic Survival Prediction")

# Create a centered layout
st.markdown("<div class='center-container'>", unsafe_allow_html=True)

# Centered Input Fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

st.markdown("</div>", unsafe_allow_html=True)

# Encode Categorical Variables
sex_encoded = 1 if sex == "male" else 0
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_mapping[embarked]

# Create Input DataFrame with the same feature order as the training model
input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex_encoded, embarked_encoded]],
                          columns=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"])

# Apply StandardScaler (Ensuring it's the same as used in training)
input_data_scaled = scaler.transform(input_data)

# Centering the Predict Button
st.markdown("<div class='center-container'>", unsafe_allow_html=True)
if st.button("🚀 Predict"):
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    # Display Result
    if prediction == 1:
        st.success(f"🟢 Survived! (Probability: {round(probability * 100, 2)}%)")
    else:
        st.error(f"🔴 Did not survive. (Probability: {round((1 - probability) * 100, 2)}%)")

st.markdown("</div>", unsafe_allow_html=True)
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model_path = "logistic_regression_model.pkl"  # Update with actual path
scaler_path = "scaler.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)  # Load the same scaler used in training

# Apply custom CSS to center elements
st.markdown(
    """
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
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚢 Titanic Survival Prediction")

# Create a centered layout
st.markdown("<div class='center-container'>", unsafe_allow_html=True)

# Centered Input Fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

st.markdown("</div>", unsafe_allow_html=True)

# Encode Categorical Variables
sex_encoded = 1 if sex == "male" else 0
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_mapping[embarked]

# Create Input DataFrame with the same feature order as the training model
input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex_encoded, embarked_encoded]],
                          columns=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"])

# Apply StandardScaler (Ensuring it's the same as used in training)
input_data_scaled = scaler.transform(input_data)

# Centering the Predict Button
st.markdown("<div class='center-container'>", unsafe_allow_html=True)
if st.button("🚀 Predict"):
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    # Display Result
    if prediction == 1:
        st.success(f"🟢 Survived! (Probability: {round(probability * 100, 2)}%)")
    else:
        st.error(f"🔴 Did not survive. (Probability: {round((1 - probability) * 100, 2)}%)")

st.markdown("</div>", unsafe_allow_html=True)
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model_path = "logistic_regression_model.pkl"  # Update with actual path
scaler_path = "scaler.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)  # Load the same scaler used in training

# Apply custom CSS to center elements
st.markdown(
    """
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
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚢 Titanic Survival Prediction")

# Create a centered layout
st.markdown("<div class='center-container'>", unsafe_allow_html=True)

# Centered Input Fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

st.markdown("</div>", unsafe_allow_html=True)

# Encode Categorical Variables
sex_encoded = 1 if sex == "male" else 0
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_mapping[embarked]

# Create Input DataFrame with the same feature order as the training model
input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex_encoded, embarked_encoded]],
                          columns=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"])

# Apply StandardScaler (Ensuring it's the same as used in training)
input_data_scaled = scaler.transform(input_data)

# Centering the Predict Button
st.markdown("<div class='center-container'>", unsafe_allow_html=True)
if st.button("🚀 Predict"):
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    # Display Result
    if prediction == 1:
        st.success(f"🟢 Survived! (Probability: {round(probability * 100, 2)}%)")
    else:
        st.error(f"🔴 Did not survive. (Probability: {round((1 - probability) * 100, 2)}%)")

st.markdown("</div>", unsafe_allow_html=True)
