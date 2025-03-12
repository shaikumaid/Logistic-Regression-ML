import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, log_loss, roc_curve

st.title('Logistic Regression Analysis')

# 🔹 Read CSV files directly (No need for file upload)
train_file_path = "data/train.csv"  # Update with actual path
test_file_path = "data/test.csv"    # Update with actual path

df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

st.write("Training Data Sample:", df_train.head())
st.write("Testing Data Sample:", df_test.head())

# 🔹 Concatenating Train & Test Data
df = pd.concat([df_train, df_test], ignore_index=True)

# 🔹 Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Cabin'].fillna("C23 C25 C27", inplace=True)
df['Embarked'].fillna("S", inplace=True)

# 🔹 Standardization
scaler = StandardScaler()
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 🔹 Label Encoding for Categorical Columns
encoder = LabelEncoder()
categorical_cols = ["Ticket", "Sex", "Cabin", "Embarked"]
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# 🔹 Preparing Data for Model
df = df.dropna(subset=['Survived'])  # Ensure no NaN values in target
X = df.select_dtypes(include=[np.number]).drop(columns=['Survived'])  # Features
Y = df['Survived'].astype(int)  # Target variable

# 🔹 Model Training
model = LogisticRegression()
if X.isnull().sum().sum() > 0:
    st.write("⚠️ Warning: Missing values found in X. Filling with median.")
    X.fillna(X.median(), inplace=True)

model.fit(X, Y)
Y_pred = model.predict(X)

# 🔹 Model Evaluation
if len(Y) > 0 and len(Y_pred) > 0:
    st.write("✅ Sensitivity score:", round(recall_score(Y, Y_pred), 2))
    st.write("✅ Accuracy Score:", round(accuracy_score(Y, Y_pred), 2))
    st.write("✅ Precision Score:", round(precision_score(Y, Y_pred), 2))
    st.write("✅ F1 Score:", round(f1_score(Y, Y_pred), 2))
    st.write("✅ Log Loss:", round(log_loss(Y, model.predict_proba(X)), 2))
    
    # 🔹 ROC Curve
    fpr, tpr, _ = roc_curve(Y, model.predict_proba(X)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)
else:
    st.write("❌ Error: Y or Y_pred is empty or not properly defined.")
