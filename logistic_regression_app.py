import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, precision_score, f1_score, roc_curve, roc_auc_score, log_loss

st.title('Logistic Regression Analysis')

# File Uploading
uploaded_train = st.file_uploader("Upload Training CSV", type=["csv"])
uploaded_test = st.file_uploader("Upload Testing CSV", type=["csv"])

if uploaded_train and uploaded_test:
    df_train = pd.read_csv(uploaded_train)
    df_test = pd.read_csv(uploaded_test)

    st.write("Training Data Sample:", df_train.head())
    st.write("Testing Data Sample:", df_test.head())

    # Data Preprocessing
    df = pd.concat([df_train, df_test], ignore_index=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Cabin'].fillna("Unknown", inplace=True)  # Prevents information leakage
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Feature Scaling
    SS = StandardScaler()
    numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    df[numerical_cols] = SS.fit_transform(df[numerical_cols])

    # Label Encoding
    LE = LabelEncoder()
    categorical_cols = ["Sex", "Embarked"]  # Removed 'Ticket' and 'Cabin' to avoid unique identifiers
    for col in categorical_cols:
        df[col] = LE.fit_transform(df[col])

    # Ensure 'Survived' is present
    df = df.dropna(subset=['Survived'])

    # Defining Features and Target
    X = df.drop(columns=['Survived', 'Ticket', 'Cabin'], errors='ignore')  # Prevents leakage
    Y = df['Survived'].astype(int)

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # Model Training
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Performance Metrics
    st.write("Sensitivity score:", round(float(recall_score(Y_test, Y_pred)), 2))
    st.write("Accuracy Score:", round(float(accuracy_score(Y_test, Y_pred)), 2))
    st.write("Precision Score:", round(float(precision_score(Y_test, Y_pred)), 2))
    st.write("F1 Score:", round(float(f1_score(Y_test, Y_pred)), 2))
    st.write("Log Loss:", round(float(log_loss(Y_test, model.predict_proba(X_test)))), 2)

    # ROC Curve
    fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)
