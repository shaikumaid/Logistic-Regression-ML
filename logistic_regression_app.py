import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, precision_score, f1_score, roc_curve, roc_auc_score, log_loss

st.title('Logistic Regression Analysis')

# File Uploading
uploaded_train = st.file_uploader("Upload Training CSV", type=["csv"])
uploaded_test = st.file_uploader("Upload Testing CSV", type=["csv"])

if uploaded_train and uploaded_test:
    df1 = pd.read_csv(uploaded_train)
    df = pd.read_csv(uploaded_test)
    st.write("Training Data Sample:", df1.head())
    st.write("Testing Data Sample:", df.head())
    
    # Concatenating Files
    df2 = pd.concat([df1, df], ignore_index=True)
    df2['Age'].fillna(df2['Age'].median(), inplace=True)
    df2['Fare'].fillna(df2['Fare'].median(), inplace=True)
    df2['Cabin'].fillna("C23 C25 C27", inplace=True)
    df2['Embarked'].fillna("S", inplace=True)
    
    # Standardization
    SS = StandardScaler()
    numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    df2[numerical_cols] = SS.fit_transform(df2[numerical_cols])
    
    # Label Encoding
    LE = LabelEncoder()
    categorical_cols = ["Ticket", "Sex", "Cabin", "Embarked"]
    for col in categorical_cols:
        df2[col] = LE.fit_transform(df2[col])
    
    # Preparing Data
    df2 = df2.dropna(subset=['Survived'])  # Ensure no NaN values in target
    X = df2.select_dtypes(include=[np.number])
    Y = df2['Survived'].astype(int)
    
    # Model Training
    model = LogisticRegression()
    if X.isnull().sum().sum() > 0:
        st.write("Warning: Missing values found in X. Filling with median.")
        X.fillna(X.median(), inplace=True)
    model.fit(X, Y)
    Y_pred = model.predict(X)
    
    # Ensure Y and Y_pred are valid
    if len(Y) > 0 and len(Y_pred) > 0:
        st.write("Sensitivity score:", recall_score(Y, Y_pred).round(2))
        st.write("Accuracy Score:", accuracy_score(Y, Y_pred).round(2))
        st.write("Precision Score:", precision_score(Y, Y_pred).round(2))
        st.write("F1 Score:", f1_score(Y, Y_pred).round(2))
        st.write("Log Loss:", log_loss(Y, Y_pred).round(2))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(Y, model.predict_proba(X)[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(plt)
    else:
        st.write("Error: Y or Y_pred is empty or not properly defined.")
