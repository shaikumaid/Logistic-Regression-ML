import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score


st.title('Logistic Regression Analysis')

### Import Files ###
import pandas as pd
df = pd.read_csv("Titanic_test.csv")
df1 = pd.read_csv("Titanic_train.csv")
df

df.info()

df1

### CONCATING TWO FILES ###
df2 = pd.concat([df1,df],ignore_index=True)
df2

# Data cleaning, Finding missing values
# For continuos data the missing values are replaced by median


df2['Age'] = df2['Age'].fillna(df2['Age'].median())
df2['Fare'] = df2['Fare'].fillna(df2['Fare'].median())


df2['Cabin'].mode()

df2['Embarked'].mode()

df2['Cabin']=df2['Cabin'].fillna("C23 C25 C27")
df2['Embarked']=df2['Embarked'].fillna("S")


df2.info()

### Exploratory Data Analysis ###

#HISTOGRAM

df2.hist("Pclass")
print("Skewdness of  Pclass:",df2['Pclass'].skew())
print("Kurtosis of  Pclass",df2['Pclass'].kurt())

df2.hist("Age")
print("Skewdness of  Pclass:",df2['Age'].skew())
print("Kurtosis of  Pclass",df2['Age'].kurt())

df2.hist("SibSp")
print("Skewdness of  Pclass:",df2['SibSp'].skew())
print("Kurtosis of  Pclass",df2['SibSp'].kurt())

df2.hist("Parch")
print("Skewdness of  Pclass:",df2['Parch'].skew())
print("Kurtosis of  Pclass",df2['Parch'].kurt())

df2.hist("Fare")
print("Skewdness of  Pclass:",df2['Fare'].skew())
print("Kurtosis of  Pclass",df2['Fare'].kurt())

#BOXPLOT

import matplotlib.pyplot as plt
plt.boxplot(df2['Pclass'],vert = False)
plt.show()

import numpy as np
q1 = np.percentile(df2["Pclass"],25)
print("25th percentile",q1)
q3 = np.percentile(df2["Pclass"],75)
print("75th percentile",q3)
iqr = q3-q1
print("Inter quartile range:",iqr)
UW = q3 + (1.5*iqr)

import matplotlib.pyplot as plt
plt.boxplot(df2['Age'],vert = False)
plt.show()

import numpy as np
q1 = np.percentile(df2["Age"],25)
print("25th percentile",q1)
q3 = np.percentile(df2["Age"],75)
print("75th percentile",q3)
iqr = q3-q1
print("Inter quartile range:",iqr)
UW = q3 + (1.5*iqr)

import matplotlib.pyplot as plt
plt.boxplot(df2['SibSp'],vert = False)
plt.show()

import numpy as np
q1 = np.percentile(df2["SibSp"],25)
print("25th percentile",q1)
q3 = np.percentile(df2["SibSp"],75)
print("75th percentile",q3)
iqr = q3-q1
print("Inter quartile range:",iqr)
UW = q3 + (1.5*iqr)

import matplotlib.pyplot as plt
plt.boxplot(df2['Parch'],vert = False)
plt.show()

import numpy as np
q1 = np.percentile(df2["Parch"],25)
print("25th percentile",q1)
q3 = np.percentile(df2["Parch"],75)
print("75th percentile",q3)
iqr = q3-q1
print("Inter quartile range:",iqr)
UW = q3 + (1.5*iqr)

import matplotlib.pyplot as plt
plt.boxplot(df2['Fare'],vert = False)
plt.show()

import numpy as np
q1 = np.percentile(df2["Fare"],25)
print("25th percentile",q1)
q3 = np.percentile(df2["Fare"],75)
print("75th percentile",q3)
iqr = q3-q1
print("Inter quartile range:",iqr)
UW = q3 + (1.5*iqr)

# STANDARDISATION
## identify the numerical columns
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
# Identify numerical columns
df_nom = df2[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]

df_nom['Pclass'] = SS.fit_transform(df_nom[['Pclass']])
df_nom['Age'] = SS.fit_transform(df_nom[['Age']])
df_nom['SibSp'] = SS.fit_transform(df_nom[['SibSp']])
df_nom['Parch'] = SS.fit_transform(df_nom[['Parch']])
df_nom['Fare'] = SS.fit_transform(df_nom[['Fare']])
df_nom.head()

# LABEL ENCODING

from sklearn.preprocessing import  LabelEncoder
LE = LabelEncoder()
df_cat = df2[["Ticket","Sex","Cabin","Embarked"]]

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df_cat["Ticket"] = LE.fit_transform(df_cat["Ticket"]) 
df_cat["Sex"] = LE.fit_transform(df_cat["Sex"])
df_cat["Cabin"] = LE.fit_transform(df_cat["Cabin"])
df_cat["Embarked"] = LE.fit_transform(df_cat["Embarked"])

# Display first few rows
df_cat.head()


#data obtained after standardization and label encoding along with target variable
df_new = pd.concat([df_nom,df_cat,df2['Survived']],axis=1)
df_new

# Analyze any correlations or patterns observed in the data

df_new.corr()

### Correlation ###

#sex=-0.54
#Pclass=-0.33
#Fare=0.25
#Embarked=-0.167
#Ticket=-0.1667
#Parch=0.08
#Age=-0.06
#SibSp=-0.03
#Cabin=0.1032

df_new.info()

# Separating data for X and Y that is Training data and Testing data and declaring target variable

X = df_new.iloc[:,0:9]
X.info()

Y=df_new["Survived"]
Y.shape

X.shape

df_new["Survived"].value_counts()

# Separating training data
train = df_new[(df_new["Survived"] == 0) | (df_new["Survived"] == 1)]

test = df_new[df_new['Survived'].isnull()]

test.shape

test.head()

train.head()

train.shape

X = train.iloc[ : ,0:9]
Y = train["Survived"]

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()  # Initialize model
model.fit(X, Y)  # Fit the model


Y_pred = model.predict(X)

Y_pred

#Evaluating the performance of the model on the testing data using accuracy, precision, recall, F1-score,and ROC-AUC score.

# Metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y,Y_pred)
cm

ac = accuracy_score(Y,Y_pred)
print("Accuracy Score:",ac)

from sklearn.metrics import recall_score,precision_score,f1_score

st.write("Sensitivity score:", recall_score(Y, Y_pred).round(2))
print("precision_score:",precision_score(Y,Y_pred).round(2))
print("f1_score:",f1_score(Y,Y_pred).round(2))

TN = cm[0,0]
FP =cm[0,1]
TNR=TN/(TN+FP)
print("Specificity:",TNR.round(2))

#Visualize the ROC curve.
from sklearn.metrics import roc_curve,roc_auc_score

fpr,tpr,dummy = roc_curve(Y,model.predict_proba(X)[:,1:])

import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.ylabel('tpr - True Positive Rate - Sensitivity')
plt.xlabel('fpr - False Positive Rate - (1-specificity)')
plt.show()

rocvalue = roc_auc_score(Y,model.predict_proba(X)[:,1:])
print("Area under curve:",rocvalue.round(3))

#LOG LOSS
from sklearn.metrics import log_loss
print("log loss:",log_loss(Y,Y_pred))

test_data=test.drop(test.columns[9],axis=1)

test_data.head()

y_pred_test_data = model.predict(test_data)

y_pred_test_data

##STEP 7
#STREAMLIT DEPLOYEMENT is DONE LOCALLY AND Observed.

#INTERVIEW QUESTIONS
#1.. What is the difference between precision and recall?

#Precision: 
#This gauges how well the model predicts the good outcomes.
#The ratio of true positive predictions to all positive predictions—true positives as well as false positives—is how it is defined. 
#The equation is
#True Positives/(True Positives + False Positives)


#Recall: 
#Also referred to as True Positive Rate or Sensitivity 
#This assesses the model's capacity to locate all pertinent instances.
#It can be defined as the ratio of real positive cases (the total of true positives and false negatives) to the number of true positive forecasts. 
#The equation is:
#True Positives/(True Positives + False Negatives)

#2. What is cross-validation, and why is it important in binary classification?


#Cross-validation is a method in statistics that assesses the effectiveness of a machine learning model by dividing the data into different subsets for training and testing the model. 
#This procedure aids in verifying that the model's effectiveness is strong and can be applied effectively to new data.

#Important points of Cross-Validation:

#Data Splitting involves partitioning the dataset into a specific amount of folds, typically 5 or 10.
#In k-fold cross-validation, the data is divided into k parts.
#The model is tested on one subset and trained on the other k-1 subsets
#This procedure is performed k times, utilizing each subset as the test set one time.

#Performance Evaluation: Once the k iterations are finished, the performance indicators (such as accuracy, precision, recall, etc.) are averaged 
#to offer a more dependable assessment of the model's performance.

#Significance in Binary Classification:

#Prevention of Overfitting: Through the use of multiple training and testing sets, cross-validation aids in preventing the model from overfitting to 
#a particular segment of the data.

#Improved Data Utilization: It ensures that all available data is utilized effectively by including each data point in both training and
#test sets in various iterations.

#Performance assessment helps in determining how well a model will do on new data,
#assisting in choosing the most suitable model or hyperparameters.

#Cross-validation aids in comprehension of the balance between bias and variance,
#enabling practitioners to make informed choices regarding model complexity.

#In general, cross-validation is essential for evaluating the credibility of a binary classification model,
 #   resulting in improved decision-making in selecting and fine-tuning models.



