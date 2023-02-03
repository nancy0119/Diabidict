#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import plotly.graph_objects as go
# from mlxtend.plotting import plot_decision_regions
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
import streamlit as st
from PIL import Image
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'D:\open source\Complete WebD\Diabetes Prediction\diabetes.csv')

# HEADINGS
st.title('Diabidict - Diabetes Prediction')
st.sidebar.header('Patient Data')

image = Image.open('intro.png')
st.image(image)

st.subheader('Training Data Stats')
st.write(data.describe())

#replacing zeros with NaN for correct Exploratory data analysis
data_copy = data.copy(deep = True)
data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace = True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace = True)
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace = True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace = True)
data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace = True)

# Scaling the Data
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(data_copy.drop(["Outcome"],axis = 1),), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()

X = data_copy.drop('Outcome', axis=1)
y = data_copy['Outcome']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=7)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def user_Report():
    pregnancies = st.sidebar.slider('Pregnancies', 0,30, 3)
    glucose = st.sidebar.slider('Glucose', 0,200, 120 )
    bp = st.sidebar.slider('Blood Pressure', 0,150, 70 )
    skinthickness = st.sidebar.slider('Skin Thickness', 0,120, 20 )
    insulin = st.sidebar.slider('Insulin', 0,800, 79 )
    bmi = st.sidebar.slider('BMI', 0,85, 20 )
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,10.0, 0.47 )
    age = st.sidebar.slider('Age', 0,100, 33 )

    user_report_data = {
      'Pregnancies':pregnancies,
      'Glucose':glucose,
      'BloodPressure':bp,
      'SkinThickness':skinthickness,
      'Insulin':insulin,
      'BMI':bmi,
      'DiabetesPedigreeFunction':dpf,
      'Age':age
  }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

user_data = user_Report()
st.subheader('Patient Data')
st.write(user_data)


# MODEL
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
user_result = rfc.predict(user_data)

# rfc_train = rfc.predict(X_train)
# rfc_predictions = rfc.predict(X_test)

# OUTPUT
st.subheader('Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rfc.predict(X_test))*100)+'%')
