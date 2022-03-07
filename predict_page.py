#Predict_page
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler




def load_model():
    with open ("Saved_steps.pkl", 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

rfc = data["model"]
chest_pain_type = ["Typical Angina","Atypical Angina","Non-Anginal Pain", "Asymptomatic"]
sc = StandardScaler()
def show_predict_page():
    st.title("Heart Failure Detection ")
    st.write("""We need data for detection""")
    age = st.slider("Age", min_value=1,max_value=115)
    sex = st.selectbox("sex", ["Male", "Female"])
    if sex == "Male":
        sex =0 
    if sex == "Female":
        sex =1
    

    chest_pain = st.selectbox("Chest pain type",chest_pain_type)
    if chest_pain == "Typical Angina":
        chest_pain =1
    if chest_pain == "Atypical Angina":
        chest_pain =2
    if chest_pain == "Non-Anginal Pain":
        chest_pain =3
    if chest_pain == "Asymptomatic":
        chest_pain =4

    bloodpressure = st.slider("Resting blood pressure",min_value=10,max_value=250)
    cholesterol = st.slider("Cholesterol",min_value=100,max_value=450)
    fasting_bloodsugar = st.selectbox("fasting bloodsugar",[0,1])
    ecg = st.selectbox("ECG",[0,1])
    max_heart_rate = st.slider("Max heart rate",min_value=50,max_value=250)
    exercise_angina = st.selectbox("Exercise angina",["Yes", "No"])
    if exercise_angina == "Yes":
        exercise_angina=1
    
    if exercise_angina == "No":
        exercise_angina=0
    old_peak = st.selectbox("Old peak",[0,1,1.5,2])
    st_slope = st.selectbox("ST Slope",[0,1,1.5,2])
    

    ok = st.button("Calculate the risk")
    
    if ok:
        X = np.array([[age,sex,chest_pain,bloodpressure,cholesterol,fasting_bloodsugar,ecg,max_heart_rate,exercise_angina,old_peak,st_slope]])
        st.write(X)
        
        
        prediction = rfc.predict(X)
        
        st.write(prediction)

    



