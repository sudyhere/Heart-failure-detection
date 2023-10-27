import streamlit as st
import pandas as pd
from PIL import Image
image1 = Image.open('Images/1.png')
image2 = Image.open('Images/2.png')
image3 = Image.open('Images/3.png')
image4 = Image.open('Images/4.png')
image5 = Image.open('Images/5.png')
image7 = Image.open('Images/7.png')
image8 = Image.open('Images/8.png')
image9 = Image.open('Images/9.png')
image10 = Image.open('Images/10.png')
image11 = Image.open('Images/11.png')
image12 = Image.open('Images/12.png')
image13 = Image.open('Images/13.png')
image14 = Image.open('Images/14.png')
image15 = Image.open('Images/15.png')
image16 = Image.open('Images/16.png')
image17 = Image.open('Images/17.png')
def show_explore_page():
    st.title("Statistics and visualizations on clinical heart related data")

    st.subheader("Plots")


    st.write("Age Distribution")
    st.image(image1, caption="The bar graph represents the age distribution of patients. The x-axis represents the age and the y axis represents counts of number of people in that age group ",width = 390)

    st.write("Sex Distribution")
    st.image(image2, caption="The bar graph represents the sex distribution of patients. The x-axis represents the sex and the y axis represents counts of number of people in that age group  ",width = 400)

    st.write("Distribution of cholesterol")
    st.image(image3, caption="Graphical representation of Cholesterol Values , ranging from 250 to 600 , majorly the cholesterol is in the range of 200-350 , x axis represent cholesterol and y axis represents number of people in that age group ",width = 400)

    st.write("Distribution of chest pain")
    st.image(image4, caption="The visualization of chest pain types asy, nap, ata and ta. Majorly people have a chest pain type of asy ranging around 600 ",width = 400)

    st.write("Fasting blood sugar")
    st.image(image7, caption="Graphical representation of Fasting blood sugar. Evidently showing majorly people have 0 fasting blood sugar in a range of 800 . and less people in a range of 1 ",width = 400)

    st.write("Resting ECG")
    st.image(image8, caption="A graphical representation of ECG , the bar graphs rightly shows that majorly people have normal ECG. Very less people have ST and least have LVH resting ECG",width = 400)

    st.write("Maximum Heart rate")
    st.image(image10, caption="Distribution of Max HR values for patients. The mean of the maximum heart rate value is 140 and the graph is slightly left skewed. X axis shows Max HR and y axis the number of people ",width = 400)

    st.write("Exercise angina distribution")
    st.image(image11, caption="Graphical representation of Exercise Angina, Maximum people do not have exercise angina while 400 people have it.",width = 400)

    st.write("ST slope distribution")
    st.image(image12, caption="Graphical representation of Exercise Angina, Maximum people do not have exercise angina while 400 people have it.",width = 400)

    st.write("Gender-Disease distribution")
    st.image(image13, caption="Graphical representation of number of each gender having a disease or not. The bar graph rightly shows that male have high risk of heart diseases as compared to females.  1. Number of males who have heart disease. 2. Number of females who have heart disease",width = 400)
    st.image(image14, caption="Box graphical representation number of each gender having a disease or not. 350 males have high risk and 69 females with no heart disease and no heart risks",width = 400)

    st.write("Chest pain type-Disease distribution")
    st.image(image15, caption="Graphical representation of chest pain type V.S Heart disease. The people having heart disease and chest pain type as asymptomatic are high in numbers whereas people with heart disease and chest pain type ta are the least",width = 400)

    st.write("ECG result-Disease distribution")
    st.image(image16, caption="The graphical representation of ECG Test results vs the heart disease. Majorly people who have ECG reading normal have less heart risk as compared to people in LVH and ST.",width = 400)

    st.write("Exercise angina-Disease distribution")
    st.image(image17, caption="Graphical representation of Exercise Angina V.S heart Disease. The X axis shows people having Angina or not in a range of 0-1 , The Y axis shows the number of people having having  exercise Angina. The graph shows majorly people do not have exercise Angina in a value of 0. ",width = 400)




    












    



