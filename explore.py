import streamlit as st
import pandas as pd
from PIL import Image
image1 = Image.open('Images/1.png')
image2 = Image.open('Images/2.png')
image3 = Image.open('Images/3.png')
image4 = Image.open('Images/4.png')
image5 = Image.open('Images/5.png')
image6 = Image.open('Images/6.png')
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
    st.image(image1, caption="The bar graph represents the age distribution of patients. The x-axis represents the age and the y axis represents counts of number of people in that age group ",width = 350)

    st.write("Sex Distribution")
    st.image(image2, caption="The bar graph represents the sex distribution of patients. The x-axis represents the sex and the y axis represents counts of number of people in that age group  ",width = 400)

    st.write("Distribution of cholesterol")
    st.image(image3, caption="Graphical representation of Cholesterol Values , ranging from 250 to 600 , majorly the cholesterol is in the range of 200-350 , x axis represent cholesterol and y axis represents number of people in that age group ",width = 600)

    st.write("Distribution of chest pain")
    st.image(image4, caption="The visualization of chest pain types asy, nap, ata and ta. Majorly people have a chest pain type of asy ranging around 600 ",width = 400)

    st.write("Fasting blood sugar")
    st.image(image7, caption="Graphical representation of Fasting blood sugar. Evidently showing majorly people have 0 fasting blood sugar in a range of 800 . and less people in a range of 1 ",width = 400)

    st.write("Maximum Heart rate")
    st.image(image7, caption="Graphical representation of Fasting blood sugar. Evidently showing majorly people have 0 fasting blood sugar in a range of 800 . and less people in a range of 1 ",width = 400)

    st.write("Maximum Heart rate")
    st.image(image6, caption="Graphical representation of Fasting blood sugar. Evidently showing majorly people have 0 fasting blood sugar in a range of 800 . and less people in a range of 1 ",width = 400)






    



