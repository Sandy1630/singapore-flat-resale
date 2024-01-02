import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import pickle

with st.sidebar:
    selected=option_menu("Main_menu",["Main_menu","selling_price","About"])
    
if selected=="Main_menu":
    st.title("""
    Welcome to the Singapore Flat Resale Price Predition app!
    
    This app aims to predict Singapore flat Resale price using machine learning  techniques based on a dataset containg various features related to flats.""")
    
    
    st.write("## Introduction")
    st.write("""
        the goal of this project is develop a machine learning model capable of accuratley predicting Singapore Flat Resale price based on features such as Squre feet,town,commence year,flat model and more..""")  
    
if selected=="About":
    st.title("About")
    st.write(
    "This Streamlit app is designed to address challenges in the Flat Resale Price by providing solutions for data analytics. "
    "Enter your data to optimize pricing decisions.")
    
    st.write("""**Singapore Flat Resale  Price** is a cutting-edge data-driven solution tailored to meet the unique challenges faced by Flat Price. We empower businesses in the Flat Price to harness the power of advanced analytics and machine learning for enhanced improved pricing strategies.""")

    st.markdown('__<p style="text-align:left; font-size: 20px; color: #FAA026">For feedback/suggestion, connect with me on</P>__',
                unsafe_allow_html=True)
    st.subheader("Email ID")
    st.write("santhoshkumar.e2000@gmail.com")
    st.subheader("Github")
    st.write("https://github.com/Sandy1630")
    st.balloons()

if selected=="selling_price":
    
    st.title("Singpore Flat Resale Price")
    col1,col2=st.columns(2)
    with col1:
        square_feet=st.slider("pleast Select range of square feet of floor",28,307,50)
        com_year=st.text_input("Please Enter Lease Commence Year in YYYY format")
        flat_type=st.selectbox("which flat type would you like to select",("1 ROOM","2 ROOM","3 ROOM","4 ROOM","5 ROOM"," EXECUTIVE ","MULTI-GENERATION"))
    with col2:
        flat_model=st.selectbox("Which Flat model would you like to select",("Model A","Improved","New Generation","Simplified","Premium Apartment","Standard","Apartment","Maisonette","Model A2","DBSS","Model A-Maisonette","Adjoined flat","Terrace","Multi Generation","Type S1","Type S2","Improved-Maisonette","Premium Apartment Loft","2-room","Premium Maisonette","3Gen"))
        town=st.selectbox("Which Town would you like to select",("TAMPINES","YISHUN","BEDOK","JURONG WEST","WOODLANDS","ANG MO KIO",'HOUGANG','BUKITBATOK','CHOA CHU KANG','BUKIT MERAH','PASIR RIS','SENGKANG','TOA PAYOH','QUEENSTOWN','GEYLANG','CLEMENTI','BUKIT PANJANG','KALLANG/WHAMPOA','JURONG EAST','SERANGOON','BISHAN',"PUNGGOL",'SEMBAWANG','MARINE PARADE','CENTRAL AREA','BUKIT TIMAH','LIM CHU KANG'))
        storey_range=st.selectbox("Please Select Storey Range",('04 TO 06','07 TO 09','01 TO 03','10 TO 12','13 TO 15','16 TO 18','19 TO 21','22 TO 24','25 TO 27','01 TO 05','06 TO 10','28 TO 30','11 TO 15','31 TO 33','34 TO 36','37 TO 39','16 TO 20','40 TO 42','21 TO 25','43 TO 45','46 TO 48','26 TO 30','49 TO 51','36 TO 40','31 TO 35'))
    flat_type_mapping={'1 ROOM':0, '3 ROOM':2, '4 ROOM':3, '5 ROOM':4, '2 ROOM':1, 'EXECUTIVE':5,
       'MULTI-GENERATION':6}
    flat_model_mapping={'Improved':6, 'New Generation':0, 'Model A':5, 'Standard':3, 'Simplified':1,
       'Model A-Maisonette':10, 'Apartment':13, 'Maisonette':12, 'Terrace':15,
       '2-room':2, 'Improved-Maisonette':9, 'Multi Generation':14,
       'Premium Apartment':8, 'Adjoined flat':11, 'Premium Maisonette':16,
       'Model A2':4, 'DBSS':18, 'Type S1':21, 'Type S2':22, 'Premium Apartment Loft':20,
       '3Gen':17}
    town_mapping={'ANG MO KIO':2, 'BEDOK':5, 'BISHAN':24, 'BUKIT BATOK':6, 'BUKIT MERAH':21,
       'BUKIT TIMAH':26, 'CENTRAL AREA':19, 'CHOA CHU KANG':20, 'CLEMENTI':8,
       'GEYLANG':4, 'HOUGANG':13, 'JURONG EAST':7, 'JURONG WEST':9,
       'KALLANG/WHAMPOA':14, 'MARINE PARADE':18, 'QUEENSTOWN':10, 'SENGKANG':25,
       'SERANGOON':15, 'TAMPINES':17, 'TOA PAYOH':11, 'WOODLANDS':12, 'YISHUN':3,
       'LIM CHU KANG':0, 'SEMBAWANG':22, 'BUKIT PANJANG':16, 'PASIR RIS':25,
       'PUNGGOL':27}
    storey_range_mapping={'10 TO 12':3, '04 TO 06':1, '07 TO 09':2, '01 TO 03':0, '13 TO 15':4,
       '19 TO 21':8, '16 TO 18':5, '25 TO 27':12, '22 TO 24':10, '28 TO 30':16,
       '31 TO 33':18, '40 TO 42':21, '37 TO 39':20, '34 TO 36':19, '06 TO 10':7,
       '01 TO 05':6, '11 TO 15':9, '16 TO 20':11, '21 TO 25':13, '26 TO 30':14,
       '36 TO 40':17, '31 TO 35':15, '46 TO 48':23, '43 TO 45':22, '49 TO 51':24}
    flat_type=flat_type_mapping[flat_type]
    flat_model=flat_model_mapping[flat_model]
    town=town_mapping[town]
    storey_range=storey_range_mapping[storey_range]
    
    
    if com_year:
        
        with open("C:\\Users\\santh\\OneDrive\\Documents\\model_falt_price.pkl","rb") as file:
            model=pickle.load(file)
        new_sample=np.array([[int(town),int(flat_type),int(storey_range),int(square_feet),int(flat_model),int(com_year)]])
        new_pred=model.predict(new_sample)
        if st.button("Price"):
            st.write("## :green[Price]",np.round(new_pred))
    
 