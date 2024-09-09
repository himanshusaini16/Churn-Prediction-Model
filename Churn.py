import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = load_model('churn.h5')

# Simulating training data for fitting the StandardScaler
sc = StandardScaler()
sc.fit(np.random.rand(100, 12))  # Replace with your actual training data

# Add custom CSS for styling input fields and other elements
st.markdown("""
    <style>
       
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #ff5733;
            text-align: center;
            font-family: 'Arial', sans-serif;
        }
        h2 {
            color: #4e73df;
            text-align: center;
        }
        label {
            font-weight: bold;
            color: ;
        }
        input[type=number], input[type=text], .stNumberInput input {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            color: #333333;
            border: 2px solid #007bff;
        }
        .stNumberInput input:focus, input[type=text]:focus {
            border-color: #ff5733;
            box-shadow: 0 0 8px rgba(255, 87, 51, 0.5);
            outline: none;
        }
        .stButton>button {
            color: white;
            background-color: #4e73df;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #3e5cb2;
        }
        .stRadio label, .stCheckbox label {
            font-weight: bold;
            color: #333333;
        }
        .stCheckbox {
            margin-bottom: 10px;
        }
        table {
            margin-top: 20px;
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 2px solid #007bff;
            padding: 8px;
            text-align: center;
            color:black;
        }
        th {
            background-color: white;
            color:black;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>Churn Prediction Model</h1>", unsafe_allow_html=True)
st.markdown("---------")

with st.container():
    st.write("### User Inputs")
    city = st.text_input("Enter City Name")
    credit_score = st.number_input("Enter Credit Score", value=0)
    gender = st.radio("Choose Gender", ['Male', 'Female'])
    age = st.number_input("Enter Age", value=0)
    tenure = st.number_input("Enter Tenure", value=0)
    bank_balance = st.number_input("Enter Bank Balance", value=0)
    no_of_product = st.number_input("Enter Number Of Products", value=0)
    has_credit_card = st.checkbox('Has Credit Card')
    is_active_member = st.checkbox('Is Active Member')
    estimated_salary = st.number_input("Enter Estimated Salary", value=0)

gender = 1 if gender == 'Male' else 0
has_credit_card = 1 if has_credit_card else 0
is_active_member = 1 if is_active_member else 0

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [bank_balance],
    'NumOfProducts': [no_of_product],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

st.write("### Input Data")
st.table(input_data)

if st.button('Predict'):
    if city == 'France':
        input_scaled = sc.transform([[1, 0, 0, credit_score, gender, age, tenure, bank_balance, no_of_product, has_credit_card, is_active_member, estimated_salary]])
        prediction = model.predict(input_scaled)
    elif city == 'Spain':
        input_scaled = sc.transform([[0, 0, 1, credit_score, gender, age, tenure, bank_balance, no_of_product, has_credit_card, is_active_member, estimated_salary]])
        prediction = model.predict(input_scaled)
    elif city == 'Germany':
        input_scaled = sc.transform([[0, 1, 0, credit_score, gender, age, tenure, bank_balance, no_of_product, has_credit_card, is_active_member, estimated_salary]])
        prediction = model.predict(input_scaled)
        
    churn_prediction = "Churn" if prediction[0][0] > 0.5 else "No Churn"
    st.markdown(f"<h2>Prediction: {churn_prediction}</h2>", unsafe_allow_html=True)
