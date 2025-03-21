import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("C:/Users/asnav/Documents/MachineLearningProjects/loan_approval_prediction_using_classification/notebook/loan_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Title of the app
st.title("Loan Approval Prediction System")
st.write("Fill the details below to check loan approval status")

# Input fields for all features
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3", "3+"])
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.text_input("Applicant's Income ($)", "5000")
coapplicant_income = st.text_input("Coapplicant Income ($)", "0")
loan_amount = st.text_input("Loan Amount ($)", "1000")
loan_term = st.text_input("Loan Amount Term (in months)", "120")
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Function to process inputs before prediction
def preprocess_inputs(gender, married, dependents, education, self_employed,
                       applicant_income, coapplicant_income, loan_amount,
                       loan_term, credit_history, property_area):
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    dependents = 3 if dependents == "3+" else int(dependents)
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_mapping = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_area = property_mapping[property_area]

    return np.array([[gender, married, dependents, education, self_employed,
                      applicant_income, coapplicant_income, loan_amount,
                      loan_term, credit_history, property_area]])

# Predict button
if st.button("Check Loan Approval"):
    try:
        # Convert inputs to numeric types
        applicant_income = float(applicant_income)
        coapplicant_income = float(coapplicant_income)
        loan_amount = float(loan_amount)
        loan_term = int(loan_term)

        input_data = preprocess_inputs(gender, married, dependents, education,
                                       self_employed, applicant_income, coapplicant_income,
                                       loan_amount, loan_term, credit_history, property_area)
        
        prediction = model.predict(input_data)
        st.write(f"Raw Model Prediction: {prediction}")

        if prediction[0] == "Y":
            st.success("Congratulations! Your loan is likely to be approved.")
        else:
            st.error("Sorry! Your loan application may be rejected.")
    
    except ValueError:
        st.error("Please enter valid numeric values for all fields.")
