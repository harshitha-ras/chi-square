import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load or train the model
@st.cache_resource
def load_model():
    # Load your actual dataset here
    data = pd.DataFrame({
        'Gender': [1, 0, 1, 0, 1],
        'Married': [1, 1, 0, 1, 0],
        'Education': [1, 0, 0, 1, 1],
        'ApplicantIncome': [5000, 3000, 4000, 2500, 6000],
        'Loan_Status': [1, 0, 1, 0, 1]
    })

    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    
    return model

model = load_model()

st.title('Loan Approval Prediction')

st.write("""
### Please enter your information to get a loan approval prediction
""")

gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
applicant_income = st.number_input('Applicant Income', min_value=0, max_value=100000, value=5000)

if st.button('Predict'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'Gender': [1 if gender == 'Male' else 0],
        'Married': [1 if married == 'Yes' else 0],
        'Education': [1 if education == 'Graduate' else 0],
        'ApplicantIncome': [applicant_income]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.success('Congratulations! Your loan is likely to be approved.')
    else:
        st.error('Sorry, your loan is likely to be rejected.')

    st.write('Please note that this is a simplified model and should not be used for actual loan decisions.')

st.write("""
### About this app
This app uses a simple decision tree model to predict loan approval based on a few input features. 
The model is trained on a small, synthetic dataset and is intended for demonstration purposes only.
""")