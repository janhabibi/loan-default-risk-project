import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_default_model.pkl")

st.title("Loan Default Risk Predictor")
st.write("By Daniel (Euichan) Kim")
st.markdown("[View project on GitHub](https://github.com/danielkim-im/loan-default-risk-project)")

st.markdown("---")

st.subheader("Loan Information")

# Input
grade_options = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7
}

loan_amnt = st.number_input("Loan Amount ($)", min_value=0, step=100)
term = st.selectbox("Loan Term (months)", [36, 60])

selected_grade_label = st.selectbox("Credit Grade", options=list(grade_options.keys()))
grade_num = grade_options[selected_grade_label]

emp_length_num = st.selectbox("Employment Length (years)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
annual_inc = st.number_input("Annual Income ($)", value=0, step=100)
dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, value=0.0, step=1.0)
fico_range_low = st.number_input("FICO Range Low", min_value=300, max_value=850)

fico_range_high = st.number_input("FICO Range High", min_value=300, max_value=850)


# Predict
if st.button("Predict"):
    features = np.array([[loan_amnt, term, grade_num, emp_length_num, annual_inc, dti, fico_range_high, fico_range_low]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("Prediction Result")
    #st.write(f"Prediction: {'Default' if prediction==1 else 'No Default'}")
    st.write(f"Probability of Default: {probability * 100:.2f}%")