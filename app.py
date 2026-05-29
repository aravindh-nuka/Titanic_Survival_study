import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("titanic_model.pkl")

# App title
st.title("Titanic Survival Prediction")

# User Inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

age = st.number_input("Age")

sibsp = st.number_input("Siblings/Spouse")

parch = st.number_input("Parents/Children")

fare = st.number_input("Fare")

embarked = st.selectbox(
    "Embarked",
    ["C", "Q", "S"]
)

# Encoding
sex = 1 if sex == "Male" else 0

embarked_mapping = {
    "C": 0,
    "Q": 1,
    "S": 2
}

embarked = embarked_mapping[embarked]

# Prediction Button
if st.button("Predict"):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger Likely Survived")
    else:
        st.error("Passenger Likely Did Not Survive")