

import joblib
import pandas as pd
import streamlit as st



model = joblib.load('model.pkl')

st.set_page_config(
    page_title="Insurance Premium Predictor",
    layout="centered",
)

st.title("Insurance Premium Predictor")
st.write("Enter the details below to estimate insurance charges")

age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=30.0, step=0.1)
children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northeast", "northwest"],
)

if st.button("Predict"):
    input_df = pd.DataFrame(
        [{
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex": sex,
            "smoker": smoker,
            "region": region,
        }]
    )
    feature_df = pd.get_dummies(
        input_df,
        columns=["sex", "smoker", "region"],
        dtype=int,
    )
    feature_df = feature_df.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(feature_df)
    st.success(f"Prediction: {prediction[0]:.2f}")