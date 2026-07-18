import joblib
import pandas as pd
import streamlit as st


model = joblib.load('model.pkl')

if hasattr(model, "feature_names_in_"):
    feature_columns = list(model.feature_names_in_)
else:
    df = pd.read_csv('heart.csv')
    feature_columns = [col for col in df.columns if col != "target"]

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

st.title("Heart Disease Prediction")
st.write("Fill in the patient details below to estimate whether heart disease is likely.")

with st.form("heart_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
        sex = st.selectbox("Sex", ["female", "male"])
        cp = st.selectbox(
            "Chest Pain Type",
            ["0 - typical angina", "1 - atypical angina", "2 - non-anginal pain", "3 - asymptomatic"],
        )
        trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=220, value=120, step=1)
        chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200, step=1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["no", "yes"])

    with col2:
        restecg = st.selectbox(
            "Resting ECG",
            ["0 - normal", "1 - ST-T wave abnormality", "2 - left ventricular hypertrophy"],
        )
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150, step=1)
        exang = st.selectbox("Exercise Induced Angina", ["no", "yes"])
        oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope", ["0 - upsloping", "1 - flat", "2 - downsloping"])
        ca = st.number_input("Number of major vessels", min_value=0, max_value=4, value=0, step=1)
        thal = st.selectbox(
            "Thalassemia",
            ["0 - normal", "1 - fixed defect", "2 - reversable defect", "3 - not specified"],
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    sex_value = 0 if sex == "female" else 1
    cp_value = int(cp.split(" - ")[0])
    fbs_value = 0 if fbs == "no" else 1
    restecg_value = int(restecg.split(" - ")[0])
    exang_value = 0 if exang == "no" else 1
    slope_value = int(slope.split(" - ")[0])
    ca_value = ca
    thal_value = int(thal.split(" - ")[0])

    input_data = {
        "age": age,
        "sex": sex_value,
        "cp": cp_value,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_value,
        "restecg": restecg_value,
        "thalach": thalach,
        "exang": exang_value,
        "oldpeak": oldpeak,
        "slope": slope_value,
        "ca": ca_value,
        "thal": thal_value,
    }

    feature_df = pd.DataFrame([input_data], columns=feature_columns)
    prediction = int(model.predict(feature_df)[0])
    probs = model.predict_proba(feature_df)[0]

    if hasattr(model, "classes_"):
        positive_index = list(model.classes_).index(1) if 1 in model.classes_ else None
    else:
        positive_index = None

    if prediction == 1:
        st.error("Prediction: Heart disease is likely.")
    else:
        st.success("Prediction: No heart disease is indicated.")

    if positive_index is not None and len(probs) > positive_index:
        st.metric("Probability of heart disease", f"{probs[positive_index] * 100:.1f}%")
    else:
        st.metric("Prediction label", prediction)