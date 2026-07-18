import joblib
import pandas as pd
import streamlit as st


model = joblib.load('model.pkl')

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🏥",
    layout="centered",
)

# --- Custom styling ---
st.markdown("""
    <style>
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        padding: 0.5em 2em;
        font-weight: bold;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 Insurance Premium Predictor")
st.write("Enter your details below to get an estimated insurance charge.")
st.divider()

# --- Sidebar inputs ---
st.sidebar.header("📋 Your Details")

age = st.sidebar.number_input(
    "Age", min_value=0, max_value=100, value=30, step=1
)
bmi = st.sidebar.number_input(
    "BMI", min_value=10.0, max_value=70.0, value=30.0, step=0.1,
    help="Body Mass Index — weight (kg) divided by height squared (m²)"
)
children = st.sidebar.number_input(
    "Number of Children", min_value=0, max_value=10, value=0, step=1
)
sex = st.sidebar.selectbox("Sex", ["female", "male"])
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox(
    "Region",
    ["southwest", "southeast", "northeast", "northwest"],
)

predict_clicked = st.sidebar.button("Predict")

# --- Main area ---
if predict_clicked:
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

    with st.spinner("Calculating your estimated premium..."):
        prediction = model.predict(feature_df)

    st.subheader("Your Estimate")
    st.metric(label="Estimated Annual Charges", value=f"${prediction[0]:,.2f}")

    st.divider()

    # A little context alongside the number
    col1, col2, col3 = st.columns(3)
    col1.metric("Age", age)
    col2.metric("BMI", bmi)
    col3.metric("Smoker", smoker.capitalize())

    st.caption(
        "This is an estimate based on a trained regression model and should not be "
        "considered actual insurance pricing."
    )
else:
    st.info("👈 Fill in your details in the sidebar and click **Predict** to see your estimate.")