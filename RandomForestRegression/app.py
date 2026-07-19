import joblib
import pandas as pd
import streamlit as st


model = joblib.load("model.pkl")
feature_names = list(model.feature_names_in_)

# Real car specs, used to auto-fill fields based on the chosen name + year
car_data = pd.read_csv("Car details v3.csv")

spec_cols = ["fuel", "seller_type", "transmission", "owner", "mileage", "engine", "max_power", "torque", "seats"]

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
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

st.title("🚗 Car Price Predictor")
st.write("Choose your car's model and year — we'll fill in the rest of the specs for you.")
st.divider()

# --- Sidebar inputs ---
st.sidebar.header("📋 Car Details")

name = st.sidebar.selectbox("Car Model", sorted(car_data["name"].unique()))

years_for_name = sorted(car_data.loc[car_data["name"] == name, "year"].unique(), reverse=True)
year = st.sidebar.selectbox("Year", years_for_name)

km_driven = st.sidebar.number_input(
    "Kilometers Driven", min_value=0, max_value=1_000_000, value=50000, step=1000
)

# Rows matching this exact name + year
subset = car_data[(car_data["name"] == name) & (car_data["year"] == year)]

st.sidebar.divider()
st.sidebar.caption("Specs below are filled in automatically. Pick an option only where more than one exists for this car.")

chosen_specs = {}
for col in spec_cols:
    unique_vals = subset[col].unique().tolist()
    if len(unique_vals) == 1:
        chosen_specs[col] = unique_vals[0]
        st.sidebar.text_input(col.replace("_", " ").title(), value=str(unique_vals[0]), disabled=True)
    else:
        chosen_specs[col] = st.sidebar.selectbox(col.replace("_", " ").title(), unique_vals)

predict_clicked = st.sidebar.button("Predict")

# --- Main area ---
if predict_clicked:
    row = dict.fromkeys(feature_names, 0)

    row["year"] = year
    row["km_driven"] = km_driven
    row["seats"] = chosen_specs["seats"]

    row[f"name_{name}"] = 1
    row[f"fuel_{chosen_specs['fuel']}"] = 1
    row[f"seller_type_{chosen_specs['seller_type']}"] = 1
    row[f"transmission_{chosen_specs['transmission']}"] = 1
    row[f"owner_{chosen_specs['owner']}"] = 1
    row[f"mileage_{chosen_specs['mileage']}"] = 1
    row[f"engine_{chosen_specs['engine']}"] = 1
    row[f"max_power_{chosen_specs['max_power']}"] = 1
    row[f"torque_{chosen_specs['torque']}"] = 1

    feature_df = pd.DataFrame([row])[feature_names]

    with st.spinner("Estimating price..."):
        prediction = model.predict(feature_df)

    st.subheader("Your Estimate")
    st.metric(label="Estimated Selling Price", value=f"₹{prediction[0]:,.0f}")

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Year", year)
    col2.metric("KM Driven", f"{km_driven:,}")
    col3.metric("Fuel", chosen_specs["fuel"])

    st.caption(
        "This is an estimate based on a trained Random Forest Regression model "
        "and should not be considered an actual valuation."
    )
else:
    st.info("👈 Choose the car model and year in the sidebar, then click **Predict** to see the estimate.")