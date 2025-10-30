import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils import haversine_distance

# --------------------------
# App Configuration
# --------------------------
st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="centered")

st.title("📦 Amazon Delivery Time Predictor")
st.markdown("### Enter the delivery details below to get an estimated delivery time (in hours).")

# --------------------------
# Load Trained Model
# --------------------------
@st.cache_resource
def load_model(path="models/best_model.pkl"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"❌ Could not load model from {path}. Please make sure training is completed.")
        return None

model = load_model()

# --------------------------
# Input Form
# --------------------------
with st.form("prediction_form"):
    st.subheader("🧾 Order & Agent Details")

    agent_age = st.number_input("Agent Age (years)", min_value=18, max_value=70, value=28)
    agent_rating = st.number_input("Agent Rating (1-5)", min_value=1.0, max_value=5.0, step=0.1, value=4.5)
    store_lat = st.number_input("Store Latitude", value=28.7041)
    store_lon = st.number_input("Store Longitude", value=77.1025)
    drop_lat = st.number_input("Drop Latitude", value=28.5355)
    drop_lon = st.number_input("Drop Longitude", value=77.3910)
    pickup_lag = st.number_input("Pickup Lag (minutes)", min_value=0.0, value=10.0)

    st.subheader("🌦️ Environmental & Delivery Details")
    weather = st.selectbox("Weather", ["Clear", "Rain", "Cloudy", "Windy", "Storm", "Haze"])
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
    vehicle = st.selectbox("Vehicle", ["Bike", "Car", "Van"])
    area = st.selectbox("Area", ["Urban", "Metropolitan", "Rural"])
    category = st.selectbox("Delivery Category", ["Grocery", "Electronics", "Clothing", "Home", "Other"])

    submitted = st.form_submit_button("🚀 Predict Delivery Time")

# --------------------------
# Prediction Logic
# --------------------------
if submitted:
    if model is None:
        st.stop()

    # Calculate distance (km)
    distance_km = haversine_distance(store_lat, store_lon, drop_lat, drop_lon)

    # Simple categorical encodings consistent with training
    mapping = {
        "Weather": {"Clear": 0, "Rain": 1, "Cloudy": 2, "Windy": 3, "Storm": 4, "Haze": 5},
        "Traffic": {"Low": 0, "Medium": 1, "High": 2, "Jam": 3},
        "Vehicle": {"Bike": 0, "Car": 1, "Van": 2},
        "Area": {"Urban": 0, "Metropolitan": 1, "Rural": 2},
        "Category": {"Grocery": 0, "Electronics": 1, "Clothing": 2, "Home": 3, "Other": 4},
    }

    # Prepare single-row dataframe for prediction
    features = pd.DataFrame([{
        "Agent_Age": agent_age,
        "Agent_Rating": agent_rating,
        "Distance_km": distance_km,
        "Pickup_Lag_min": pickup_lag,
        "Hour": 12,
        "DayOfWeek": 2,
        "Weather_enc": mapping["Weather"].get(weather, -1),
        "Traffic_enc": mapping["Traffic"].get(traffic, -1),
        "Vehicle_enc": mapping["Vehicle"].get(vehicle, -1),
        "Area_enc": mapping["Area"].get(area, -1),
        "Category_enc": mapping["Category"].get(category, -1)
    }])

    try:
        prediction = model.predict(features)[0]
        st.success(f"✅ Estimated Delivery Time: **{prediction:.2f} hours**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
