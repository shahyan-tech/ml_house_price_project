# frontend/app.py
# This is the Streamlit-based frontend for the House Price Prediction app.
# It provides a simple UI for users to input house features and get predictions from the FastAPI backend.
# The app calls the /predict endpoint of the API (running on the same K8s cluster or locally).

import streamlit as st  # Streamlit for building the web UI
import requests  # For making HTTP requests to the API

# =====================
# Config
# =====================
# API_URL: Points to the FastAPI service. In K8s, use the service name; locally in Docker, use host.docker.internal.
API_URL = "http://host.docker.internal:8000"

# Set page config for better UI (title and centered layout)
st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

# App title and description
st.title("🏠 House Price Prediction")
st.write("Enter house details to predict the price")

# =====================
# Input Form
# =====================
# Using Streamlit's form to group inputs and submit at once (prevents re-runs on every input change).
# Inputs match a subset of the API's expected features (from PredictionRequest schema).
# You can expand this to include all features from your model for a full demo.
with st.form("prediction_form"):
    # Numerical inputs with validation (min/max values based on typical house data)
    LotArea = st.number_input("Lot Area (sq ft)", min_value=100, max_value=200000, value=8450)
    OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 7)  # Slider for quality rating
    OverallCond = st.slider("Overall Condition (1–10)", 1, 10, 5)  # Slider for condition rating
    YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2025, value=2003)
    GrLivArea = st.number_input("Above Ground Living Area", min_value=200, max_value=6000, value=1710)

    # Submit button
    submit = st.form_submit_button("Predict")

# =====================
# API Call
# =====================
# When form is submitted, prepare JSON payload matching the API's input schema.
# Make a POST request to /predict, handle success/error, and display results.
if submit:
    # Build the payload as a dict (matches PredictionRequest from schemas.py)
    payload = {
        "LotArea": LotArea,
        "OverallQual": OverallQual,
        "OverallCond": OverallCond,
        "YearBuilt": YearBuilt,
        "GrLivArea": GrLivArea
    }

    # Wrap in "features" as per API schema
    data = {"features": payload}

    try:
        # Send POST request with JSON data and a 5-second timeout
        response = requests.post(f"{API_URL}/predict", json=data, timeout=5)

        if response.status_code == 200:
            # Parse JSON response and display success with formatted price
            prediction = response.json()["prediction"]
            st.success(f"💰 Predicted Price: ${prediction:,.2f}")
        else:
            # Handle API errors (e.g., invalid input)
            st.error(f"API Error: {response.text}")

    except requests.exceptions.RequestException as e:
        # Handle network/connection errors
        st.error(f"Could not connect to API: {e}")

# =====================
# Notes
# =====================
# - This is a basic demo; expand inputs to match all model features for accuracy.
# - In production, add authentication, more validation, or integrate with MongoDB for history.
# - For K8s: Deploy as a separate service; use Ingress for routing.