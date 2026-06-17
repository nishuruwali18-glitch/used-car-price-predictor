# ============================================================
# USED CAR PRICE PREDICTOR
# STREAMLIT FRONTEND
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import os

# ============================================================
# LOAD MODEL FILES
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    script_dir,
    "best_xgboost_car_price_model.pkl"
)

columns_path = os.path.join(
    script_dir,
    "model_columns.pkl"
)

model = joblib.load(model_path)

model_columns = joblib.load(columns_path)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ============================================================
# PAGE TITLE
# ============================================================

st.title("🚗 Used Car Price Predictor")

st.markdown(
    """
    Enter the vehicle details below and click
    **Predict Price** to estimate the resale value.
    """
)

# ============================================================
# USER INPUTS
# ============================================================

present_price = st.number_input(
    "Current Showroom Price (Lakhs)",
    min_value=0.0,
    value=5.0
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=20000
)

owner = st.selectbox(
    "Previous Owners",
    [0, 1, 2, 3]
)

car_age = st.slider(
    "Car Age (Years)",
    min_value=0,
    max_value=25,
    value=5
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("Predict Price"):

    # Create empty dataframe
    input_df = pd.DataFrame(
        columns=model_columns
    )

    # Fill with zeros
    input_df.loc[0] = 0

    # Numerical features

    input_df["Present_Price"] = present_price
    input_df["Kms_Driven"] = kms_driven
    input_df["Owner"] = owner
    input_df["Car_Age"] = car_age

    # Fuel Type

    if "Fuel_Type_Diesel" in model_columns:
        input_df["Fuel_Type_Diesel"] = (
            1 if fuel_type == "Diesel" else 0
        )

    if "Fuel_Type_Petrol" in model_columns:
        input_df["Fuel_Type_Petrol"] = (
            1 if fuel_type == "Petrol" else 0
        )

    # Seller Type

    if "Seller_Type_Individual" in model_columns:
        input_df["Seller_Type_Individual"] = (
            1 if seller_type == "Individual" else 0
        )

    # Transmission

    if "Transmission_Manual" in model_columns:
        input_df["Transmission_Manual"] = (
            1 if transmission == "Manual" else 0
        )

    # Make Prediction

    prediction = model.predict(
        input_df
    )[0]

    # Display Prediction

    st.success(
        f"Estimated Resale Value: ₹ {prediction:.2f} Lakhs"
    )

    # Additional message

    st.info(
        "Prediction generated using XGBoost Machine Learning Model"
    )