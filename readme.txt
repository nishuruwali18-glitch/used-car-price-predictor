# 🚗 Used Car Price Prediction using XGBoost

## Overview

This project predicts the resale value of a used car using Machine Learning.

The model was trained using historical used-car data and deployed through a Streamlit web application that allows users to enter vehicle details and receive real-time price predictions.

---

## Features

* Data Cleaning and Preprocessing
* Feature Engineering
* XGBoost Regression Model
* Model Evaluation using:

  * MAE
  * RMSE
  * R² Score
* Streamlit Web Application
* Real-Time Price Prediction

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Streamlit
* Joblib

---

## Project Structure

Used-Car-Price-Predictor/

├── app.py

├── train.py

├── car_data.csv

├── best_xgboost_car_price_model.pkl

├── model_columns.pkl

├── requirements.txt

└── README.md

---

## Installation

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Example Inputs

* Showroom Price: ₹10 Lakhs
* Kilometers Driven: 25,000
* Owners: 1
* Fuel Type: Petrol
* Seller Type: Individual
* Transmission: Manual

Predicted Resale Value:

₹7.82 Lakhs

---

## Future Improvements

* Hyperparameter Tuning
* Cross Validation
* SHAP Explainability
* Cloud Deployment
* Advanced UI Design

---

## Author

Nishant Ruwali
