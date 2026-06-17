# ============================================================
# USED CAR PRICE PREDICTION - MODEL TRAINING
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

import os

print("Script Location:")
print(os.path.dirname(os.path.abspath(__file__)))

script_dir = os.path.dirname(os.path.abspath(__file__))

# Build CSV path
csv_path = os.path.join(script_dir, "car_data.csv")
os.chdir(script_dir)
# Load dataset
df = pd.read_csv(csv_path)

print("\nCurrent Working Directory:")
print(os.getcwd())

print("\nDataset Loaded Successfully")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# ============================================================
# DATA OVERVIEW
# ============================================================

print("\nFirst 5 Rows")
print(df.head())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

# ============================================================
# REMOVE DUPLICATES
# ============================================================

df.drop_duplicates(inplace=True)

print("\nDataset Shape After Removing Duplicates")
print(df.shape)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating Car_Age Feature...")

CURRENT_YEAR = 2026

df["Car_Age"] = CURRENT_YEAR - df["Year"]

# Remove unnecessary columns

df.drop(
    ["Year", "Car_Name"],
    axis=1,
    inplace=True
)

# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

print("\nEncoding Categorical Variables...")

df = pd.get_dummies(
    df,
    columns=[
        "Fuel_Type",
        "Seller_Type",
        "Transmission"
    ],
    drop_first=True
)

# ============================================================
# DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Selling_Price",
    axis=1
)

y = df["Selling_Price"]

print("\nFeature Columns:")
print(X.columns.tolist())

# ============================================================
# SAVE FEATURE COLUMNS
# ============================================================

joblib.dump(
    X.columns.tolist(),
    os.path.join(script_dir, "model_columns.pkl")
)

print("\nFeature Columns Saved")

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost Model...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)

print("Training Complete")

# ============================================================
# MAKE PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test
)

# ============================================================
# EVALUATE MODEL
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features")
print(feature_importance.head(10))

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    os.path.join(
        script_dir,
        "best_xgboost_car_price_model.pkl"
    )
)

print("\nModel Saved Successfully")

print(
    "\nSaved Files:"
)

print(
    "- best_xgboost_car_price_model.pkl"
)

print(
    "- model_columns.pkl"
)

print("\nTraining Pipeline Complete")