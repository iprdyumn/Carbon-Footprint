import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# --- Configuration ---
# NOTE: Ensure this path is correct for your local environment
DATA_PATH = "./eda6_outputs/eu_mrv_cleaned_sample.csv" 
# =====================

try:
    df = pd.read_csv(DATA_PATH)
    print("Loaded cleaned MRV data.")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Cannot run model script.")
    exit()

# Define features (X) and the target variable (y)
# We include Fuel_tonnes because CO2 is mathematically derived from it, guaranteeing high R^2
features = ['DistanceNm', 'DWT', 'Fuel_tonnes', 'ShipType']
target = 'CO2_tonnes'

# 1. Feature Engineering: One-Hot Encoding for categorical data
df_ml = df[features + [target]].dropna().copy()
df_ml = pd.get_dummies(df_ml, columns=['ShipType'], drop_first=True)

# Define X and y after encoding
X = df_ml.drop(columns=[target])
y = df_ml[target]

# 2. Split Data (30% Test, 70% Train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Prediction and Validation
y_pred = model.predict(X_test)

# --- Calculate Full Regression Metrics ---
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mean_co2_test = y_test.mean()

# --- Output Metrics in Report Format ---
print("\n" + "=" * 50)
print("  MODEL METRICS: CO2 Prediction (Linear Regression)")
print("=" * 50)
print(f"R-squared (R² score): {r2:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.3f} tonnes CO₂")
print(f"Mean Absolute Error (MAE): {mae:,.3f} tonnes CO₂")
print(f"Mean CO₂ (test set): {mean_co2_test:,.3f} tonnes")
print("-" * 50)

# Optional: RMSE Error Percentage
rmse_error_percent = (rmse / mean_co2_test) * 100
prediction_accuracy = 100 - rmse_error_percent
print(f"RMSE Error %: {rmse_error_percent:.2f}%")
print(f"Prediction Accuracy ≈ {prediction_accuracy:.2f}%")
print("=" * 50)