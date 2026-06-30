import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# --- 1. System Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'model1_shipping_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

print("🚚 Starting Model 1: Lead Time Regressor...")

# --- 2. Load Processed Data ---
df = pd.read_csv(PROCESSED_DATA_PATH)

# --- 3. Feature Engineering & Encoding ---
features = ['Ship Mode', 'Region', 'Origin_Factory', 'Distance_Miles', 'Units']
target = 'Delivery_Time_Days'

X = df[features].copy()
y = df[target]

print("⚙️ Encoding categorical variables...")
label_encoders = {}
for col in ['Ship Mode', 'Region', 'Origin_Factory']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# --- 4. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Train the model ---
print("🧠 Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# --- 6. Evaluate the Model ---
print("📊 Evaluating Model Performance...")
y_pred = rf_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} days")
print(f"Mean Absolute Error (MAE):      {mae:.2f} days")
print(f"R-Squared (R2 Score):           {r2:.2f}")

# --- 7. Save the Model and Encoders ---
model_path = os.path.join(MODELS_DIR, 'shipping_regressor.pkl')
encoder_path = os.path.join(MODELS_DIR, 'shipping_encoders.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)

with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoders, f)

print(f"\n💾 SUCCESS! Models saved to {MODELS_DIR}")