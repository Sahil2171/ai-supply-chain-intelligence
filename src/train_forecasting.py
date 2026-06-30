import pandas as pd
import numpy as np
import os
import pickle
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Suppress harmless statistical warnings
warnings.filterwarnings('ignore')

# --- 1. System Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'model2_demand_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

print("📈 Starting Advanced Multi-Product Forecasting Engine...")

# --- 2. Load Processed Data ---
try:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {PROCESSED_DATA_PATH}. Run preprocessing.py first.")
    exit()

# Storing all our trained models inside one pkl file
forecasters = {}

# --- 3. Train the Aggregate (Master) Model ---
print("\n⚙️ Training Aggregate Master Model...")
# Resample all units to a weekly timeline
aggregate_demand = df.groupby('Order Date')['Units'].sum().resample('W').sum().fillna(0)

# Train the model with trend and 4-week seasonality
agg_model = ExponentialSmoothing(
    aggregate_demand, trend='add', seasonal='add', seasonal_periods=4
).fit(optimized=True)

forecasters['All Products (Aggregate)'] = agg_model
print("✅ Master Model Trained.")

# --- 4. Train Individual Product Models ---
products = df['Product Name'].unique()
print(f"\n⚙️ Training {len(products)} Individual Product Models...")

for product in products:
    # Isolate data for just this one product
    prod_df = df[df['Product Name'] == product]
    prod_demand = prod_df.groupby('Order Date')['Units'].sum().resample('W').sum().fillna(0)


    prod_demand = prod_demand.reindex(aggregate_demand.index, fill_value=0)

    try:
        # full Holt-Winters with seasonality
        model = ExponentialSmoothing(
            prod_demand, trend='add', seasonal='add', seasonal_periods=4
        ).fit(optimized=True)
    except ValueError:

        # If seasonality fall back to standard Exponential Smoothing.
        model = ExponentialSmoothing(
            prod_demand, trend=None, seasonal=None
        ).fit(optimized=True)

    forecasters[product] = model
    print(f"  -> Trained: {product}")

# --- 5. Save the Multi-Model Dictionary ---
model_path = os.path.join(MODELS_DIR, 'demand_forecaster.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(forecasters, f)

print(f"\n💾 SUCCESS! Saved {len(forecasters)} distinct AI models to: {model_path}")