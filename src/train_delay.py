import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import warnings

warnings.filterwarnings('ignore')

# --- 1. System Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("🚦 Starting Delay Classification AI...")

# --- 2. Simulate Historical Order Data ---

np.random.seed(42)
n_samples = 2000

regions = ['Atlantic', 'Interior', 'Pacific', 'South', 'Northeast']
ship_modes = ['Standard Class', 'Second Class', 'First Class', 'Same Day']
products = ['Wonka Bar - Milk Chocolate', 'Laffy Taffy', 'SweeTARTS', 'Nerds']

data = {
    'Region': np.random.choice(regions, n_samples),
    'Ship_Mode': np.random.choice(ship_modes, n_samples, p=[0.6, 0.2, 0.15, 0.05]),
    'Units': np.random.randint(10, 500, n_samples),
}
df = pd.DataFrame(data)

# SLA Definitions (How many days we promised the customer)
sla_map = {'Standard Class': 6, 'Second Class': 4, 'First Class': 2, 'Same Day': 1}
df['Promised_SLA_Days'] = df['Ship_Mode'].map(sla_map)


# Simulate Actual Delivery Days
def simulate_actual_days(row):
    base = row['Promised_SLA_Days']
    # Pacific and South have higher delay risks
    region_penalty = 1.5 if row['Region'] in ['Pacific', 'South'] else 0
    # Heavy orders take longer to process
    weight_penalty = 1 if row['Units'] > 300 else 0

    # Introduce randomness
    actual = base + region_penalty + weight_penalty + np.random.normal(0, 1.2)
    return max(1, int(round(actual)))


df['Actual_Days'] = df.apply(simulate_actual_days, axis=1)

# Target Variable: Is it delayed? (Actual > Promised)
df['Is_Delayed'] = (df['Actual_Days'] > df['Promised_SLA_Days']).astype(int)

# --- 3. Encoding & Training ---
encoders = {}
for col in ['Region', 'Ship_Mode']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[['Region', 'Ship_Mode', 'Units']]
y = df['Is_Delayed']

print("⚙️ Training Random Forest Classifier for Delay Risk...")
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X, y)

# --- 4. Save Models ---
model_path = os.path.join(MODELS_DIR, 'delay_classifier.pkl')
encoder_path = os.path.join(MODELS_DIR, 'delay_encoders.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
with open(encoder_path, 'wb') as f:
    pickle.dump(encoders, f)

print("✅ SUCCESS! Delay AI Trained and Saved.")