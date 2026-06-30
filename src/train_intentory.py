import pandas as pd
import numpy as np
import os
import pickle

# --- 1. System Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'model3_inventory_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

print("🏭 Starting Model 3: Inventory Optimization Engine...")

# --- 2. Load Processed Data ---
try:
    df = pd.read_csv(PROCESSED_DATA_PATH)
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {PROCESSED_DATA_PATH}.")
    exit()


# --- 3. Build the Prescriptive Analytics Engine ---
class InventoryOptimizer:

    def __init__(self):
        self.risk_multiplier = 1.2  # 20% extra safety margin for supply chain disruptions

    def evaluate_stock(self, current_stock, daily_demand, lead_time, safety_stock):
        # 1. Calculate Reorder Point (ROP)
        # Formula: (Lead Time * Daily Demand) + Safety Stock
        reorder_point = int((lead_time * daily_demand) + (safety_stock * self.risk_multiplier))

        # 2. Calculate Days of Supply Left
        days_of_supply = int(current_stock / daily_demand) if daily_demand > 0 else 999

        # 3. Determine Prescriptive Action
        if current_stock <= safety_stock:
            status = "🚨 CRITICAL: Stockout Imminent"
            action = f"EMERGENCY REORDER: {reorder_point - current_stock} units"
        elif current_stock <= reorder_point:
            status = "⚠️ WARNING: Low Inventory"
            action = f"STANDARD REORDER: {reorder_point} units"
        elif days_of_supply > 90:
            status = "🧊 OVERSTOCK: Capital Tied Up"
            action = "HALT ORDERS: Run discount promotions"
        else:
            status = "✅ HEALTHY: Optimal Stock"
            action = "NO ACTION REQUIRED"

        return pd.Series([reorder_point, days_of_supply, status, action])


print("⚙️ Compiling Inventory Rules and Formulas...")
engine = InventoryOptimizer()

# --- 4. Run the Engine on Current Warehouse Data ---
print("🔍 Auditing Current Warehouse Inventory...")

# Apply the engine logic to dataset
df[['Reorder_Point', 'Days_of_Supply_Left', 'Health_Status', 'Recommended_Action']] = df.apply(
    lambda row: engine.evaluate_stock(
        row['Current_Stock'],
        row['Avg_Daily_Demand'],
        row['Replenish_Lead_Time_Days'],
        row['Safety_Stock_Threshold']
    ), axis=1
)

# --- 5. Generate Warehouse Report
print("\n" + "=" * 60)
print(" 📋 DAILY INVENTORY ACTION REPORT (URGENT ALERTS ONLY)")
print("=" * 60)

urgent_items = df[df['Health_Status'].str.contains('CRITICAL|WARNING|OVERSTOCK')]

if urgent_items.empty:
    print("All inventory levels are perfectly optimized. No action needed.")
else:
    for _, row in urgent_items.iterrows():
        print(f"\n📦 Product: {row['Product Name']}")
        print(f"   Status:  {row['Health_Status']}")
        print(f"   Stock:   {row['Current_Stock']} units ({row['Days_of_Supply_Left']} days left)")
        print(f"   ACTION:  👉 {row['Recommended_Action']}")

# --- 6. Save the Engine & Report ---
engine_path = os.path.join(MODELS_DIR, 'inventory_engine.pkl')
report_path = os.path.join(BASE_DIR, 'data', 'processed', 'final_inventory_report.csv')

# Save the Python class so the Web API can use it for real-time calculations later
with open(engine_path, 'wb') as f:
    pickle.dump(engine, f)

# Save the df to display tables
df.to_csv(report_path, index=False)

print("\n" + "=" * 60)
print(f"💾 SUCCESS! Inventory Engine saved to: {engine_path}")
print(f"📊 SUCCESS! Master Report saved to: {report_path}")