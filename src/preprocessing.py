import pandas as pd
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt
import warnings

warnings.filterwarnings('ignore')

# --- 1. System Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'nassau_candy.csv')
ZIPS_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'us_zips.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("🚀 Starting Data Preprocessing Pipeline...")


# --- 2. Helper Function: Haversine Distance ---
def haversine(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = min(1.0, max(0.0, sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2))
    c = 2 * asin(sqrt(a))
    return 3958.8 * c


# --- 3. Load Raw Data ---
df = pd.read_csv(RAW_DATA_PATH)
zips_df = pd.read_csv(ZIPS_DATA_PATH)

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y', dayfirst=True)

# ==========================================
# DATASET 1: Shipping Optimization (With Geospatial Math)

print("📦 Calculating Geospatial Distances for Shipping Model...")

# Hardcoded Mappings
product_factory_map = {
    'Wonka Bar - Nutty Crunch Surprise': "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows': "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious': "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate': "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel': "Wicked Choccy's",
    'Laffy Taffy': 'Sugar Shack',
    'SweeTARTS': 'Sugar Shack',
    'Nerds': 'Sugar Shack',
    'Fun Dip': 'Sugar Shack',
    'Fizzy Lifting Drinks': 'Sugar Shack',
    'Everlasting Gobstopper': 'Secret Factory',
    'Hair Toffee': 'The Other Factory',
    'Lickable Wallpaper': 'Secret Factory',
    'Wonka Gum': 'Secret Factory',
    'Kazookles': 'The Other Factory'
}

factory_coords = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107)
}

# Apply Mappings
shipping_df = df.copy()
shipping_df['Origin_Factory'] = shipping_df['Product Name'].map(product_factory_map)
shipping_df['Factory_Lat'] = shipping_df['Origin_Factory'].map(lambda x: factory_coords.get(x, (None, None))[0])
shipping_df['Factory_Long'] = shipping_df['Origin_Factory'].map(lambda x: factory_coords.get(x, (None, None))[1])

# Merge Zips
shipping_df['Postal Code'] = shipping_df['Postal Code'].astype(str).str.zfill(5)
zips_df['zip'] = zips_df['zip'].astype(str).str.zfill(5)
zips_coords = zips_df[['zip', 'latitude', 'longitude']].drop_duplicates(subset=['zip'])

shipping_df = shipping_df.merge(zips_coords, left_on='Postal Code', right_on='zip', how='left')
shipping_df.rename(columns={'latitude': 'Customer_Lat', 'longitude': 'Customer_Long'}, inplace=True)

# Calculate Distance & Target
shipping_df['Distance_Miles'] = np.vectorize(haversine)(shipping_df['Factory_Lat'], shipping_df['Factory_Long'],
                                                        shipping_df['Customer_Lat'], shipping_df['Customer_Long'])
shipping_df['Distance_Miles'] = shipping_df['Distance_Miles'].fillna(shipping_df['Distance_Miles'].median())
shipping_df['Delivery_Time_Days'] = (shipping_df['Ship Date'] - shipping_df['Order Date']).dt.days

# needed columns
final_shipping_df = shipping_df[
    ['Ship Mode', 'Region', 'Origin_Factory', 'Distance_Miles', 'Units', 'Delivery_Time_Days']]


final_shipping_df = final_shipping_df[final_shipping_df['Delivery_Time_Days'] >= 0]

# Save to processed folder
final_shipping_df.to_csv(os.path.join(PROCESSED_DIR, 'model1_shipping_data.csv'), index=False)

# ==========================================
# DATASET 2: Demand Forecasting Model

print("📈 Generating Demand Forecasting dataset...")
# For forecasting, we need Time-Series data. We group by Date and Product to see daily demand.
demand_df = df.groupby(['Order Date', 'Product Name', 'Region'])['Units'].sum().reset_index()


demand_df = demand_df.sort_values(by='Order Date')

# Save to processed folder
demand_df.to_csv(os.path.join(PROCESSED_DIR, 'model2_demand_data.csv'), index=False)

# ==========================================
# DATASET 3: Inventory Optimization

print("🏭 Generating Inventory Optimization dataset...")
inventory_list = []
np.random.seed(42)  # For reproducible results

# Calculated historical daily demand for each product
for product in df['Product Name'].unique():
    product_data = df[df['Product Name'] == product]

    # Calculated average daily units sold
    total_days = max(1, (product_data['Order Date'].max() - product_data['Order Date'].min()).days)
    daily_demand = product_data['Units'].sum() / total_days


    # 1. Lead Time
    replenish_lead_time = np.random.randint(7, 15)

    # 2. Safety Stock
    safety_stock = int(daily_demand * replenish_lead_time * 1.5)

    # 3. Current Stock
    current_stock = int(np.random.uniform(safety_stock * 0.5, daily_demand * 90))

    inventory_list.append({
        'Product Name': product,
        'Avg_Daily_Demand': round(daily_demand, 2),
        'Replenish_Lead_Time_Days': replenish_lead_time,
        'Safety_Stock_Threshold': safety_stock,
        'Current_Stock': current_stock
    })

# Save to processed folder
pd.DataFrame(inventory_list).to_csv(os.path.join(PROCESSED_DIR, 'model3_inventory_data.csv'), index=False)

print("🎉 Preprocessing Complete! All 3 datasets are saved.")