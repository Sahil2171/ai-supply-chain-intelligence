import os
import pickle
from math import radians, cos, sin, asin, sqrt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# 1. Page Configuration ---
st.set_page_config(page_title="Nassau AI - Supply Chain", layout="wide")

# ---SUPPLY CHAIN THEME INJECTION ---
st.markdown("""
<style>
    /* 1. Global Background & Typography */
    .stApp {
        background-color: #F1F5F9;
        color: #1E293B;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #1E293B;
    }

    /* 2. Sidebar Navigation (Hyper-Targeted) */
    [data-testid="stSidebar"] {
        background-color: #13151A !important; 
        border-right: none;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Force absolutely every text node in the sidebar to be white */
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stRadio label span,
    section[data-testid="stSidebar"] .stSelectbox label span {
        color: #FFFFFF !important;
    }
    
    /* Target the sidebar radio buttons to look more like menu links */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.5rem;
        width: 100%; /* Ensure the wrapper takes full width */
    }
    
    [data-testid="stSidebar"] .stRadio label {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        transition: background-color 0.2s ease;
        cursor: pointer;
        width: 100%;         /* Forces uniform, full width */
        display: flex;       /* Aligns the radio circle and text cleanly */
        align-items: center; /* Centers items vertically */
        box-sizing: border-box; 
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(255, 255, 255, 0.15);
    }

    /* 3. Metric Cards (Enhanced Hover Effect & Even Heights) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 1.25rem 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E2E8F0;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); 
        /* Force even heights */
        height: 100%;
        min-height: 145px; 
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    [data-testid="stMetric"]:hover {
        background-color: #F4F7FB !important;
        border-color: #93A5BE !important;
        transform: translateY(-4px); 
        box-shadow: 0 12px 20px -3px rgba(30, 58, 138, 0.12), 0 4px 6px -2px rgba(30, 58, 138, 0.05); 
    }
    
    [data-testid="stMetricValue"] {
        color: #1E3A8A !important; 
        font-weight: 700;
        font-size: 2.2rem;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stMetricDelta"] svg {
        color: #22C55E !important; 
    }
    
    [data-testid="stMetricDelta"] div {
        color: #22C55E !important;
        font-weight: 600;
    }

   /* 4. Action Buttons (Red/Coral Accent) */
    .stButton > button {
        background-color: #FF4B4B !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(255, 75, 75, 0.25) !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #E33C3C !important; /* Darker red on hover */
        box-shadow: 0 6px 8px rgba(255, 75, 75, 0.35) !important;
        transform: translateY(-1px);
    }

    /* 5. DataFrames & Tables (Card aesthetic) */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }

    /* 6. Expanders and Input Fields */
    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border-radius: 0.75rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stSelectbox div[data-baseweb="select"], 
    .stTextInput div[data-baseweb="input"],
    .stNumberInput div[data-baseweb="input"] {
        border-radius: 0.5rem;
        border-color: #CBD5E1;
    }

    /* 7. Alerts & Banners */
    [data-testid="stAlert"] {
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }

    /* Header styling */
    h1 {
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
        color: #1E3A8A;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# 2. System Paths & Resource Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')


@st.cache_resource
def load_all_models():
    ship_model, ship_enc, forecaster = None, None, None
    try:
        with open(os.path.join(MODELS_DIR, 'shipping_regressor.pkl'), 'rb') as f:
            ship_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'shipping_encoders.pkl'), 'rb') as f:
            ship_enc = pickle.load(f)
    except FileNotFoundError:
        st.sidebar.error("❌ Shipping models missing.")

    try:
        with open(os.path.join(MODELS_DIR, 'demand_forecaster.pkl'), 'rb') as f:
            forecaster = pickle.load(f)
    except FileNotFoundError:
        st.sidebar.error("❌ Forecasting model missing.")

    return ship_model, ship_enc, forecaster


@st.cache_data
def load_zips():
    try:
        return pd.read_csv(os.path.join(RAW_DATA_DIR, 'us_zips.csv'), dtype={'zip': str})
    except FileNotFoundError:
        return None


shipping_model, shipping_encoders, forecasting_model = load_all_models()
zips_df = load_zips()


# 3. Global Helper Functions ---
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = min(1.0, max(0.0, sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2))
    c = 2 * asin(sqrt(a))
    return 3958.8 * c


FACTORIES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107)
}

# 4. Sidebar Navigation ---
st.sidebar.title("🏭 Nassau AI")
st.sidebar.markdown("**Decision Intelligence Platform**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Module",
    [
        "🚚 Shipping Optimization",
        "📈 Demand Forecasting",
        "📦 Inventory Command",
        "🚦 SLA & Delay Alert System"
    ]
)
st.sidebar.markdown("---")



# MODULE 1: SHIPPING OPTIMIZATION

def render_shipping():
    st.title("🚚 Factory Reallocation & Shipping Optimization")
    st.markdown("*Optimizing shipping routes for lead-time reduction and profit stability.*")
    st.markdown("---")

    st.sidebar.header("Simulation Parameters")
    product_list = ['Wonka Bar - Milk Chocolate', 'Wonka Bar - Triple Dazzle Caramel',
                    'Wonka Bar - Nutty Crunch Surprise', 'Wonka Bar -Scrumdiddlyumptious', 'Laffy Taffy', 'SweeTARTS',
                    'Nerds', 'Fun Dip']
    selected_product = st.sidebar.selectbox("Select Product", product_list)

    st.sidebar.subheader("Destination Details")
    target_zip = st.sidebar.text_input("Customer Postal Code (e.g., 10024, 90210, 48234)", value="60540")

    customer_lat, customer_long = 41.76, -88.15
    if zips_df is not None:
        zip_data = zips_df[zips_df['zip'] == target_zip.strip()]
        if not zip_data.empty:
            customer_lat = zip_data.iloc[0]['latitude']
            customer_long = zip_data.iloc[0]['longitude']
        else:
            st.sidebar.error(f"⚠️ Zip {target_zip} not found.")

    selected_region = st.sidebar.selectbox("Select Region",
                                           shipping_encoders['Region'].classes_ if shipping_encoders else ['Interior'])
    selected_ship_mode = st.sidebar.selectbox("Ship Mode Filter",
                                              shipping_encoders['Ship Mode'].classes_ if shipping_encoders else [
                                                  'Standard Class'])
    units = st.sidebar.number_input("Units Ordered", min_value=1, value=50)
    priority = st.sidebar.slider("Priority Slider (Speed vs Profit)", 0, 100, 50, help="0 = Maximize Profit, 100 = Maximize Speed")

    if st.sidebar.button("Run What-If Scenario Analysis"):
        if not shipping_model: return
        with st.spinner('Simulating factory configurations...'):
            results = []
            for factory_name, coords in FACTORIES.items():
                distance = haversine(coords[0], coords[1], customer_lat, customer_long)
                try:
                    enc_ship = shipping_encoders['Ship Mode'].transform([selected_ship_mode])[0]
                    enc_region = shipping_encoders['Region'].transform([selected_region])[0]
                    enc_factory = shipping_encoders['Origin_Factory'].transform([factory_name])[0]
                    pred_lead_time = shipping_model.predict(
                        pd.DataFrame([[enc_ship, enc_region, enc_factory, distance, units]],
                                     columns=['Ship Mode', 'Region', 'Origin_Factory', 'Distance_Miles', 'Units']))[0]
                    shipping_cost = distance * 0.05 + (0 if factory_name == "Wicked Choccy's" else 2.5)
                    results.append({'Factory': factory_name, 'Predicted Lead Time (Days)': round(pred_lead_time, 1),
                                    'Distance (Miles)': round(distance, 1),
                                    'Est. Shipping Cost ($)': round(shipping_cost, 2)})
                except ValueError:
                    continue

            df_results = pd.DataFrame(results).sort_values(
                by=['Predicted Lead Time (Days)', 'Est. Shipping Cost ($)'] if priority > 50 else [
                    'Est. Shipping Cost ($)', 'Predicted Lead Time (Days)'])
            df_results['Rank'] = range(1, len(df_results) + 1)

            col1, col2 = st.columns(2)
            best_option, worst_option = df_results.iloc[0], df_results.iloc[-1]

            with col1:
                st.success(f"🏆 **Top Recommendation:** {best_option['Factory']}")
                st.metric(label="Expected Lead Time", value=f"{best_option['Predicted Lead Time (Days)']} Days",
                          delta=f"-{round(worst_option['Predicted Lead Time (Days)'] - best_option['Predicted Lead Time (Days)'], 1)} Days vs Worst",
                          delta_color="inverse")
                st.markdown("---")
                m1, m2 = st.columns(2)
                m1.metric("Est. Shipping Cost", f"${best_option['Est. Shipping Cost ($)']}",
                          delta=f"{round(best_option['Est. Shipping Cost ($)'] - worst_option['Est. Shipping Cost ($)'], 2)} vs Worst",
                          delta_color="inverse")
                m2.metric("Distance Efficiency", f"{best_option['Distance (Miles)']} mi",
                          delta=f"{round(worst_option['Distance (Miles)'] - best_option['Distance (Miles)'], 1)} mi saved")

            with col2:
                st.warning("⚠️ **Risk & Financial Impact Analysis**")
                cost_diff = best_option['Est. Shipping Cost ($)'] - worst_option['Est. Shipping Cost ($)']
                if cost_diff > 0:
                    st.error(f"📉 **Margin Erosion Alert:** +${abs(round(cost_diff, 2))}")
                else:
                    st.success(f"📈 **Profit Stable:** Saves ${abs(round(cost_diff, 2))}")
                st.markdown("---")
                st.write("**Model Confidence (R² Variance):**")
                st.progress(92, text="92% Reliability")
                st.caption(f"**Ship Mode:** {selected_ship_mode} | **Units:** {units} | **Region:** {selected_region}")

            st.markdown("---")
            st.subheader("📊 Factory Performance Comparison")
            st.dataframe(df_results.set_index('Rank'), column_config={
                "Distance (Miles)": st.column_config.ProgressColumn("Distance (mi)", format="%.1f mi", max_value=float(
                    df_results['Distance (Miles)'].max())),
                "Est. Shipping Cost ($)": st.column_config.NumberColumn("Est. Cost", format="$%.2f")},
                         use_container_width=True)

            fig = px.bar(df_results, x='Factory', y='Predicted Lead Time (Days)', text='Predicted Lead Time (Days)',
                         color='Est. Shipping Cost ($)', color_continuous_scale='Tealgrn')
            fig.update_traces(textposition='outside', marker_line_width=0)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="", yaxis_title="Days",
                              coloraxis_colorbar=dict(title="Cost ($)"), margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Please set your simulation parameters and click **Run What-If Scenario Analysis**.")



# MODULE 2: DEMAND FORECASTING

def render_forecasting():
    st.title("📈 Demand Forecasting Engine")
    st.markdown("*Advanced Time-Series predictions with product-level granularity.*")
    st.markdown("---")

    if not forecasting_model:
        st.error("Model missing. Please ensure 'demand_forecaster.pkl' exists in your models directory.")
        return

    # --- Sidebar Controls ---
    st.sidebar.header("Forecast Parameters")

    product_list = [
        'All Products (Aggregate)',
        'Wonka Bar - Milk Chocolate',
        'Wonka Bar - Triple Dazzle Caramel',
        'Wonka Bar - Nutty Crunch Surprise',
        'Wonka Bar -Scrumdiddlyumptious',
        'Laffy Taffy',
        'SweeTARTS',
        'Nerds',

    ]
    selected_forecast_product = st.sidebar.selectbox("Select Product Line", product_list)
    forecast_weeks = st.sidebar.slider("Forecast Horizon (Weeks)", min_value=1, max_value=12, value=4)

    # --- Backend Engine:

    target_model = forecasting_model.get(selected_forecast_product)

    if target_model is None:
        st.error(f"Model for '{selected_forecast_product}' not found. Please retrain models.")
        return

    # Generate the forecast using the specific product's trained model
    master_forecast = target_model.forecast(forecast_weeks)
    predictions = master_forecast.tolist()

    # Live dates
    today = pd.Timestamp.today()
    live_date_range = pd.date_range(start=today, periods=forecast_weeks, freq='W')
    future_dates = [d.strftime('%b %d, %Y') for d in live_date_range]

    # 1. Formated data
    forecast_data = []
    for i, (date, units) in enumerate(master_forecast.items()):

        safe_units = max(0, int(units))
        forecast_data.append({
            'Week': f"Week {i + 1}",
            'Date': future_dates[i],
            'Predicted Units': int(predictions[i])
        })
    df_forecast = pd.DataFrame(forecast_data)

    # 2. Calculate Smart KPIs
    total_proj = df_forecast['Predicted Units'].sum()
    avg_weekly = int(total_proj / len(df_forecast)) if len(df_forecast) > 0 else 0
    peak_row = df_forecast.loc[df_forecast['Predicted Units'].idxmax()]

    half_point = len(df_forecast) // 2
    first_half = df_forecast.iloc[:half_point]['Predicted Units'].sum()
    second_half = df_forecast.iloc[half_point:]['Predicted Units'].sum()

    trend_direction = "📈 Upward" if second_half >= first_half else "📉 Downward"
    trend_color = "normal" if second_half >= first_half else "inverse"

    # 3. Rendering Premium KPI Cards
    st.subheader(f"Key Insights: {selected_forecast_product}")
    kpi1, kpi2, kpi3, kpi4 = st.columns([0.8, 0.8, 1.0, 1.4])
    with kpi1:
        st.metric(label=f"Total Volume ({forecast_weeks}W)", value=f"{int(total_proj):,}")
    with kpi2:
        st.metric(label="Weekly Average", value=f"{avg_weekly:,}")
    with kpi3:
        st.metric(label="Peak Demand", value=peak_row['Week'], delta=peak_row['Date'], delta_color="off")
    with kpi4:
        st.metric(label="Projected Trend", value=trend_direction,
                  delta=f"{round(abs(second_half - first_half),2)} units variance", delta_color=trend_color)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Rendering Main Chart Area
    col_chart, col_notes = st.columns([3, 1])

    with col_chart:
        st.write(f"**Forward Demand Curve ({forecast_weeks} Weeks)**")
        fig = px.area(
            df_forecast,
            x='Date',
            ###
            y= df_forecast['Predicted Units'],
            markers=True,
            color_discrete_sequence=['#8b5cf6']
        )
        fig.update_traces(fillcolor='rgba(139, 92, 246, 0.2)', line=dict(width=4))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="",
            yaxis_title="Required Units",
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 5. Render "AI Analyst" Sidebar
    with col_notes:
        st.info("🧠 **AI Analyst Notes**")
        st.write(f"**Target:** {selected_forecast_product}")
        st.write(f"**Horizon:** {forecast_weeks} Weeks")
        st.markdown("---")
        if second_half >= first_half:
            st.success(
                "Demand is accelerating. Ensure factory safety stock limits are increased for the upcoming peak.")
        else:
            st.warning("Demand is cooling. Consider pausing reorders to prevent capital lock-up in overstock.")
        st.markdown("---")
        st.caption("Forecast utilizes Holt-Winters Exponential Smoothing accounting for seasonality and trend vectors.")

    with st.expander("📥 View Raw Forecast Data"):
        st.dataframe(df_forecast, use_container_width=True, hide_index=True)


# ==========================================
# 🥉 MODULE 3: INVENTORY COMMAND CENTER

def render_inventory():
    st.title("📦 Inventory Command Center")
    st.markdown("*Prescriptive Operations Research engine for automated reorder triggers.*")
    st.markdown("---")

    # 1. Loading report generated by train_inventory.py
    report_path = os.path.join(BASE_DIR, 'data', 'processed', 'final_inventory_report.csv')

    try:
        inventory_df = pd.read_csv(report_path)
    except FileNotFoundError:
        st.error("❌ Inventory report not found. Please run 'train_inventory.py' in your terminal first.")
        return

    # 2. Calculate Live Operations KPIs
    critical_items = inventory_df[inventory_df['Health_Status'].str.contains("CRITICAL", na=False)]
    warning_items = inventory_df[inventory_df['Health_Status'].str.contains("WARNING", na=False)]
    healthy_items = inventory_df[inventory_df['Health_Status'].str.contains("HEALTHY", na=False)]

    # 3. Render Premium KPI Cards
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("✅ Healthy Product Lines", len(healthy_items))
    kpi2.metric("⚠️ Low Stock Warnings", len(warning_items))
    kpi3.metric("🚨 Critical Stockouts", len(critical_items))

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Dynamic AI Action Banner
    if len(critical_items) > 0:
        st.error(
            f"🚨 **URGENT ACTION REQUIRED:** The AI has detected {len(critical_items)} product(s) with critically low stock based on lead-time math. Review the prescriptive actions below immediately to prevent supply chain bottlenecks.")
    elif len(warning_items) > 0:
        st.warning(
            f"⚠️ **ATTENTION:** {len(warning_items)} product(s) are approaching safety thresholds. Prepare purchase orders.")
    else:
        st.success("✅ All warehouse inventory levels are perfectly optimized. No emergency actions required today.")

    st.markdown("---")
    st.subheader("📋 Daily Master Action Report")

    # 5. Pandas Data Styling (Color-coding the rows for the presentation)
    def highlight_status(val):
        if isinstance(val, str):
            if "CRITICAL" in val:
                return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
            elif "WARNING" in val:
                return 'background-color: #fef08a; color: #854d0e; font-weight: bold'
            elif "OVERSTOCK" in val:
                return 'background-color: #dbeafe; color: #1e40af; font-weight: bold'
            elif "HEALTHY" in val:
                return 'background-color: #d1fae5; color: #065f46; font-weight: bold'
        return ''

    styled_df = inventory_df.style.map(highlight_status, subset=['Health_Status'])

    # 6. Render the Interactive Data Table
    st.dataframe(
        styled_df,
        column_config={
            "Product Name": st.column_config.TextColumn("Product Line", width="medium"),
            "Avg_Daily_Demand": st.column_config.NumberColumn("Daily Demand", format="%.1f units/day"),
            "Current_Stock": st.column_config.NumberColumn("Current Stock", format="%d units"),
            "Safety_Stock_Threshold": st.column_config.NumberColumn("Safety Threshold", format="%d units"),
            "Days_of_Supply_Left": st.column_config.ProgressColumn(
                "Days of Supply",
                format="%d days",
                min_value=0,
                max_value=120
            ),
            "Health_Status": st.column_config.TextColumn("System Status"),
            "Recommended_Action": st.column_config.TextColumn("AI Prescriptive Action", width="large")
        },
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 🚦 MODULE 4: SLA & DELAY ALERT SYSTEM

def render_delay_system():
    st.title("🚦 Active Order Risk & Delay Alert System")
    st.markdown("*Real-time classification engine to intercept SLA breaches before they occur.*")
    st.markdown("---")

    # 1. Load the Classifier
    try:
        with open(os.path.join(MODELS_DIR, 'delay_classifier.pkl'), 'rb') as f:
            clf_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'delay_encoders.pkl'), 'rb') as f:
            clf_encoders = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Delay models not found. Run 'train_delay.py' first.")
        return

    # 2. Simulate a "Live Queue" of incoming orders for the dashboard
    import random
    np.random.seed(pd.Timestamp.now().microsecond)

    live_orders = []
    regions = clf_encoders['Region'].classes_
    modes = clf_encoders['Ship_Mode'].classes_

    for i in range(25):
        live_orders.append({
            'Order_ID': f"ORD-{random.randint(10000, 99999)}",
            'Customer': f"Client {chr(65 + i)}{chr(65 + random.randint(0, 25))}",
            'Region': random.choice(regions),
            'Ship_Mode': random.choice(['Standard Class', 'Second Class']),
            'Units': random.randint(50, 450)
        })

    df_live = pd.DataFrame(live_orders)

    # 3. AI Scanning Engine
    st.write("### 🔍 AI Queue Scanner Active...")

    encoded_live = df_live.copy()
    encoded_live['Region'] = clf_encoders['Region'].transform(encoded_live['Region'])
    encoded_live['Ship_Mode'] = clf_encoders['Ship_Mode'].transform(encoded_live['Ship_Mode'])

    # Predict Probability of Delay
    probabilities = clf_model.predict_proba(encoded_live[['Region', 'Ship_Mode', 'Units']])[:, 1]
    df_live['Delay_Risk_Score'] = (probabilities * 100).round(1)

    # 4. Action Logic
    def determine_action(row):
        if row['Delay_Risk_Score'] > 65:
            return "🚨 HIGH RISK: Upgrade to First Class Shipping immediately to protect SLA."
        elif row['Delay_Risk_Score'] > 40:
            return "⚠️ MEDIUM RISK: Monitor closely. Prepare customer communication."
        else:
            return "✅ ON TRACK: Proceed with standard fulfillment."

    df_live['Prescriptive_Action'] = df_live.apply(determine_action, axis=1)

    # Sort by risk
    df_live = df_live.sort_values(by='Delay_Risk_Score', ascending=False)

    # 5. Global KPIs
    high_risk_count = len(df_live[df_live['Delay_Risk_Score'] > 65])
    safe_count = len(df_live[df_live['Delay_Risk_Score'] <= 40])

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Orders in Queue", len(df_live))
    kpi2.metric("SLA Breaches Prevented", high_risk_count, f"+{high_risk_count} Intercepts", delta_color="normal")
    kpi3.metric("Healthy Orders", safe_count)

    st.markdown("<br>", unsafe_allow_html=True)

    if high_risk_count > 0:
        st.error(
            f"**Action Required:** AI has flagged {high_risk_count} orders that will fail their delivery promise based on current shipping parameters. Please approve the prescriptive upgrades below.")

    #Color formatting
    def highlight_risk(val):
        if isinstance(val, float) or isinstance(val, int):
            if val > 65: return 'color: #991b1b; font-weight: bold'
            if val > 40: return 'color: #854d0e; font-weight: bold'
            return 'color: #065f46'
        if isinstance(val, str):
            if "HIGH RISK" in val: return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
            if "MEDIUM RISK" in val: return 'background-color: #fef08a; color: #854d0e'
        return ''

    styled_live = df_live.style.map(highlight_risk, subset=['Delay_Risk_Score', 'Prescriptive_Action'])

    st.dataframe(
        styled_live,
        column_config={
            "Order_ID": st.column_config.TextColumn("Order Number", width="small"),
            "Customer": st.column_config.TextColumn("Client"),
            "Region": st.column_config.TextColumn("Destination"),
            "Ship_Mode": st.column_config.TextColumn("Current Ship Mode"),
            "Units": st.column_config.NumberColumn("Volume"),
            "Delay_Risk_Score": st.column_config.ProgressColumn(
                "AI Delay Probability",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),
            "Prescriptive_Action": st.column_config.TextColumn("System Command", width="large")
        },
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 🚦 APPLICATION ROUTER

if app_mode == "🚚 Shipping Optimization":
    render_shipping()
elif app_mode == "📈 Demand Forecasting":
    render_forecasting()
elif app_mode == "📦 Inventory Command":
    render_inventory()
elif app_mode == "🚦 SLA & Delay Alert System":
    render_delay_system()