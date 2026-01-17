import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Import InventoryOptimizer for inventory optimization functionality
from inventory_optimization import InventoryOptimizer

# Page configuration
st.set_page_config(
    page_title="XLPE Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔮 XLPE Demand Forecasting & Inventory Optimization Dashboard</div>', 
            unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Go to", 
                        ["Overview", "Model Performance", "Inventory Optimization", "Forecasting Tool", "About"])

st.sidebar.markdown("---")
st.sidebar.info("""
**ARABCAB AI Competition**  
AI-Based Demand Forecasting for Cable Industry  
Egypt • Bahrain • UAE
""")

# Load data
@st.cache_data
def load_data():
    try:
        # Model results
        model_results = pd.read_csv('outputs/model_comparison_results.csv')
        
        # Inventory optimization
        with open('outputs/inventory_optimization_results.json', 'r') as f:
            inventory_results = json.load(f)
        
        # Inventory forecast
        inventory_forecast = pd.read_csv('outputs/inventory_forecast_12months.csv')
        
        # Metadata
        with open('outputs/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Best pipeline (updated from best_model.pkl)
        with open('outputs/best_pipeline.pkl', 'rb') as f:
            best_model = pickle.load(f)
        
        # Historical data
        historical_data = pd.read_excel('historical_xlpe_demand.xlsx')
        
        return {
            'model_results': model_results,
            'inventory_results': inventory_results,
            'inventory_forecast': inventory_forecast,
            'metadata': metadata,
            'best_model': best_model,
            'historical_data': historical_data
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run Models.py and inventory_optimization.py first to generate required files.")
        return None

data = load_data()

if data is None:
    st.stop()

# ====================
# PAGE: OVERVIEW
# ====================
if page == "Overview":
    st.header("📊 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Best Model",
            value=data['metadata']['best_model'],
            delta="Selected"
        )
    
    with col2:
        st.metric(
            label="Model Accuracy",
            value=f"{data['metadata']['best_forecast_accuracy']:.2f}%",
            delta="High Performance"
        )
    
    with col3:
        st.metric(
            label="Service Level",
            value=f"{data['inventory_results']['service_level_percent']:.1f}%",
            delta="Target Met"
        )
    
    with col4:
        annual_cost = data['inventory_results']['annual_costs']['total_cost']
        st.metric(
            label="Annual Inventory Cost",
            value=f"${annual_cost/1000000:.2f}M",
            delta="Optimized"
        )
    
    st.markdown("---")
    
    # Key insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Findings")
        st.markdown(f"""
        - **{data['metadata']['train_size']}** training samples used
        - **{data['metadata']['test_size']}** testing samples for validation
        - **{len(data['metadata']['features'])}** predictive features
        - **12-month** demand forecast generated
        - **EOQ Strategy** implemented for cost optimization
        """)
    
    with col2:
        st.subheader("💡 Business Impact")
        st.markdown("""
        - ✅ Reduced stockout risk to <5%
        - ✅ Optimized inventory holding costs
        - ✅ Improved demand forecasting accuracy
        - ✅ Data-driven procurement decisions
        - ✅ Enhanced supply chain resilience
        """)
    
    st.markdown("---")
    
    # Historical demand trend
    st.subheader("📈 Historical XLPE Demand Trend")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['historical_data']['Date'],
        y=data['historical_data']['xlpe_demand_Million_tons'],
        mode='lines+markers',
        name='XLPE Demand',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="XLPE Demand Over Time",
        xaxis_title="Date",
        yaxis_title="Demand (Million Tons)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ====================
# PAGE: MODEL PERFORMANCE
# ====================
elif page == "Model Performance":
    st.header("🤖 Model Performance Analysis")
    
    # Model comparison
    st.subheader("📊 Model Accuracy Comparison")
    
    fig = px.bar(
        data['model_results'],
        x='Model',
        y='Accuracy (%)',
        color='Model',
        text='Accuracy (%)',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis_range=[0, 100]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    st.subheader("📈 Detailed Model Metrics")
    st.dataframe(
        data['model_results'].style.highlight_max(axis=0, subset=['Accuracy (%)', 'R² Score'])
                                   .highlight_min(axis=0, subset=['MAE (million tons)', 'MAPE (%)']),
        use_container_width=True
    )
    
    # Best model details
    st.markdown("---")
    st.subheader(f"🏆 Best Model: {data['metadata']['best_model']}")
    
    col1, col2, col3 = st.columns(3)
    
    best_model_data = data['model_results'][
        data['model_results']['Model'] == data['metadata']['best_model']
    ].iloc[0]
    
    with col1:
        st.metric("Accuracy", f"{best_model_data['Accuracy (%)']:.2f}%")
    with col2:
        st.metric("MAE", f"{best_model_data['MAE (million tons)']:.6f} M tons")
    with col3:
        st.metric("R² Score", f"{best_model_data['R² Score']:.4f}")
    
    # Feature importance (if available)
    if os.path.exists('outputs/feature_importance.csv'):
        st.markdown("---")
        st.subheader("🎯 Feature Importance")
        
        feature_imp = pd.read_csv('outputs/feature_importance.csv')
        
        fig = px.bar(
            feature_imp,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ====================
# PAGE: INVENTORY OPTIMIZATION
# ====================

if page == "Inventory Optimization":
    st.header("📦 Interactive Inventory Optimization Scenario Tool")
    # User parameters
    service_level = st.slider("Target Service Level (%)", min_value=80, max_value=99, value=95)
    lead_time_days = st.number_input("Lead Time (days)", value=30)
    holding_cost = st.number_input("Holding Cost Per Ton ($)", value=50)
    ordering_cost = st.number_input("Ordering Cost Per Order ($)", value=5000)
    stockout_cost = st.number_input("Stockout Cost Per Ton ($)", value=500)
    material_cost = st.number_input("Material Cost Per Ton ($)", value=2000)

    # Calculate avg_demand and std_demand from historical data
    avg_demand = data['historical_data']['xlpe_demand_Million_tons'].mean()
    std_demand = data['historical_data']['xlpe_demand_Million_tons'].std()

    # Pass these into InventoryOptimizer instance
    optimizer = InventoryOptimizer(
        avg_demand=avg_demand,  # from your historical stats
        std_demand=std_demand,
        lead_time_days=lead_time_days,
        service_level=service_level/100.0
    )
    optimizer.holding_cost_per_ton = holding_cost
    optimizer.ordering_cost = ordering_cost
    optimizer.stockout_cost_per_ton = stockout_cost
    optimizer.material_cost_per_ton = material_cost

    safety_stock = optimizer.calculate_safety_stock()
    rop = optimizer.calculate_reorder_point()
    eoq = optimizer.calculate_economic_order_quantity()
    total_cost_dict = optimizer.calculate_total_inventory_cost(eoq, safety_stock)
    total_cost = total_cost_dict['total_cost']

    st.markdown(f"""
    **Calculated Optimization Results:**
    - Safety Stock: `{safety_stock:.2f}` tons
    - Reorder Point: `{rop:.2f}` tons
    - Economic Order Quantity (EOQ): `{eoq:.2f}` tons/order
    - **Total Inventory Cost:** `${total_cost:,.2f}`
    """)

    # Optionally: plot cost curves, let user simulate more scenarios, etc.
# ====================
# PAGE: FORECASTING TOOL
# ====================
elif page == "Forecasting Tool":
    st.header("🔮 Interactive Forecasting Tool")
    
    st.info("Enter economic indicators to predict XLPE demand for the next month.")
    
    # Feature inputs
    features = data['metadata']['features']
    
    st.subheader("📝 Input Economic Indicators")
    
    col1, col2 = st.columns(2)
    
    input_values = {}
    
    # Get latest values as defaults
    historical_clean = data['historical_data'].drop(columns=['Date', 'Year', 'Month']).dropna()
    latest_values = historical_clean.iloc[-1]
    
    for i, feature in enumerate(features):
        default_val = float(latest_values[feature])
        
        if i % 2 == 0:
            with col1:
                input_values[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    value=default_val,
                    format="%.6f",
                    key=feature
                )
        else:
            with col2:
                input_values[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    value=default_val,
                    format="%.6f",
                    key=feature
                )
    
    if st.button("🚀 Generate Forecast", type="primary"):
        # Prepare input
        input_df = pd.DataFrame([input_values])
        
        # Make prediction
        prediction = data['best_model'].predict(input_df)[0]
        
        # Display result
        st.markdown("---")
        st.success("✅ Forecast Generated Successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted XLPE Demand",
                f"{prediction:.6f} M tons",
                delta=f"{prediction*1000:.2f} tons"
            )
        
        with col2:
            avg_demand = historical_clean['xlpe_demand_Million_tons'].mean()
            diff_pct = ((prediction - avg_demand) / avg_demand) * 100
            st.metric(
                "vs Historical Average",
                f"{diff_pct:+.2f}%",
                delta="Comparison"
            )
        
        with col3:
            # Check inventory recommendation
            if prediction < data['inventory_results']['safety_stock_million_tons']:
                recommendation = "✅ Current stock sufficient"
                st.metric("Inventory Status", "SAFE", delta=recommendation)
            elif prediction < data['inventory_results']['reorder_point_million_tons']:
                recommendation = "⚠️ Monitor closely"
                st.metric("Inventory Status", "WATCH", delta=recommendation)
            else:
                recommendation = "🔴 Reorder needed"
                st.metric("Inventory Status", "ORDER", delta=recommendation)

# ====================
# PAGE: ABOUT
# ====================
elif page == "About":
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## ARABCAB Scientific Competition
    ### AI-Based Demand Forecasting & Inventory Optimization
    
    **Competition Focus:** Cable & Metals Industry (Egypt • Bahrain • UAE)
    
    ---
    
    ### 🎯 Project Objectives
    
    This AI-powered solution addresses critical challenges in the cable manufacturing industry:
    
    - **Demand Volatility:** Unpredictable fluctuations in XLPE (Cross-Linked Polyethylene) demand
    - **Price Volatility:** Unstable raw material costs affecting profitability
    - **Inventory Inefficiency:** Balancing overstock and stockout risks
    
    ---
    
    ### 🤖 Technical Approach
    
    **Machine Learning Models:**
    - Linear Regression
    - K-Nearest Neighbors (KNN)
    - Decision Tree Regressor
    - Random Forest Regressor
    
    **Optimization Techniques:**
    - Economic Order Quantity (EOQ)
    - Safety Stock Calculation
    - Reorder Point Optimization
    - Service Level Targeting (95%)
    
    **Features Used:**
    """)
    
    for feature in data['metadata']['features']:
        st.markdown(f"- {feature.replace('_', ' ').title()}")
    
    st.markdown("""
    ---
    
    ### 📊 Deliverables
    
    1. ✅ **Working Code:** Python implementation (Models.py, inventory_optimization.py)
    2. ✅ **Model Outputs:** Accuracy metrics, predictions, visualizations
    3. ✅ **Dashboard:** Interactive Streamlit application
    4. ✅ **Report:** Comprehensive methodology documentation
    
    ---
    
    ### 🏆 Expected Impact
    
    - Reduce inventory costs by 15-25%
    - Decrease stockout incidents by 80%
    - Improve forecasting accuracy to >95%
    - Enable data-driven procurement decisions
    - Strengthen supply chain resilience
    
    ---
    
    ### 👥 Team Information
    
    **University:** The American University in Cairo (AUC)
    
    **Faculty Leads:** Dr. Seif Eldawlatly, Dr. Nouri Sakr
    
    **Student Members:** Salma Waleed Elmara, Marina Nazeh, Mennatallah Zaid, Mariam Abdo, Omr Alhussein
    
    ---
    """)
    
    st.success("🌟 This dashboard represents our team's commitment to solving real-world industry challenges through innovative AI solutions.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>ARABCAB AI Competition 2026</strong></p>
    <p>AI-Based Demand Forecasting & Inventory Optimization</p>
    <p>Egypt • Bahrain • UAE</p>
</div>
""", unsafe_allow_html=True)
