import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

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
        
        # Best model
        with open('outputs/best_model.pkl', 'rb') as f:
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
elif page == "Inventory Optimization":
    st.header("📦 Inventory Optimization Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    inv_results = data['inventory_results']
    
    with col1:
        st.metric(
            "Safety Stock",
            f"{inv_results['safety_stock_tons']:.2f} tons",
            delta="Buffer inventory"
        )
    
    with col2:
        st.metric(
            "Reorder Point",
            f"{inv_results['reorder_point_tons']:.2f} tons",
            delta="Trigger level"
        )
    
    with col3:
        st.metric(
            "EOQ",
            f"{inv_results['eoq_tons']:.2f} tons",
            delta="Order quantity"
        )
    
    with col4:
        st.metric(
            "Max Inventory",
            f"{inv_results['max_inventory_tons']:.2f} tons",
            delta="Peak level"
        )
    
    st.markdown("---")
    
    # Cost analysis
    st.subheader("💰 Annual Cost Analysis")
    
    costs = inv_results['annual_costs']
    cost_df = pd.DataFrame({
        'Cost Type': ['Ordering Cost', 'Holding Cost', 'Material Cost'],
        'Amount (USD)': [costs['ordering_cost'], costs['holding_cost'], costs['material_cost']]
    })
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = px.pie(
            cost_df,
            values='Amount (USD)',
            names='Cost Type',
            title='Cost Breakdown',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Cost Summary")
        st.markdown(f"""
        - **Ordering Cost:** ${costs['ordering_cost']:,.2f}
        - **Holding Cost:** ${costs['holding_cost']:,.2f}
        - **Material Cost:** ${costs['material_cost']:,.2f}
        - **Total Annual Cost:** ${costs['total_cost']:,.2f}
        """)
    
    st.markdown("---")
    
    # 12-month simulation
    st.subheader("📅 12-Month Inventory Simulation")
    
    forecast_df = data['inventory_forecast']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=forecast_df['Month'],
        y=forecast_df['Ending_Inventory'],
        mode='lines+markers',
        name='Inventory Level',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_hline(
        y=inv_results['reorder_point_million_tons'],
        line_dash="dash",
        line_color="red",
        annotation_text="Reorder Point",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=inv_results['safety_stock_million_tons'],
        line_dash="dash",
        line_color="orange",
        annotation_text="Safety Stock",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Projected Inventory Levels - Next 12 Months",
        xaxis_title="Month",
        yaxis_title="Inventory (Million Tons)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Simulation results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Simulation Metrics")
        sim_results = inv_results['simulation_12months']
        st.markdown(f"""
        - **Average Inventory:** {sim_results['avg_inventory_million_tons']:.6f} M tons
        - **Total Orders:** {sim_results['total_orders']}
        - **Total Stockouts:** {sim_results['total_stockouts_million_tons']:.6f} M tons
        - **Service Level:** {sim_results['service_level_achieved']:.2f}%
        """)
    
    with col2:
        st.subheader("📋 Monthly Details")
        st.dataframe(forecast_df, use_container_width=True, height=400)

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
