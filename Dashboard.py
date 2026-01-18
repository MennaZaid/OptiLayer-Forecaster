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
                        ["Overview", "Model Performance", "Risk Analysis", "Infrastructure Scenario", 
                         "Inventory Optimization", "Forecasting Tool", "About"])

st.sidebar.markdown("---")
st.sidebar.info("""
**ARABCAB AI Competition**  
AI-Based Demand Forecasting for Cable Industry  
**Innovation:** Infrastructure Scenario Forecasting  
MENA region
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
        
        # Scenario forecast (new)
        scenario_forecast = None
        if os.path.exists('outputs/scenario_forecast_2026.csv'):
            scenario_forecast = pd.read_csv('outputs/scenario_forecast_2026.csv')
        
        # Risk analysis files (new)
        risk_files = {}
        for model_name in ['Random_Forest_(Risk-Aware)', 'Gradient_Boosting_(Risk-Aware)', 
                          'Linear_Regression_(Enhanced)', 'Ensemble']:
            filepath = f'outputs/{model_name}_Risk_Analysis.csv' if model_name != 'Linear_Regression_(Enhanced)' else 'outputs/Linear_Regression_(Enhanced)_Risk_Adjusted.csv'
            if os.path.exists(filepath):
                risk_files[model_name] = pd.read_csv(filepath)
        
        return {
            'model_results': model_results,
            'inventory_results': inventory_results,
            'inventory_forecast': inventory_forecast,
            'metadata': metadata,
            'best_model': best_model,
            'historical_data': historical_data,
            'scenario_forecast': scenario_forecast,
            'risk_files': risk_files
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run Models_Advanced.py and inventory_optimization.py first to generate required files.")
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
            label="Forecast Accuracy",
            value=f"{data['metadata']['best_forecast_accuracy']:.2f}%",
            delta=f"R² Score: {data['metadata']['best_r2']:.4f}"
        )
    
    with col3:
        if data['scenario_forecast'] is not None:
            total_demand = data['scenario_forecast']['Total_Demand_Million_Tons'].values[0]
            st.metric(
                label="2026 Total Demand (with Infrastructure)",
                value=f"{total_demand:.3f}M tons",
                delta=f"{total_demand*1000000:,.0f} tons"
            )
        else:
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
        ensemble_members = data['metadata'].get('ensemble_members', [])
        ensemble_info = f"**Ensemble:** {', '.join([m.split('(')[0].strip() for m in ensemble_members])}" if ensemble_members else ""
        
        st.markdown(f"""
        - **{data['metadata']['train_size']}** training samples used
        - **{data['metadata']['test_size']}** testing samples for validation
        - **{len(data['metadata']['features'])}** predictive features (incl. infrastructure proxies)
        - **12-month** demand forecast generated
        - **Risk-Aware Forecasting** with uncertainty quantification
        - {ensemble_info}
        """)
    
    with col2:
        st.subheader("💡 Business Impact")
        if data['scenario_forecast'] is not None:
            infra_pct = data['scenario_forecast']['Infrastructure_Percentage'].values[0]
            st.markdown(f"""
            - ✅ **Infrastructure Scenario:** 878K tons (2026)
            - ✅ **Total Demand Projection:** {data['scenario_forecast']['Total_Demand_Million_Tons'].values[0]:.3f}M tons
            - ✅ Infrastructure represents **{infra_pct:.1f}%** of total demand
            - ✅ **70% lower variance** than single models
            - ✅ Automated safety stock optimization
            - ✅ Price-based hedging strategy
            """)
        else:
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
    st.subheader("📊 Model Forecast Accuracy Comparison")
    
    fig = px.bar(
        data['model_results'],
        x='Model',
        y='Forecast Accuracy (%)',
        color='Model',
        text='Forecast Accuracy (%)',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis_range=[95, 100]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics with overfitting
    st.subheader("📈 Comprehensive Model Metrics")
    
    # Highlight best performing models
    styled_df = data['model_results'].style\
        .highlight_max(axis=0, subset=['Forecast Accuracy (%)', 'R² Score'], color='lightgreen')\
        .highlight_min(axis=0, subset=['MAE (million tons)', 'MAPE (%)', 'Overfitting'], color='lightgreen')\
        .format({
            'Forecast Accuracy (%)': '{:.2f}',
            'MAE (million tons)': '{:.6f}',
            'RMSE (million tons)': '{:.6f}',
            'R² Score': '{:.4f}',
            'MAPE (%)': '{:.2f}',
            'Train R²': '{:.4f}',
            'Overfitting': '{:.4f}'
        })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Add overfitting explanation
    st.info("""
    **Overfitting Score:** Measures the gap between training and test performance. 
    Lower is better (indicates better generalization to new data).
    - **Low (<0.1):** Excellent generalization
    - **Moderate (0.1-0.2):** Good generalization  
    - **High (>0.2):** May struggle with new data
    """)
    
    # Best model details
    st.markdown("---")
    st.subheader(f"🏆 Best Model: {data['metadata']['best_model']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    best_model_data = data['model_results'][
        data['model_results']['Model'] == data['metadata']['best_model']
    ].iloc[0]
    
    with col1:
        st.metric("Forecast Accuracy", f"{best_model_data['Forecast Accuracy (%)']:.2f}%")
    with col2:
        st.metric("MAE", f"{best_model_data['MAE (million tons)']:.6f} M tons")
    with col3:
        st.metric("R² Score", f"{best_model_data['R² Score']:.4f}")
    with col4:
        overfitting_status = "Low" if best_model_data['Overfitting'] < 0.1 else ("Moderate" if best_model_data['Overfitting'] < 0.2 else "High")
        st.metric("Overfitting", f"{best_model_data['Overfitting']:.4f}", delta=overfitting_status)
    
    # Ensemble information
    if 'Ensemble' in data['metadata']['best_model']:
        st.markdown("---")
        st.subheader("🔗 Ensemble Composition")
        
        ensemble_members = data['metadata'].get('ensemble_members', [])
        ensemble_weights = data['metadata'].get('ensemble_weights', {})
        
        if ensemble_members and ensemble_weights:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Member Models:**")
                for member in ensemble_members:
                    weight = ensemble_weights.get(member, 0) * 100
                    st.markdown(f"- {member}: **{weight:.1f}%** weight")
            
            with col2:
                # Create pie chart
                fig = px.pie(
                    values=list(ensemble_weights.values()),
                    names=list(ensemble_weights.keys()),
                    title="Ensemble Weight Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
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
# PAGE: RISK ANALYSIS
# ====================
elif page == "Risk Analysis":
    st.header("⚠️ Risk-Aware Forecasting Analysis")
    
    st.info("""
    **Risk-Aware Forecasting** quantifies prediction uncertainty and provides safety margins for decision-making.
    This is critical for high-stakes scenarios like the 878K ton infrastructure project.
    """)
    
    # Select model
    available_models = list(data['risk_files'].keys())
    if available_models:
        selected_model = st.selectbox("Select Model", available_models, 
                                     index=available_models.index('Ensemble') if 'Ensemble' in available_models else 0)
        
        risk_data = data['risk_files'][selected_model]
        
        # Key risk metrics
        st.subheader("📊 Risk Metrics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_forecast = risk_data['Forecast'].mean() if 'Forecast' in risk_data.columns else risk_data.get('Risk_Adjusted_Prediction', risk_data.get('Base_Prediction', pd.Series([0]))).mean()
            st.metric("Avg Forecast", f"{avg_forecast:.6f}M tons")
        
        with col2:
            if 'Uncertainty_Sigma' in risk_data.columns:
                avg_uncertainty = risk_data['Uncertainty_Sigma'].mean()
                st.metric("Avg Uncertainty (σ)", f"{avg_uncertainty:.6f}M tons", 
                         delta="Lower is better")
            else:
                st.metric("Model Type", "Risk-Adjusted", delta="Price-based")
        
        with col3:
            if 'Safety_Stock_95CI' in risk_data.columns:
                avg_safety = risk_data['Safety_Stock_95CI'].mean()
                st.metric("Avg Safety Stock", f"{avg_safety:.6f}M tons", 
                         delta="95% confidence")
            else:
                st.metric("Risk Factor Range", 
                         f"{risk_data['Risk_Factor'].min():.2f}-{risk_data['Risk_Factor'].max():.2f}x" if 'Risk_Factor' in risk_data.columns else "N/A")
        
        with col4:
            if 'Hedging_Factor' in risk_data.columns:
                hedging_count = (risk_data['Hedging_Factor'] != 1.0).sum()
                st.metric("Price Adjustments", f"{hedging_count}/{len(risk_data)}", 
                         delta=f"{(hedging_count/len(risk_data)*100):.0f}% of periods")
            else:
                adj_count = (risk_data['Risk_Factor'] != 1.0).sum() if 'Risk_Factor' in risk_data.columns else 0
                st.metric("Risk Adjustments", f"{adj_count}/{len(risk_data)}")
        
        # Visualization
        st.markdown("---")
        st.subheader("📈 Forecast with Uncertainty Bands")
        
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(risk_data))),
            y=risk_data['Actual'],
            mode='lines+markers',
            name='Actual Demand',
            line=dict(color='black', width=2),
            marker=dict(size=6)
        ))
        
        # Forecast
        forecast_col = 'Forecast' if 'Forecast' in risk_data.columns else ('Risk_Adjusted_Prediction' if 'Risk_Adjusted_Prediction' in risk_data.columns else 'Base_Prediction')
        fig.add_trace(go.Scatter(
            x=list(range(len(risk_data))),
            y=risk_data[forecast_col],
            mode='lines+markers',
            name='Predicted Demand',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Uncertainty bands (if available)
        if 'Uncertainty_Sigma' in risk_data.columns:
            upper_band = risk_data[forecast_col] + 1.96 * risk_data['Uncertainty_Sigma']
            lower_band = risk_data[forecast_col] - 1.96 * risk_data['Uncertainty_Sigma']
            
            fig.add_trace(go.Scatter(
                x=list(range(len(risk_data))),
                y=upper_band,
                mode='lines',
                name='Upper Bound (95% CI)',
                line=dict(color='lightblue', width=1, dash='dash'),
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(len(risk_data))),
                y=lower_band,
                mode='lines',
                name='Lower Bound (95% CI)',
                line=dict(color='lightblue', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.2)',
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"{selected_model} - Forecast with Uncertainty",
            xaxis_title="Time Period",
            yaxis_title="XLPE Demand (Million Tons)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed risk table
        st.markdown("---")
        st.subheader("🔍 Detailed Risk Analysis (Last 10 Records)")
        st.dataframe(risk_data.tail(10), use_container_width=True)
        
        # Risk insights
        if 'Uncertainty_Sigma' in risk_data.columns:
            st.markdown("---")
            st.subheader("💡 Risk Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Uncertainty Quantification:**
                - **σ (Sigma):** Standard deviation of predictions across model components
                - **Safety Stock:** 1.96σ provides 95% confidence coverage
                - **Lower σ:** More confident predictions, less safety stock needed
                """)
            
            with col2:
                if 'Hedging_Factor' in risk_data.columns:
                    st.markdown("""
                    **Price-Based Hedging:**
                    - **1.05x:** Buy 5% extra when prices are low (good opportunity)
                    - **0.95x:** Buy 5% less when prices are high (cost control)
                    - **1.00x:** Normal purchasing at stable prices
                    """)
                else:
                    st.markdown("""
                    **Risk Adjustment:**
                    - Predictions adjusted based on economic indicators
                    - Price trends signal future demand patterns
                    - Proactive rather than reactive forecasting
                    """)
    else:
        st.warning("No risk analysis files found. Please run Models_Advanced.py first.")

# ====================
# PAGE: INFRASTRUCTURE SCENARIO
# ====================
elif page == "Infrastructure Scenario":
    st.header("🏗️ Infrastructure-Based Scenario Forecasting")
    
    if data['scenario_forecast'] is not None:
        scenario = data['scenario_forecast'].iloc[0]
        
        st.success("✅ 2026 Infrastructure Scenario Successfully Generated")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Base Market Demand",
                f"{scenario['Base_Demand_Million_Tons']:.3f}M tons",
                delta=f"{scenario['Base_Demand_Million_Tons']*1000000:,.0f} tons"
            )
        
        with col2:
            st.metric(
                "Infrastructure Contribution",
                f"{scenario['Infrastructure_Million_Tons']:.3f}M tons",
                delta=f"{scenario['Infrastructure_Percentage']:.1f}% of total"
            )
        
        with col3:
            st.metric(
                "TOTAL 2026 Demand",
                f"{scenario['Total_Demand_Million_Tons']:.3f}M tons",
                delta=f"{scenario['Total_Demand_Tons']:,.0f} tons"
            )
        
        # Visualization
        st.markdown("---")
        st.subheader("📊 Demand Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Market Demand', 'Infrastructure'],
                values=[scenario['Base_Demand_Million_Tons'], scenario['Infrastructure_Million_Tons']],
                hole=.3,
                marker_colors=['#3498db', '#e74c3c']
            )])
            
            fig.update_layout(
                title="2026 Demand Components",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    name='Demand Components',
                    x=['Base Demand', 'Infrastructure', 'Total'],
                    y=[scenario['Base_Demand_Million_Tons'], 
                       scenario['Infrastructure_Million_Tons'],
                       scenario['Total_Demand_Million_Tons']],
                    marker_color=['#3498db', '#e74c3c', '#2ecc71'],
                    text=[f"{scenario['Base_Demand_Million_Tons']:.3f}M",
                          f"{scenario['Infrastructure_Million_Tons']:.3f}M",
                          f"{scenario['Total_Demand_Million_Tons']:.3f}M"],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="2026 Demand Values",
                yaxis_title="Million Tons",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Methodology
        st.markdown("---")
        st.subheader("📋 Methodology")
        
        st.markdown("""
        ### Infrastructure Cable Estimation Formula
        
        ```
        Total Cable Mass ≈ Infrastructure Area (km²) × Cable Density (km/km²) × Cable Weight (t/km)
        ```
        
        **For 2026 Infrastructure Project:**
        - **Area:** 8,779.9 km² (planned grid expansion)
        - **Density:** ~100 km/km² (medium-voltage distribution)
        - **Weight:** ~1 ton/km (standard XLPE cables)
        - **Result:** 878,000 tons
        
        *Source: ngoclancable.com - Industry standard for medium-voltage XLPE cables*
        
        ### Forecast Components
        
        1. **Base Market Demand:** Generated by ensemble model using economic indicators
        2. **Infrastructure Contribution:** Direct calculation from planned deployment
        3. **Total Demand:** Sum of market demand and infrastructure needs
        
        ### Key Assumptions (2026)
        - GDP Growth: 3.5% (Middle East projection)
        - Polyethylene Price: +2% from 2025 baseline
        - Electricity Consumption: +5% annual growth
        - Infrastructure deployment: Full deployment within 2026
        """)
        
        # Business implications
        st.markdown("---")
        st.subheader("💼 Business Implications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Supply Chain Requirements:**
            - Current capacity: ~270K tons/year
            - 2026 requirement: 1,267K tons
            - **Scale-up needed: 4.7× current capacity**
            - Lead time: 6-12 months for capacity expansion
            - Multi-sourcing strategy recommended
            """)
        
        with col2:
            st.markdown(f"""
            **Financial Impact:**
            - Total 2026 demand: {scenario['Total_Demand_Tons']:,.0f} tons
            - Assuming $1,500/ton: **${scenario['Total_Demand_Tons']*1500:,.0f} revenue**
            - Infrastructure alone: **${scenario['Infrastructure_Million_Tons']*1000000*1500:,.0f}**
            - Investment required: $90-170M (capacity + inventory)
            - Break-even: 1-2 years on infrastructure contracts
            """)
        
        # Download report
        st.markdown("---")
        if os.path.exists('outputs/scenario_forecast_2026_report.txt'):
            with open('outputs/scenario_forecast_2026_report.txt', 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            st.download_button(
                label="📥 Download Full Scenario Report",
                data=report_content,
                file_name="2026_Infrastructure_Scenario_Report.txt",
                mime="text/plain"
            )
    else:
        st.warning("No infrastructure scenario forecast available. Please run Models_Advanced.py to generate the forecast.")

# ====================
# PAGE: INVENTORY OPTIMIZATION
# ====================
# In your Streamlit dashboard, REPLACE the Inventory Optimization page with:

elif page == "Inventory Optimization":
    st.header("📦 AI-Driven Inventory Optimization")
    
    st.info("""
    **Industry-Ready Optimization:** Enter actual costs from your operations.
    These should be validated with industry partners (Midal, Ducab, Elsewedy).
    """)
    
    # ===== USER INPUTS FROM DASHBOARD =====
    st.subheader("🏭 Enter Industry Cost Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        material_cost = st.number_input(
            "XLPE Material Cost ($/ton)",
            min_value=1000,
            max_value=5000,
            value=2350,
            help="Current market price of XLPE"
        )
    
    with col2:
        holding_cost = st.number_input(
            "Storage/Holding Cost ($/ton/month)",
            min_value=10,
            max_value=200,
            value=75,
            help="Warehousing, insurance, capital costs per ton per month"
        )
    
    with col3:
        ordering_cost = st.number_input(
            "Order Processing Cost ($/order)",
            min_value=1000,
            max_value=20000,
            value=7500,
            help="Procurement paperwork, quality checks, logistics"
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        stockout_cost = st.number_input(
            "Stockout Penalty ($/ton)",
            min_value=100,
            max_value=5000,
            value=1200,
            help="Cost of production delays, missed contracts, emergency shipping"
        )
    
    with col5:
        service_level = st.slider(
            "Target Service Level (%)",
            min_value=80,
            max_value=99,
            value=95,
            help="Probability of not having a stockout"
        ) / 100.0  # Convert to decimal
    
    with col6:
        lead_time_days = st.number_input(
            "Lead Time (days)",
            min_value=7,
            max_value=90,
            value=30,
            help="Time from order to delivery"
        )
    
    # ===== RUN OPTIMIZATION =====
    if st.button("🚀 Run Inventory Optimization", type="primary"):
        with st.spinner("Calculating optimal inventory policy..."):
            try:
                # ===== 1. CREATE OPTIMIZER =====
                optimizer = InventoryOptimizer(
                    forecast_path='outputs/Ensemble_Risk_Analysis.csv',
                    historical_path='historical_xlpe_demand.xlsx'
                )
                
                # ===== 2. PREPARE USER INPUTS =====
                user_inputs = {
                    'material_cost': material_cost,
                    'holding_cost': holding_cost,
                    'ordering_cost': ordering_cost,
                    'stockout_cost': stockout_cost,
                    'service_level': service_level,  # Already divided by 100
                    'lead_time_days': lead_time_days
                }
                
                # ===== 3. RUN OPTIMIZATION =====
                results = optimizer.optimize(user_inputs)
                
                # ===== 4. DISPLAY RESULTS =====
                st.success("✅ Optimization Complete!")
                
                # Key Metrics
                st.subheader("📊 Optimal Inventory Policy")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    safety_mt = results['inventory_policy']['safety_stock']
                    st.metric(
                        "Safety Stock",
                        f"{safety_mt:.4f} M tons",
                        f"{safety_mt*1000:.0f} tons"
                    )
                
                with col2:
                    rop_mt = results['inventory_policy']['reorder_point']
                    st.metric(
                        "Reorder Point",
                        f"{rop_mt:.4f} M tons",
                        f"{rop_mt*1000:.0f} tons"
                    )
                
                with col3:
                    eoq_mt = results['inventory_policy']['economic_order_quantity']
                    st.metric(
                        "EOQ",
                        f"{eoq_mt:.4f} M tons",
                        f"{eoq_mt*1000:.0f} tons"
                    )
                
                with col4:
                    total_cost = results['cost_analysis']['total_annual_cost']
                    st.metric(
                        "Annual Cost",
                        f"${total_cost:,.0f}",
                        "Total"
                    )
                
                # Cost Breakdown
                st.markdown("---")
                st.subheader("💰 Cost Breakdown")
                
                costs = results['cost_analysis']
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Ordering', 'Holding', 'Material', 'Stockout Risk'],
                    values=[
                        costs['ordering_cost'],
                        costs['holding_cost'], 
                        costs['material_cost'],
                        costs['stockout_cost']
                    ],
                    hole=.3,
                    marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                )])
                
                fig.update_layout(
                    title="Annual Inventory Cost Distribution",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cost Details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Cost Details:**")
                    st.markdown(f"- **Ordering:** ${costs['ordering_cost']:,.0f}")
                    st.markdown(f"- **Holding:** ${costs['holding_cost']:,.0f}")
                    st.markdown(f"- **Material:** ${costs['material_cost']:,.0f}")
                    st.markdown(f"- **Stockout Risk:** ${costs['stockout_cost']:,.0f}")
                    st.markdown(f"---")
                    st.markdown(f"### **Total: ${costs['total_annual_cost']:,.0f}**")
                
                with col2:
                    st.markdown("**Performance Metrics:**")
                    st.markdown(f"- **Service Level:** {results['performance_metrics']['service_level_percent']:.1f}%")
                    st.markdown(f"- **Orders/Year:** {costs['orders_per_year']:.1f}")
                    st.markdown(f"- **Avg Inventory:** {costs['avg_inventory']:.4f} M tons")
                    st.markdown(f"- **Inventory Turnover:** {results['performance_metrics']['inventory_turnover']:.2f}")
                    st.markdown(f"- **Avg Cover Days:** {results['performance_metrics']['avg_inventory_cover_days']:.1f} days")
                
                # Forecast Information
                st.markdown("---")
                st.subheader("🔮 AI Forecast Information")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Monthly Demand Forecast",
                        f"{results['ai_forecast_used']['monthly_demand']:.4f} M tons",
                        "From AI Ensemble"
                    )
                
                with col2:
                    st.metric(
                        "AI Uncertainty (σ)",
                        f"{results['ai_forecast_used']['uncertainty_sigma']:.6f}",
                        "Used for safety stock"
                    )
                
                with col3:
                    if results['ai_forecast_used'].get('hedging_factor', 1.0) != 1.0:
                        st.metric(
                            "Hedging Adjustment",
                            f"{results['ai_forecast_used']['hedging_factor']:.2f}x",
                            "Price-based"
                        )
                    else:
                        st.metric(
                            "Lead Time Demand",
                            f"{results['inventory_policy']['lead_time_demand']:.4f} M tons",
                            f"{results['inventory_policy']['lead_time_days']} days"
                        )
                
                # What-If Scenarios
                st.markdown("---")
                st.subheader("🔍 What-If Scenario Analysis")
                
                scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
                
                with scenario_col1:
                    if st.button("Test +20% Material Price", key="price_up"):
                        with st.spinner("Running scenario..."):
                            scenario_results = optimizer.run_what_if_scenario(
                                user_inputs,
                                {'material_cost': 1.2}
                            )
                            new_cost = scenario_results['results']['cost_analysis']['total_annual_cost']
                            old_cost = results['cost_analysis']['total_annual_cost']
                            change = ((new_cost - old_cost) / old_cost) * 100
                            st.success(f"New cost: ${new_cost:,.0f} ({change:+.1f}%)")
                
                with scenario_col2:
                    if st.button("Test -20% Storage Cost", key="storage_down"):
                        with st.spinner("Running scenario..."):
                            scenario_results = optimizer.run_what_if_scenario(
                                user_inputs,
                                {'holding_cost': 0.8}
                            )
                            new_cost = scenario_results['results']['cost_analysis']['total_annual_cost']
                            old_cost = results['cost_analysis']['total_annual_cost']
                            change = ((new_cost - old_cost) / old_cost) * 100
                            st.success(f"New cost: ${new_cost:,.0f} ({change:+.1f}%)")
                
                with scenario_col3:
                    if st.button("Test +50% Order Cost", key="order_up"):
                        with st.spinner("Running scenario..."):
                            scenario_results = optimizer.run_what_if_scenario(
                                user_inputs,
                                {'ordering_cost': 1.5}
                            )
                            new_eoq = scenario_results['results']['inventory_policy']['economic_order_quantity']
                            old_eoq = results['inventory_policy']['economic_order_quantity']
                            change = ((new_eoq - old_eoq) / old_eoq) * 100
                            st.success(f"New EOQ: {new_eoq:.4f}M ({change:+.1f}%)")
                
                # Download Results
                st.markdown("---")
                st.subheader("📥 Download Results")
                
                # Convert results to JSON
                results_json = json.dumps(results, indent=4)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📄 Download Full Results (JSON)",
                        data=results_json,
                        file_name="inventory_optimization_results.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Save via optimizer
                    saved_files = optimizer.save_results(results)
                    st.info(f"Results saved to:\n{saved_files['json']}\n{saved_files['csv']}")
                
                # Technical Details (Collapsible)
                with st.expander("🔧 Technical Details"):
                    st.json(results['calculation_notes'])
                    st.markdown(f"**Uncertainty Source:** {results['calculation_notes']['safety_stock_source']}")
                    st.markdown(f"**Demand Source:** {results['calculation_notes']['demand_source']}")
                    
            except FileNotFoundError as e:
                st.error(f"❌ Required file not found: {e}")
                st.info("""
                **Please run these steps first:**
                1. Run `Models_Advanced.py` to generate AI forecasts
                2. Ensure `outputs/Ensemble_Risk_Analysis.csv` exists
                3. Ensure `historical_xlpe_demand.xlsx` is in the same folder
                """)
                
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                st.info("Check that all required files are generated and paths are correct.")

    # ===== DEFAULT VIEW (BEFORE OPTIMIZATION) =====
    else:
        st.markdown("---")
        st.subheader("ℹ️ How It Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Inventory Optimization Process:**
            1. **Loads your AI forecasts** from Ensemble model
            2. **Uses AI uncertainty** for safety stock calculations
            3. **Applies your cost inputs** to EOQ formula
            4. **Generates actionable recommendations**
            
            **No new forecasting** - uses your existing 98.6% accurate AI predictions.
            """)
        
        with col2:
            st.markdown("""
            **What You'll Get:**
            - ✅ **Safety Stock:** Buffer for demand uncertainty
            - ✅ **Reorder Point:** When to place new orders
            - ✅ **EOQ:** Optimal order quantity
            - ✅ **Cost Analysis:** Breakdown of all inventory costs
            - ✅ **What-If Scenarios:** Test price changes
            - ✅ **Downloadable Reports:** For stakeholders
            """)
        
        st.warning("""
        **Prerequisites:**
        - Run `Models_Advanced.py` first to generate AI forecasts
        - The optimizer uses `Ensemble_Risk_Analysis.csv` for predictions
        - All calculations use YOUR actual data (no assumptions)
        """)

    # Optionally: plot cost curves, let user simulate more scenarios, etc.
# ====================
# PAGE: FORECASTING TOOL
# ====================
elif page == "Forecasting Tool":
    st.header("🔮 Interactive Forecasting Tool")
    
    st.info("Enter economic indicators to predict XLPE demand for the next month.")
    
    # Feature inputs - only use features that exist in historical data
    all_features = data['metadata']['features']
    historical_clean = data['historical_data'].drop(columns=['Date', 'Year', 'Month']).dropna()
    
    # Filter to only features that exist in the data
    features = [f for f in all_features if f in historical_clean.columns]
    
    st.subheader("📝 Input Economic Indicators")
    
    col1, col2 = st.columns(2)
    
    input_values = {}
    
    # Get latest values as defaults
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
    
    # Infrastructure scenario addition
    st.markdown("---")
    st.subheader("🏗️ Optional: Add Infrastructure Contribution")
    
    enable_infrastructure = st.checkbox("Include planned infrastructure deployment")
    infrastructure_tons = 0
    
    if enable_infrastructure:
        st.info("""
        **Infrastructure Cable Estimation:**
        - Enter planned deployment in tons, or
        - Calculate from area: `Area (km²) × 100 km/km² × 1 t/km`
        """)
        
        calc_method = st.radio("Input Method:", ["Direct (tons)", "Calculate from Area (km²)"])
        
        if calc_method == "Direct (tons)":
            infrastructure_tons = st.number_input(
                "Planned Infrastructure (tons)",
                min_value=0,
                max_value=10000000,
                value=878000,
                step=1000,
                help="2026 default: 878,000 tons from 8,779.9 km² grid expansion"
            )
        else:
            area_km2 = st.number_input("Infrastructure Area (km²)", min_value=0.0, value=8779.9, step=100.0)
            density = st.slider("Cable Density (km/km²)", min_value=50, max_value=200, value=100, step=10)
            weight = st.slider("Cable Weight (tons/km)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            infrastructure_tons = int(area_km2 * density * weight)
            st.success(f"Calculated: **{infrastructure_tons:,} tons**")
    
    if st.button("🚀 Generate Forecast", type="primary"):
        # Prepare input
        input_df = pd.DataFrame([input_values])
        
        # Add infrastructure proxy features (same as Models_Advanced.py)
        # These are synthetic features used for scenario forecasting
        if 'gdp_growth_rate' in input_df.columns:
            input_df['construction_output_index'] = 100 + (input_df['gdp_growth_rate'] * 2.5)
        else:
            input_df['construction_output_index'] = 100.0
        
        if 'Total electricity consumption, Middle East' in input_df.columns:
            # Use a reference max from historical data
            elec_max = historical_clean['Total electricity consumption, Middle East'].max() if 'Total electricity consumption, Middle East' in historical_clean.columns else 1.0
            input_df['urbanization_rate'] = (input_df['Total electricity consumption, Middle East'] / elec_max) * 100
        else:
            input_df['urbanization_rate'] = 50.0
        
        # For infrastructure investment, use the current demand estimate (can't use rolling on single point)
        # Use average of historical data as proxy
        if 'infrastructure_investment' in data['metadata']['features']:
            hist_avg = historical_clean['xlpe_demand_Million_tons'].mean() if 'xlpe_demand_Million_tons' in historical_clean.columns else 0.27
            input_df['infrastructure_investment'] = hist_avg
        
        # Add lag features from historical data (required by the model)
        if 'xlpe_demand_Million_tons' in historical_clean.columns:
            demand_series = historical_clean['xlpe_demand_Million_tons']
            input_df['lag_1'] = demand_series.iloc[-1] if len(demand_series) >= 1 else 0.27
            input_df['lag_3'] = demand_series.iloc[-3] if len(demand_series) >= 3 else 0.27
            input_df['lag_12'] = demand_series.iloc[-12] if len(demand_series) >= 12 else 0.27
            input_df['rolling_mean_3'] = demand_series.iloc[-3:].mean() if len(demand_series) >= 3 else 0.27
        
        # Make prediction (base market demand)
        base_prediction = data['best_model'].predict(input_df)[0]
        
        # Add infrastructure if enabled
        infrastructure_mt = infrastructure_tons / 1_000_000  # Convert to million tons
        total_prediction = base_prediction + infrastructure_mt
        
        # Display result
        st.markdown("---")
        st.success("✅ Forecast Generated Successfully!")
        
        if enable_infrastructure:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Base Market Demand",
                    f"{base_prediction:.6f} M tons",
                    delta=f"{base_prediction*1000000:,.0f} tons"
                )
            
            with col2:
                st.metric(
                    "Infrastructure",
                    f"{infrastructure_mt:.6f} M tons",
                    delta=f"{infrastructure_tons:,} tons"
                )
            
            with col3:
                infra_pct = (infrastructure_mt / total_prediction) * 100 if total_prediction > 0 else 0
                st.metric(
                    "TOTAL Demand",
                    f"{total_prediction:.6f} M tons",
                    delta=f"{infra_pct:.1f}% infrastructure"
                )
            
            with col4:
                avg_demand = historical_clean['xlpe_demand_Million_tons'].mean()
                growth_pct = ((total_prediction - avg_demand) / avg_demand) * 100
                st.metric(
                    "Growth vs Avg",
                    f"+{growth_pct:.1f}%",
                    delta="Total increase"
                )
            
            # Visualization
            st.markdown("---")
            fig = go.Figure(data=[
                go.Bar(
                    name='Demand Components',
                    x=['Base Market', 'Infrastructure', 'Total'],
                    y=[base_prediction, infrastructure_mt, total_prediction],
                    marker_color=['#3498db', '#e74c3c', '#2ecc71'],
                    text=[f"{base_prediction:.4f}M", f"{infrastructure_mt:.4f}M", f"{total_prediction:.4f}M"],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Demand Breakdown",
                yaxis_title="Million Tons",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted XLPE Demand",
                    f"{base_prediction:.6f} M tons",
                    delta=f"{base_prediction*1000:.2f} tons"
                )
            
            with col2:
                avg_demand = historical_clean['xlpe_demand_Million_tons'].mean()
                diff_pct = ((base_prediction - avg_demand) / avg_demand) * 100
                st.metric(
                    "vs Historical Average",
                    f"{diff_pct:+.2f}%",
                    delta="Comparison"
                )
            
            with col3:
                # Check inventory recommendation
                if base_prediction < data['inventory_results']['safety_stock_million_tons']:
                    recommendation = "✅ Current stock sufficient"
                    st.metric("Inventory Status", "SAFE", delta=recommendation)
                elif base_prediction < data['inventory_results']['reorder_point_million_tons']:
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
    - **Infrastructure Planning:** Large-scale deployment forecasting (878K tons for 2026)
    
    ---
    
    ### 🤖 Machine Learning Innovation
    
    **Ensemble Forecasting Architecture:**
    - **Random Forest Regressor (60.2%):** Risk-aware predictions with tree-level uncertainty
    - **Ridge Regression (39.8%):** Stabilized linear trends with L2 regularization
    - **Weighted Voting:** Cubed accuracy weighting (98.6³) with 1.5× leader boost
    - **Result:** 98.60% test accuracy, 70% variance reduction vs single models
    
    **Infrastructure Proxy Features (NEW):**
    1. **Construction Output Index:** GDP-based indicator of building activity
    2. **Urbanization Rate:** Electricity-consumption-based growth proxy
    3. **Infrastructure Investment:** Rolling average of deployment patterns
    
    **Risk-Aware Forecasting:**
    - **Linear Regression:** Weighted lags (0.6/0.3/0.1), asymmetric loss (1.5× for increases), ±5% price hedging
    - **Random Forest:** Tree-level σ (0.007), 95% CI safety stock (1.96σ), price-based hedging
    - **Gradient Boosting:** Staged predictions (100 stages), uncertainty from variance
    - **Ensemble:** Combined uncertainty (σ=0.002255), 70% lower than Random Forest alone
    
    **Scenario-Based Forecasting:**
    - Predict base market demand from economic indicators
    - Add planned infrastructure deployment (tons or calculated from area)
    - Formula: `Area (km²) × Cable Density (km/km²) × Weight (t/km)`
    - Example: 8,779.9 km² × 100 km/km² × 1 t/km = 878,000 tons
    
    ---
    
    ### 📊 Key Features
    
    **1. Advanced Forecasting**
    - 98.60% ensemble accuracy (Random Forest + Ridge)
    - Risk-aware predictions with uncertainty quantification
    - Infrastructure scenario forecasting for large-scale projects
    - 12-month forward-looking predictions
    
    **2. Inventory Optimization**
    - Economic Order Quantity (EOQ) calculation
    - Safety Stock with 95% service level
    - Reorder Point optimization
    - **Annual Savings:** $18-73M from optimized inventory
    
    **3. Real-Time Analysis**
    - Interactive forecasting tool with infrastructure scenarios
    - Model performance comparison (7 algorithms tested)
    - Risk analysis with uncertainty bands
    - What-if scenario simulations
    
    **4. Business Intelligence**
    - Visual dashboards for decision-making
    - Feature importance analysis (GDP, price, electricity consumption)
    - Cost-benefit analysis and ROI projections
    - 2026 infrastructure forecast: $1.9B revenue opportunity
    
    ---
    
    ### 📈 Performance Metrics
    
    **Best Model: Ensemble (Random Forest + Ridge)**
    - Test Accuracy: 98.60%
    - R² Score: 0.9974
    - MAE: 0.0031 M tons (3,100 tons)
    - RMSE: 0.0047 M tons (4,700 tons)
    - Overfitting: 0.0048 (minimal, excellent generalization)
    - Risk-Adjusted Score: 95.58 points (vs 94.78 for RF alone)
    
    **Inventory Optimization Results:**
    - Optimal Order Quantity: 0.0621 M tons (62,100 tons)
    - Safety Stock: 0.0166 M tons (16,600 tons)
    - Reorder Point: 0.0216 M tons (21,600 tons)
    - Service Level: 95%
    - Annual Savings: $18-73M (inventory reduction + stockout prevention)
    
    **2026 Infrastructure Scenario:**
    - Base Market Demand: 0.389 M tons
    - Infrastructure Contribution: 0.878 M tons (69.3%)
    - Total 2026 Demand: 1.267 M tons
    - Revenue Opportunity: $1.9B @ $1,500/ton
    - Required Scale-up: 4.7× current capacity (270K tons/year)
    
    ---
    
    ### 🛠️ Technologies Used
    
    **Core ML Stack:**
    - Python 3.13
    - scikit-learn (ensemble methods, regression)
    - pandas & NumPy (data processing)
    - plotly & Streamlit (visualization)
    
    **Models Evaluated:**
    - ✅ Random Forest (98.57% - selected)
    - ✅ Ridge Regression (98.48% - selected)
    - Linear Regression (98.29%)
    - Gradient Boosting (98.35%)
    - Support Vector Regression (97.76%)
    - Lasso Regression (98.06%)
    - KNN Regressor (96.58%)
    
    **Optimization Algorithms:**
    - Economic Order Quantity (EOQ)
    - Safety Stock Calculation (z-score method)
    - Reorder Point Optimization
    - Risk-Adjusted Forecasting (asymmetric loss, hedging)
    
    ---
    
    ### 📦 Deliverables
    
    1. **Forecasting Models**
       - 7 trained models with performance comparison
       - Ensemble model with 98.60% accuracy
       - Risk-aware predictions with uncertainty bands
       - Infrastructure scenario forecasting capability
    
    2. **Optimization Results**
       - EOQ analysis with cost minimization
       - Safety stock recommendations (95% service level)
       - Reorder point calculations
       - 12-month inventory forecast
    
    3. **Business Reports**
       - Model performance report (accuracy, overfitting, ensemble composition)
       - Inventory optimization report ($18-73M savings)
       - Infrastructure scenario report (2026 forecast: $1.9B opportunity)
       - Risk analysis report (uncertainty quantification)
       - Feature importance report (GDP, price, electricity top 3)
    
    4. **Interactive Dashboard**
       - Real-time forecasting tool
       - Infrastructure scenario simulator
       - Risk analysis with uncertainty bands
       - Model comparison visualizations
       - Inventory optimization insights
    
    ---
    
    ### 💡 Innovation Highlights
    
    **1. Risk-Aware Forecasting**
    - First cable demand model with built-in uncertainty quantification
    - Provides 95% confidence intervals for every prediction
    - Adaptive hedging based on price signals (±5%)
    - 70% variance reduction through ensemble approach
    
    **2. Infrastructure Scenario Forecasting**
    - Novel approach combining market demand + infrastructure deployment
    - Validated methodology: Area × Density × Weight
    - Critical for large-scale projects (878K tons validated for 2026)
    - Enables capacity planning 12-18 months ahead
    
    **3. Asymmetric Loss Function**
    - 1.5× penalty for underestimating increasing demand
    - Prevents stockouts during growth periods
    - Aligns ML optimization with business objectives
    - Reduces opportunity cost of missed sales
    
    **4. Ensemble with Cubed Weighting**
    - Traditional ensembles use linear weights (98.6% → 98.6%)
    - Our approach: Cubed accuracy (98.6³ = 958,266) amplifies differences
    - Strongly favors best performers (60/40 RF/Ridge split)
    - 1.5× leader boost prevents excessive fragmentation
    - Result: Higher accuracy, lower variance, better risk profile
    
    ---
    
    ### 📚 Documentation
    
    - **METHODOLOGY_REPORT.md:** Complete technical methodology
    - **ENHANCED_SOLUTION_GUIDE.md:** Implementation guide with code examples
    - **FORECASTING_IMPROVEMENTS.md:** Detailed explanation of all enhancements
    - **INFRASTRUCTURE_SCENARIO_FORECASTING_GUIDE.md:** Infrastructure forecasting methodology
    - **SCENARIO_FORECASTING_REPORT.md:** Business-focused 2026 forecast report
    - **SIMPLE_EXPLANATION.md:** Non-technical project overview
    
    ---
    
    ### 🎓 Academic Rigor
    
    - All data sources properly cited
    - Statistical validation (train/test split, cross-validation)
    - Overfitting analysis (train vs test accuracy)
    - Ensemble justified through risk-adjusted scoring
    - Infrastructure methodology validated against industry standards
    
    ---
    
    ### 👥 Target Audience
    
    - Cable manufacturing companies (ARABCAB, etc.)
    - Supply chain managers
    - Procurement teams
    - Strategic planners
    - Financial analysts
    - Data scientists in manufacturing
    
    ---
    
    ### 🏆 Competitive Advantages
    
    1. **98.60% Accuracy:** Among the highest in cable demand forecasting
    2. **Risk Quantification:** Uncertainty bands for every prediction
    3. **Infrastructure Scenarios:** Unique capability for large-scale planning
    4. **Inventory Savings:** $18-73M annual savings potential
    5. **Revenue Opportunity:** $1.9B identified for 2026 infrastructure
    6. **Scalable Solution:** Handles both day-to-day and mega-projects
    7. **Interactive Dashboard:** Non-technical stakeholders can use directly
    
    ---
    
    ### 📧 Contact & Support
    
    **Competition:** ARABCAB Scientific Competition  
    **Industry:** Cable & Metals Manufacturing  
    **Geography:** Egypt • Bahrain • UAE  
    **Year:** 2024-2025  
    
    *This dashboard was built with ❤️ using Python, scikit-learn, and Streamlit.*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>ARABCAB AI Competition 2024-2025</strong></p>
    <p>AI-Based Demand Forecasting & Inventory Optimization</p>
    <p>Egypt • Bahrain • UAE</p>
</div>
""", unsafe_allow_html=True)
