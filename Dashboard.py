import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import hashlib

from inventory_optimization import InventoryOptimizer

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CableFlow-AI · XLPE Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
/* ── Remove Streamlit default top padding ── */
div[data-testid="block-container"] { padding-top: 1rem !important; }
div[data-testid="stAppViewContainer"] > section > div:first-child { padding-top: 0 !important; }

/* ── Auth screen ── */
.auth-wrapper {
    display: flex; align-items: flex-start; justify-content: center;
    padding: 0.5rem 2rem 2rem 2rem;
}
.auth-box {
    background: white; border-radius: 14px; padding: 2.5rem;
    max-width: 420px; width: 100%;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    border: 1px solid #e8f5f0;
}
.auth-logo { display: flex; align-items: center; gap: 12px; margin-bottom: 1.8rem; }
.auth-logo-icon {
    width: 46px; height: 46px; background: linear-gradient(135deg, #1D9E75, #0F6E56);
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-size: 22px; color: white;
}
.auth-logo-name { font-size: 22px; font-weight: 700; color: #0F6E56; }
.auth-logo-sub  { font-size: 12px; color: #888; margin-top: 1px; }
.role-grid { display: flex; gap: 8px; margin-bottom: 1.2rem; }
.role-chip {
    flex: 1; padding: 7px 0; text-align: center; border-radius: 8px;
    font-size: 12px; font-weight: 600; cursor: pointer;
    border: 1.5px solid #d0e8e0; background: #f4faf7; color: #1D9E75;
    transition: all .15s;
}
.role-chip.selected { background: #1D9E75; color: white; border-color: #1D9E75; }
.demo-hint {
    background: #f4faf7; border-radius: 8px; padding: 10px 14px;
    font-size: 12px; color: #555; margin-top: 1.2rem; line-height: 1.7;
    border-left: 3px solid #1D9E75;
}

/* ── Top bar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    background: white; padding: 10px 24px; border-radius: 10px;
    margin-bottom: 1.5rem; border: 1px solid #e8f5f0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.topbar-left { display: flex; align-items: center; gap: 10px; }
.topbar-logo {
    width: 32px; height: 32px; background: linear-gradient(135deg, #1D9E75, #0F6E56);
    border-radius: 8px; display: flex; align-items: center; justify-content: center;
    font-size: 16px; color: white;
}
.topbar-title { font-size: 18px; font-weight: 700; color: #0F6E56; }
.topbar-right { display: flex; align-items: center; gap: 12px; }
.user-avatar {
    width: 34px; height: 34px; border-radius: 50%;
    background: linear-gradient(135deg, #1D9E75, #0F6E56);
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; color: white;
}
.user-name  { font-size: 14px; font-weight: 600; color: #333; }
.user-email { font-size: 11px; color: #888; }
.role-badge {
    padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700;
}
.role-badge-admin   { background: #fde8e8; color: #c0392b; }
.role-badge-analyst { background: #e8f4fd; color: #185FA5; }
.role-badge-viewer  { background: #e8f5f0; color: #0F6E56; }

/* ── KPI cards ── */
.kpi-card {
    background: white; border-radius: 12px; padding: 1.2rem 1.4rem;
    border: 1px solid #e8f5f0; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.kpi-label { font-size: 12px; color: #888; margin-bottom: 4px; }
.kpi-value { font-size: 26px; font-weight: 700; color: #0F6E56; }
.kpi-delta { font-size: 11px; color: #1D9E75; margin-top: 2px; }

/* ── Access denied ── */
.access-denied {
    text-align: center; padding: 4rem 2rem; color: #aaa;
}
.access-denied .icon { font-size: 52px; margin-bottom: 1rem; }
.access-denied h3 { color: #555; font-size: 20px; margin-bottom: 6px; }

/* ── Misc ── */
.section-title {
    font-size: 15px; font-weight: 700; color: #0F6E56;
    margin-bottom: .8rem; padding-bottom: .4rem;
    border-bottom: 2px solid #e8f5f0;
}
.info-box {
    background: #f4faf7; border-left: 4px solid #1D9E75;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-size: 13px; color: #444; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# AUTH CONFIG
# ============================================================
USERS = {
    "admin@arabcab.com": {
        "name": "Admin User",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "Admin",
        "initials": "AU",
        "permissions": ["Overview", "Model Performance", "Risk Analysis",
                        "Infrastructure Scenario", "Inventory Optimization",
                        "Forecasting Tool", "About"],
    },
    "analyst@arabcab.com": {
        "name": "Data Analyst",
        "password_hash": hashlib.sha256("analyst123".encode()).hexdigest(),
        "role": "Analyst",
        "initials": "DA",
        "permissions": ["Overview", "Model Performance", "Risk Analysis",
                        "Infrastructure Scenario", "Inventory Optimization",
                        "Forecasting Tool", "About"],
    },
    "viewer@arabcab.com": {
        "name": "Procurement Viewer",
        "password_hash": hashlib.sha256("viewer123".encode()).hexdigest(),
        "role": "Viewer",
        "initials": "PV",
        "permissions": ["Overview", "Inventory Optimization",
                        "Forecasting Tool", "About"],
    },
}

DEMO_CREDENTIALS = {
    "Admin":   ("admin@arabcab.com",   "admin123"),
    "Analyst": ("analyst@arabcab.com", "analyst123"),
    "Viewer":  ("viewer@arabcab.com",  "viewer123"),
}

ROLE_BADGE_CLASS = {
    "Admin":   "role-badge-admin",
    "Analyst": "role-badge-analyst",
    "Viewer":  "role-badge-viewer",
}


# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
for key, val in [
    ("authenticated", False),
    ("user_email", None),
    ("user_info", None),
    ("login_error", ""),
    ("selected_demo_role", "Admin"),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ============================================================
# LOGIN PAGE
# ============================================================
def show_login():
    st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class="auth-box">
          <div class="auth-logo">
            <div class="auth-logo-icon">📊</div>
            <div>
              <div class="auth-logo-name">CableFlow-AI</div>
              <div class="auth-logo-sub">XLPE Demand Intelligence Platform</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Quick demo — pick a role:**")
        cols = st.columns(3)
        for i, role in enumerate(["Admin", "Analyst", "Viewer"]):
            with cols[i]:
                if st.button(role, key=f"demo_{role}", use_container_width=True):
                    st.session_state.selected_demo_role = role

        demo_email, demo_pw = DEMO_CREDENTIALS[st.session_state.selected_demo_role]
        role = st.session_state.selected_demo_role
        desc = {
            "Admin":   "Full access to all pages",
            "Analyst": "Full access to all pages",
            "Viewer":  "Overview, Inventory & Forecast only",
        }[role]
        st.caption(f"🔑 **{role}** — {desc}")

        st.markdown("---")
        with st.form("login_form"):
            email = st.text_input("Email", value=demo_email, placeholder="you@arabcab.com")
            password = st.text_input("Password", value=demo_pw, type="password")

            if st.form_submit_button("Sign in →", use_container_width=True, type="primary"):
                _do_login(email.strip(), password)

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

        st.markdown("""
        <div class="demo-hint">
          <b>Demo accounts</b><br>
          admin@arabcab.com / admin123 (Admin)<br>
          analyst@arabcab.com / analyst123 (Analyst)<br>
          viewer@arabcab.com / viewer123 (Viewer — restricted)
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def _do_login(email, password):
    user = USERS.get(email)
    if user and user["password_hash"] == hashlib.sha256(password.encode()).hexdigest():
        st.session_state.authenticated = True
        st.session_state.user_email = email
        st.session_state.user_info = user
        st.session_state.login_error = ""
        st.rerun()
    else:
        st.session_state.login_error = "❌ Invalid email or password."


def logout():
    for k in ["authenticated", "user_email", "user_info"]:
        st.session_state[k] = False if k == "authenticated" else None
    st.rerun()


# ============================================================
# TOP BAR (shown when logged in)
# ============================================================
def show_topbar():
    u = st.session_state.user_info
    badge_cls = ROLE_BADGE_CLASS.get(u["role"], "role-badge-viewer")
    st.markdown(f"""
    <div class="topbar">
      <div class="topbar-left">
        <div class="topbar-logo">📊</div>
        <span class="topbar-title">CableFlow-AI</span>
        <span style="font-size:13px;color:#aaa;margin-left:4px">| XLPE Demand Intelligence</span>
      </div>
      <div class="topbar-right">
        <div style="display:flex;align-items:center;gap:8px">
          <div class="user-avatar">{u['initials']}</div>
          <div>
            <div class="user-name">{u['name']}</div>
            <div class="user-email">{st.session_state.user_email}</div>
          </div>
        </div>
        <span class="role-badge {badge_cls}">{u['role']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ACCESS DENIED
# ============================================================
def show_access_denied(page_name):
    st.markdown(f"""
    <div class="access-denied">
      <div class="icon">🔒</div>
      <h3>Access Restricted</h3>
      <p>Your role (<b>{st.session_state.user_info['role']}</b>) does not have permission
         to view <b>{page_name}</b>.</p>
      <p style="font-size:13px;margin-top:.5rem">
        Contact your administrator or sign in with a higher-privilege account.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DATA LOADING (cached)
# ============================================================
@st.cache_data
def load_data():
    try:
        model_results     = pd.read_csv('outputs/model_comparison_results.csv')
        with open('outputs/inventory_optimization_results.json') as f:
            inventory_results = json.load(f)
        inventory_forecast = pd.read_csv('outputs/inventory_forecast_12months.csv')
        with open('outputs/model_metadata.json') as f:
            metadata = json.load(f)
        with open('outputs/best_pipeline.pkl', 'rb') as f:
            best_model = pickle.load(f)
        historical_data   = pd.read_excel('historical_xlpe_demand.xlsx')

        scenario_forecast = None
        if os.path.exists('outputs/scenario_forecast_2026.csv'):
            scenario_forecast = pd.read_csv('outputs/scenario_forecast_2026.csv')

        risk_files = {}
        for mn in ['Random_Forest_(Risk-Aware)', 'Gradient_Boosting_(Risk-Aware)',
                   'Linear_Regression_(Enhanced)', 'Ensemble']:
            fp = (f'outputs/{mn}_Risk_Analysis.csv' if mn != 'Linear_Regression_(Enhanced)'
                  else 'outputs/Linear_Regression_(Enhanced)_Risk_Adjusted.csv')
            if os.path.exists(fp):
                risk_files[mn] = pd.read_csv(fp)

        return dict(model_results=model_results, inventory_results=inventory_results,
                    inventory_forecast=inventory_forecast, metadata=metadata,
                    best_model=best_model, historical_data=historical_data,
                    scenario_forecast=scenario_forecast, risk_files=risk_files)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Run Models_Advanced.py and inventory_optimization.py first.")
        return None


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
ALL_PAGES = [
    ("📊 Overview",               "Overview"),
    ("🤖 Model Performance",      "Model Performance"),
    ("⚠️ Risk Analysis",          "Risk Analysis"),
    ("🏗️ Infrastructure Scenario","Infrastructure Scenario"),
    ("📦 Inventory Optimization", "Inventory Optimization"),
    ("🔮 Forecasting Tool",       "Forecasting Tool"),
    ("ℹ️ About",                  "About"),
]

def show_sidebar(user_permissions):
    with st.sidebar:
        st.markdown("## 📋 Navigation")
        page = None
        for label, key in ALL_PAGES:
            locked = key not in user_permissions
            icon   = " 🔒" if locked else ""
            btn    = st.button(f"{label}{icon}", key=f"nav_{key}",
                               use_container_width=True,
                               disabled=locked)
            if btn:
                st.session_state["current_page"] = key

        st.markdown("---")
        st.info("**ARABCAB Competition**\nAI Demand Forecasting\nEgypt · UAE · Bahrain")
        st.markdown("---")
        if st.button("🚪 Sign out", use_container_width=True):
            logout()

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Overview"
    return st.session_state["current_page"]


# ============================================================
# PAGE: OVERVIEW
# ============================================================
def page_overview(data):
    st.markdown('<div class="section-title">📊 Executive Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Best Model", data['metadata']['best_model'], "Selected")
    with c2:
        st.metric("Forecast Accuracy",
                  f"{data['metadata']['best_forecast_accuracy']:.2f}%",
                  f"R² = {data['metadata']['best_r2']:.4f}")
    with c3:
        if data['scenario_forecast'] is not None:
            td = data['scenario_forecast']['Total_Demand_Million_Tons'].values[0]
            st.metric("2026 Total Demand", f"{td:.3f} M tons",
                      f"{td*1e6:,.0f} tons")
        else:
            st.metric("Service Level",
                      f"{data['inventory_results']['service_level_percent']:.1f}%",
                      "Target met")
    with c4:
        ac = data['inventory_results']['annual_costs']['total_cost']
        st.metric("Annual Inventory Cost", f"${ac/1e6:.2f}M", "Optimized")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎯 Key Findings")
        members = data['metadata'].get('ensemble_members', [])
        ens_txt = f"**Ensemble:** {', '.join(m.split('(')[0].strip() for m in members)}" if members else ""
        st.markdown(f"""
        - **{data['metadata']['train_size']}** training samples
        - **{data['metadata']['test_size']}** test samples
        - **{len(data['metadata']['features'])}** predictive features (incl. infra proxies)
        - **12-month** forward forecast generated
        - Risk-Aware forecasting with uncertainty quantification
        - {ens_txt}
        """)
    with c2:
        st.subheader("💡 Business Impact")
        if data['scenario_forecast'] is not None:
            ip = data['scenario_forecast']['Infrastructure_Percentage'].values[0]
            td = data['scenario_forecast']['Total_Demand_Million_Tons'].values[0]
            st.markdown(f"""
            - ✅ Infrastructure scenario: **878K tons (2026)**
            - ✅ Total demand projection: **{td:.3f}M tons**
            - ✅ Infrastructure share: **{ip:.1f}%** of total
            - ✅ **70% lower variance** vs single models
            - ✅ Automated safety-stock optimisation
            - ✅ Price-based hedging strategy
            """)
        else:
            st.markdown("""
            - ✅ Stockout risk reduced to < 5%
            - ✅ Holding costs optimised
            - ✅ 98.60% forecast accuracy
            - ✅ Data-driven procurement decisions
            """)

    st.markdown("---")
    st.subheader("📈 Historical XLPE Demand Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['historical_data']['Date'],
        y=data['historical_data']['xlpe_demand_Million_tons'],
        mode='lines+markers', name='XLPE Demand',
        line=dict(color='#1D9E75', width=2), marker=dict(size=4)
    ))
    fig.update_layout(title="XLPE Demand Over Time",
                      xaxis_title="Date", yaxis_title="Demand (Million Tons)",
                      hovermode='x unified', height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
def page_model_performance(data):
    st.header("🤖 Model Performance Analysis")

    st.subheader("📊 Forecast Accuracy Comparison")
    fig = px.bar(data['model_results'], x='Model', y='Forecast Accuracy (%)',
                 color='Model', text='Forecast Accuracy (%)',
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(showlegend=False, height=400, yaxis_range=[95, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Comprehensive Model Metrics")
    styled = (data['model_results'].style
              .highlight_max(axis=0, subset=['Forecast Accuracy (%)', 'R² Score'], color='lightgreen')
              .highlight_min(axis=0, subset=['MAE (million tons)', 'MAPE (%)', 'Overfitting'], color='lightgreen')
              .format({'Forecast Accuracy (%)': '{:.2f}', 'MAE (million tons)': '{:.6f}',
                       'RMSE (million tons)': '{:.6f}', 'R² Score': '{:.4f}',
                       'MAPE (%)': '{:.2f}', 'Train R²': '{:.4f}', 'Overfitting': '{:.4f}'}))
    st.dataframe(styled, use_container_width=True)
    st.info("**Overfitting Score:** Lower = better generalisation. <0.1 excellent, 0.1–0.2 good, >0.2 moderate.")

    st.markdown("---")
    st.subheader(f"🏆 Best Model: {data['metadata']['best_model']}")
    bm = data['model_results'][data['model_results']['Model'] == data['metadata']['best_model']].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Accuracy", f"{bm['Forecast Accuracy (%)']:.2f}%")
    with c2: st.metric("MAE", f"{bm['MAE (million tons)']:.6f} M tons")
    with c3: st.metric("R²", f"{bm['R² Score']:.4f}")
    with c4:
        ov_lbl = "Low" if bm['Overfitting'] < 0.1 else ("Moderate" if bm['Overfitting'] < 0.2 else "High")
        st.metric("Overfitting", f"{bm['Overfitting']:.4f}", delta=ov_lbl)

    if 'Ensemble' in data['metadata']['best_model']:
        st.markdown("---")
        st.subheader("🔗 Ensemble Composition")
        members = data['metadata'].get('ensemble_members', [])
        weights = data['metadata'].get('ensemble_weights', {})
        if members and weights:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Member models:**")
                for m in members:
                    st.markdown(f"- {m}: **{weights.get(m, 0)*100:.1f}%** weight")
            with c2:
                fig = px.pie(values=list(weights.values()), names=list(weights.keys()),
                             title="Ensemble Weight Distribution")
                st.plotly_chart(fig, use_container_width=True)

    if os.path.exists('outputs/feature_importance.csv'):
        st.markdown("---")
        st.subheader("🎯 Feature Importance")
        fi = pd.read_csv('outputs/feature_importance.csv')
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Greens')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: RISK ANALYSIS
# ============================================================
def page_risk_analysis(data):
    st.header("⚠️ Risk-Aware Forecasting Analysis")
    st.markdown("""
    <div class="info-box">
      <b>Risk-Aware Forecasting</b> quantifies prediction uncertainty and provides safety margins.
      Critical for high-stakes decisions like the 878K-ton infrastructure project.
    </div>
    """, unsafe_allow_html=True)

    available = list(data['risk_files'].keys())
    if not available:
        st.warning("No risk analysis files found. Run Models_Advanced.py first.")
        return

    sel = st.selectbox("Select model",  available,
                       index=available.index('Ensemble') if 'Ensemble' in available else 0)
    rd  = data['risk_files'][sel]

    c1, c2, c3, c4 = st.columns(4)
    fc_col = ('Forecast' if 'Forecast' in rd.columns
              else ('Risk_Adjusted_Prediction' if 'Risk_Adjusted_Prediction' in rd.columns
                    else 'Base_Prediction'))
    with c1: st.metric("Avg Forecast", f"{rd[fc_col].mean():.6f} M tons")
    with c2:
        if 'Uncertainty_Sigma' in rd.columns:
            st.metric("Avg σ (uncertainty)", f"{rd['Uncertainty_Sigma'].mean():.6f} M tons")
        else:
            st.metric("Model type", "Risk-Adjusted")
    with c3:
        if 'Safety_Stock_95CI' in rd.columns:
            st.metric("Avg Safety Stock", f"{rd['Safety_Stock_95CI'].mean():.6f} M tons", "95% CI")
        elif 'Risk_Factor' in rd.columns:
            st.metric("Risk Factor Range", f"{rd['Risk_Factor'].min():.2f}–{rd['Risk_Factor'].max():.2f}×")
    with c4:
        if 'Hedging_Factor' in rd.columns:
            hc = (rd['Hedging_Factor'] != 1.0).sum()
            st.metric("Price Adjustments", f"{hc}/{len(rd)}", f"{hc/len(rd)*100:.0f}% of periods")

    st.markdown("---")
    st.subheader("📈 Forecast with Uncertainty Bands")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(rd))), y=rd['Actual'],
                             mode='lines+markers', name='Actual',
                             line=dict(color='black', width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=list(range(len(rd))), y=rd[fc_col],
                             mode='lines+markers', name='Predicted',
                             line=dict(color='#1D9E75', width=2), marker=dict(size=4)))
    if 'Uncertainty_Sigma' in rd.columns:
        upper = rd[fc_col] + 1.96 * rd['Uncertainty_Sigma']
        lower = rd[fc_col] - 1.96 * rd['Uncertainty_Sigma']
        fig.add_trace(go.Scatter(x=list(range(len(rd))), y=upper, mode='lines',
                                 name='Upper CI', line=dict(color='lightgreen', dash='dash')))
        fig.add_trace(go.Scatter(x=list(range(len(rd))), y=lower, mode='lines',
                                 name='Lower CI', line=dict(color='lightgreen', dash='dash'),
                                 fill='tonexty', fillcolor='rgba(29,158,117,0.1)'))
    fig.update_layout(title=f"{sel} — Forecast with Uncertainty",
                      xaxis_title="Period", yaxis_title="XLPE Demand (M tons)",
                      hovermode='x unified', height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Detailed Risk Table (last 10 records)")
    st.dataframe(rd.tail(10), use_container_width=True)

    if 'Uncertainty_Sigma' in rd.columns:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **Uncertainty Quantification:**
            - **σ** = std deviation of predictions across model trees
            - **Safety Stock** = 1.96σ (95% CI)
            - Lower σ → higher confidence, less buffer needed
            """)
        with c2:
            if 'Hedging_Factor' in rd.columns:
                st.markdown("""
                **Price-Based Hedging:**
                - **1.05×** — Buy 5% extra when prices are falling
                - **0.95×** — Buy 5% less when prices are high
                - **1.00×** — Normal order at stable prices
                """)


# ============================================================
# PAGE: INFRASTRUCTURE SCENARIO
# ============================================================
def page_infrastructure(data):
    st.header("🏗️ Infrastructure-Based Scenario Forecasting")

    if data['scenario_forecast'] is None:
        st.warning("No scenario forecast found. Run Models_Advanced.py first.")
        return

    sc = data['scenario_forecast'].iloc[0]
    st.success("✅ 2026 Infrastructure Scenario Ready")

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Base Market Demand",
                        f"{sc['Base_Demand_Million_Tons']:.3f} M tons",
                        f"{sc['Base_Demand_Million_Tons']*1e6:,.0f} tons")
    with c2: st.metric("Infrastructure Contribution",
                        f"{sc['Infrastructure_Million_Tons']:.3f} M tons",
                        f"{sc['Infrastructure_Percentage']:.1f}% of total")
    with c3: st.metric("TOTAL 2026 Demand",
                        f"{sc['Total_Demand_Million_Tons']:.3f} M tons",
                        f"{sc['Total_Demand_Tons']:,.0f} tons")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(data=[go.Pie(
            labels=['Market Demand', 'Infrastructure'],
            values=[sc['Base_Demand_Million_Tons'], sc['Infrastructure_Million_Tons']],
            hole=.35, marker_colors=['#185FA5', '#E24B4A']
        )])
        fig.update_layout(title="2026 Demand Components", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(data=[go.Bar(
            x=['Base Demand', 'Infrastructure', 'Total'],
            y=[sc['Base_Demand_Million_Tons'], sc['Infrastructure_Million_Tons'],
               sc['Total_Demand_Million_Tons']],
            marker_color=['#185FA5', '#E24B4A', '#1D9E75'],
            text=[f"{sc['Base_Demand_Million_Tons']:.3f}M",
                  f"{sc['Infrastructure_Million_Tons']:.3f}M",
                  f"{sc['Total_Demand_Million_Tons']:.3f}M"],
            textposition='outside'
        )])
        fig.update_layout(title="Demand Values (M tons)", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Methodology")
    st.markdown("""
    **Formula:**
    ```
    Total Cable Mass ≈ Infrastructure Area (km²) × Cable Density (km/km²) × Cable Weight (t/km)
    ```
    **2026 calculation:** 8,779.9 km² × 100 km/km² × 1 t/km = **878,000 tons**
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Supply Chain Requirements:**
        - Current capacity: ~270K tons/year
        - 2026 requirement: 1,267K tons
        - **Scale-up: 4.7× current capacity**
        """)
    with c2:
        st.markdown(f"""
        **Financial Impact:**
        - Total: {sc['Total_Demand_Tons']:,.0f} tons @ $1,500/ton
        - **Revenue: ${sc['Total_Demand_Tons']*1500:,.0f}**
        - Infra alone: **${sc['Infrastructure_Million_Tons']*1e6*1500:,.0f}**
        """)

    if os.path.exists('outputs/scenario_forecast_2026_report.txt'):
        with open('outputs/scenario_forecast_2026_report.txt', 'r') as f:
            st.download_button("📥 Download Full Scenario Report", f.read(),
                               "2026_Infrastructure_Scenario_Report.txt", "text/plain")


# ============================================================
# PAGE: INVENTORY OPTIMIZATION
# ============================================================
def page_inventory(data):
    st.header("📦 AI-Driven Inventory Optimization")
    st.markdown("""
    <div class="info-box">
      Enter your actual cost parameters. Results are validated against ensemble forecasts.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🏭 Cost Parameters")
    c1, c2, c3 = st.columns(3)
    with c1: material_cost = st.number_input("XLPE Material Cost ($/ton)", 1000, 5000, 2350)
    with c2: holding_cost  = st.number_input("Holding Cost ($/ton/month)", 10, 200, 75)
    with c3: ordering_cost = st.number_input("Order Processing Cost ($/order)", 1000, 20000, 7500)

    c4, c5, c6 = st.columns(3)
    with c4: stockout_cost   = st.number_input("Stockout Penalty ($/ton)", 100, 5000, 1200)
    with c5: service_level   = st.slider("Service Level (%)", 80, 99, 95) / 100.0
    with c6: lead_time_days  = st.number_input("Lead Time (days)", 7, 90, 30)

    if st.button("🚀 Run Inventory Optimisation", type="primary"):
        with st.spinner("Calculating optimal inventory policy…"):
            try:
                optimizer = InventoryOptimizer(
                    forecast_path='outputs/Ensemble_Risk_Analysis.csv',
                    historical_path='historical_xlpe_demand.xlsx'
                )
                results = optimizer.optimize({
                    'material_cost': material_cost,
                    'holding_cost':  holding_cost,
                    'ordering_cost': ordering_cost,
                    'stockout_cost': stockout_cost,
                    'service_level': service_level,
                    'lead_time_days': lead_time_days,
                })
                st.success("✅ Optimisation complete!")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    ss = results['inventory_policy']['safety_stock']
                    st.metric("Safety Stock", f"{ss:.4f} M tons", f"{ss*1000:.0f} tons")
                with c2:
                    rop = results['inventory_policy']['reorder_point']
                    st.metric("Reorder Point", f"{rop:.4f} M tons", f"{rop*1000:.0f} tons")
                with c3:
                    eoq = results['inventory_policy']['economic_order_quantity']
                    st.metric("EOQ", f"{eoq:.4f} M tons", f"{eoq*1000:.0f} tons")
                with c4:
                    total_cost = results['cost_analysis']['total_annual_cost']
                    st.metric("Annual Cost", f"${total_cost:,.0f}", "Total")

                # Cost Breakdown
                st.markdown("---")
                st.subheader("💰 Cost Breakdown")
                costs = results['cost_analysis']
                fig = go.Figure(data=[go.Pie(
                    labels=['Ordering', 'Holding', 'Material', 'Stockout Risk'],
                    values=[costs['ordering_cost'], costs['holding_cost'],
                            costs['material_cost'], costs['stockout_cost']],
                    hole=.3,
                    marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                )])
                fig.update_layout(title="Annual Inventory Cost Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Cost Details:**")
                    st.markdown(f"- **Ordering:** ${costs['ordering_cost']:,.0f}")
                    st.markdown(f"- **Holding:** ${costs['holding_cost']:,.0f}")
                    st.markdown(f"- **Material:** ${costs['material_cost']:,.0f}")
                    st.markdown(f"- **Stockout Risk:** ${costs['stockout_cost']:,.0f}")
                    st.markdown("---")
                    st.markdown(f"### **Total: ${costs['total_annual_cost']:,.0f}**")
                with c2:
                    pm = results['performance_metrics']
                    st.markdown("**Performance Metrics:**")
                    st.markdown(f"- **Service Level:** {pm['service_level_percent']:.1f}%")
                    st.markdown(f"- **Orders/Year:** {costs['orders_per_year']:.1f}")
                    st.markdown(f"- **Avg Inventory:** {costs['avg_inventory']:.4f} M tons")
                    st.markdown(f"- **Inventory Turnover:** {pm['inventory_turnover']:.2f}")
                    st.markdown(f"- **Avg Cover Days:** {pm['avg_inventory_cover_days']:.1f} days")

                # Forecast Information
                st.markdown("---")
                st.subheader("🔮 AI Forecast Information")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Monthly Demand Forecast",
                               f"{results['ai_forecast_used']['monthly_demand']:.4f} M tons",
                               "From AI Ensemble")
                with c2:
                    st.metric("AI Uncertainty (σ)",
                               f"{results['ai_forecast_used']['uncertainty_sigma']:.6f}",
                               "Used for safety stock")
                with c3:
                    hf = results['ai_forecast_used'].get('hedging_factor', 1.0)
                    if hf != 1.0:
                        st.metric("Hedging Adjustment", f"{hf:.2f}×", "Price-based")
                    else:
                        st.metric("Lead Time Demand",
                                   f"{results['inventory_policy']['lead_time_demand']:.4f} M tons",
                                   f"{results['inventory_policy']['lead_time_days']} days")

                # What-If Scenarios
                st.markdown("---")
                st.subheader("🔍 What-If Scenario Analysis")
                sc1, sc2, sc3 = st.columns(3)
                user_inputs_local = {
                    'material_cost': material_cost, 'holding_cost': holding_cost,
                    'ordering_cost': ordering_cost, 'stockout_cost': stockout_cost,
                    'service_level': service_level, 'lead_time_days': lead_time_days,
                }
                with sc1:
                    if st.button("Test +20% Material Price", key="price_up"):
                        with st.spinner("Running scenario…"):
                            sr = optimizer.run_what_if_scenario(user_inputs_local, {'material_cost': 1.2})
                            new = sr['results']['cost_analysis']['total_annual_cost']
                            chg = ((new - total_cost) / total_cost) * 100
                            st.success(f"New cost: ${new:,.0f} ({chg:+.1f}%)")
                with sc2:
                    if st.button("Test -20% Storage Cost", key="storage_down"):
                        with st.spinner("Running scenario…"):
                            sr = optimizer.run_what_if_scenario(user_inputs_local, {'holding_cost': 0.8})
                            new = sr['results']['cost_analysis']['total_annual_cost']
                            chg = ((new - total_cost) / total_cost) * 100
                            st.success(f"New cost: ${new:,.0f} ({chg:+.1f}%)")
                with sc3:
                    if st.button("Test +50% Order Cost", key="order_up"):
                        with st.spinner("Running scenario…"):
                            sr = optimizer.run_what_if_scenario(user_inputs_local, {'ordering_cost': 1.5})
                            new_eoq = sr['results']['inventory_policy']['economic_order_quantity']
                            chg = ((new_eoq - eoq) / eoq) * 100
                            st.success(f"New EOQ: {new_eoq:.4f}M ({chg:+.1f}%)")

                # Download Results
                st.markdown("---")
                st.subheader("📥 Download Results")
                results_json = json.dumps(results, indent=4)
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("📄 Download Full Results (JSON)", results_json,
                                       "inventory_optimization_results.json", "application/json")
                with c2:
                    saved_files = optimizer.save_results(results)
                    st.info(f"Results saved to:\n{saved_files['json']}\n{saved_files['csv']}")

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
                st.error(f"Optimisation error: {e}")
                st.info("Check that all required files are generated and paths are correct.")
    else:
        # Default view before running optimisation
        st.markdown("---")
        st.subheader("ℹ️ How It Works")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **Inventory Optimisation Process:**
            1. **Loads your AI forecasts** from Ensemble model
            2. **Uses AI uncertainty** for safety stock calculations
            3. **Applies your cost inputs** to EOQ formula
            4. **Generates actionable recommendations**

            **No new forecasting** — uses your existing 98.6% accurate AI predictions.
            """)
        with c2:
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


# ============================================================
# PAGE: FORECASTING TOOL
# ============================================================
def page_forecasting_tool(data):
    st.header("🔮 Forecasting Tool")

    # Drop non-feature cols
    drop_cols = ['Date', 'xlpe_demand_Million_tons',
                 'construction_output_index', 'urbanization_rate',
                 'infrastructure_investment', 'lag_1', 'lag_3', 'lag_12', 'rolling_mean_3']
    historical_clean = data['historical_data'].drop(
        columns=[c for c in drop_cols if c in data['historical_data'].columns], errors='ignore')
    features = [f for f in data['metadata']['features']
                if f not in ['construction_output_index', 'urbanization_rate',
                             'infrastructure_investment', 'lag_1', 'lag_3', 'lag_12', 'rolling_mean_3']
                and f in historical_clean.columns]

    st.markdown("""
    <div class="info-box">
      Adjust macroeconomic inputs below and click <b>Generate Forecast</b> to predict XLPE demand.
      Optionally add planned infrastructure deployment.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📥 Input Parameters")
    latest = historical_clean.iloc[-1]
    c1, c2 = st.columns(2)
    input_vals = {}
    for i, feat in enumerate(features):
        with (c1 if i % 2 == 0 else c2):
            input_vals[feat] = st.number_input(
                feat.replace('_', ' ').title(),
                value=float(latest[feat]), format="%.6f", key=feat
            )

    st.markdown("---")
    st.subheader("🏗️ Optional: Add Infrastructure Contribution")
    enable_infra = st.checkbox("Include planned infrastructure deployment")
    infrastructure_tons = 0

    if enable_infra:
        method = st.radio("Input method:", ["Direct (tons)", "Calculate from Area (km²)"])
        if method == "Direct (tons)":
            infrastructure_tons = st.number_input(
                "Planned Infrastructure (tons)", 0, 10_000_000, 878_000, step=1000)
        else:
            area    = st.number_input("Infrastructure Area (km²)", 0.0, value=8779.9, step=100.0)
            density = st.slider("Cable Density (km/km²)", 50, 200, 100, step=10)
            weight  = st.slider("Cable Weight (t/km)", 0.5, 2.0, 1.0, step=0.1)
            infrastructure_tons = int(area * density * weight)
            st.success(f"Calculated: **{infrastructure_tons:,} tons**")

    if st.button("🚀 Generate Forecast", type="primary"):
        input_df = pd.DataFrame([input_vals])

        # Synthetic infrastructure proxy features
        input_df['construction_output_index'] = (
            100 + (input_df['gdp_growth_rate'] * 2.5)
            if 'gdp_growth_rate' in input_df else 100.0)

        elec_col = 'Total electricity consumption, Middle East'
        if elec_col in input_df.columns and elec_col in data['historical_data'].columns:
            emax = data['historical_data'][elec_col].max()
            input_df['urbanization_rate'] = (input_df[elec_col] / emax) * 100
        else:
            input_df['urbanization_rate'] = 50.0

        if 'infrastructure_investment' in data['metadata']['features']:
            input_df['infrastructure_investment'] = (
                data['historical_data']['xlpe_demand_Million_tons'].mean()
                if 'xlpe_demand_Million_tons' in data['historical_data'].columns else 0.27)

        # Lag features
        if 'xlpe_demand_Million_tons' in data['historical_data'].columns:
            ds = data['historical_data']['xlpe_demand_Million_tons']
            input_df['lag_1']         = ds.iloc[-1]  if len(ds) >= 1  else 0.27
            input_df['lag_3']         = ds.iloc[-3]  if len(ds) >= 3  else 0.27
            input_df['lag_12']        = ds.iloc[-12] if len(ds) >= 12 else 0.27
            input_df['rolling_mean_3']= ds.iloc[-3:].mean() if len(ds) >= 3 else 0.27

        base_pred   = data['best_model'].predict(input_df)[0]
        infra_mt    = infrastructure_tons / 1_000_000
        total_pred  = base_pred + infra_mt

        st.markdown("---")
        st.success("✅ Forecast Generated!")

        if enable_infra:
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Base Market Demand", f"{base_pred:.6f} M tons",
                                f"{base_pred*1e6:,.0f} tons")
            with c2: st.metric("Infrastructure",     f"{infra_mt:.6f} M tons",
                                f"{infrastructure_tons:,} tons")
            with c3:
                ip = (infra_mt / total_pred * 100) if total_pred > 0 else 0
                st.metric("TOTAL Demand", f"{total_pred:.6f} M tons",
                           f"{ip:.1f}% infrastructure")
            with c4:
                avg = data['historical_data']['xlpe_demand_Million_tons'].mean()
                st.metric("Growth vs Avg", f"+{((total_pred-avg)/avg)*100:.1f}%")

            fig = go.Figure(data=[go.Bar(
                x=['Base Market', 'Infrastructure', 'Total'],
                y=[base_pred, infra_mt, total_pred],
                marker_color=['#185FA5', '#E24B4A', '#1D9E75'],
                text=[f"{base_pred:.4f}M", f"{infra_mt:.4f}M", f"{total_pred:.4f}M"],
                textposition='outside'
            )])
            fig.update_layout(title="Demand Breakdown", yaxis_title="Million Tons",
                               height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Predicted Demand",    f"{base_pred:.6f} M tons")
            with c2:
                avg = data['historical_data']['xlpe_demand_Million_tons'].mean()
                st.metric("vs Historical Avg", f"{((base_pred-avg)/avg)*100:+.2f}%")
            with c3:
                ss  = data['inventory_results']['safety_stock_million_tons']
                rop = data['inventory_results']['reorder_point_million_tons']
                if base_pred < ss:
                    st.metric("Inventory Status", "SAFE", "✅ Current stock sufficient")
                elif base_pred < rop:
                    st.metric("Inventory Status", "WATCH", "⚠️ Monitor closely")
                else:
                    st.metric("Inventory Status", "ORDER", "🔴 Reorder needed")


# ============================================================
# PAGE: ABOUT
# ============================================================
def page_about():
    st.header("ℹ️ About This Project")
    st.markdown("""
    ## ARABCAB Scientific Competition
    ### AI-Based Demand Forecasting & Inventory Optimization

    **Competition Focus:** Cable & Metals Industry (Egypt • Bahrain • UAE)

    ---

    ### Project Objectives

    This AI-powered solution addresses critical challenges in the cable manufacturing industry:

    - **Demand Volatility:** Unpredictable fluctuations in XLPE (Cross-Linked Polyethylene) demand
    - **Price Volatility:** Unstable raw material costs affecting profitability
    - **Inventory Inefficiency:** Balancing overstock and stockout risks
    - **Infrastructure Planning:** Large-scale deployment forecasting (878K tons for 2026)

    ---

    ### Machine Learning Innovation

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

    ### Team
    Salma Elmarakby, Marina Nazeh, Omar Al-Hussein, Mariam Abdo, Mennatallah Zaid
    *The American University in Cairo — Team AUC*
    """)


# ============================================================
# MAIN ROUTER
# ============================================================
def main():
    if not st.session_state.authenticated:
        show_login()
        return

    data = load_data()
    if data is None:
        return

    user = st.session_state.user_info
    show_topbar()
    current_page = show_sidebar(user["permissions"])

    if current_page not in user["permissions"]:
        show_access_denied(current_page)
        return

    if   current_page == "Overview":               page_overview(data)
    elif current_page == "Model Performance":       page_model_performance(data)
    elif current_page == "Risk Analysis":           page_risk_analysis(data)
    elif current_page == "Infrastructure Scenario": page_infrastructure(data)
    elif current_page == "Inventory Optimization":  page_inventory(data)
    elif current_page == "Forecasting Tool":        page_forecasting_tool(data)
    elif current_page == "About":                   page_about()

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#aaa;padding:16px;font-size:12px'>
      CableFlow-AI · ARABCAB Competition 2024–2025 · Team AUC
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()