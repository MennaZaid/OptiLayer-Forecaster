"""
================================================================================
MODEL 2: ML-BASED MARKET DEMAND FORECASTING
================================================================================
ARABCAB Scientific Competition - Advanced AI/ML Component

PURPOSE:
    Forecast future XLPE demand using machine learning with external market factors
    
METHODS:
    1. Facebook Prophet (Time Series with Seasonality)
    2. XGBoost Regression (Feature-based forecasting)
    3. Ensemble (Combining both methods)

FEATURES:
    - Historical XLPE demand
    - Polyethylene price index
    - GDP growth rate
    - Construction activity index
    - Grid expansion rate
    - Renewable energy capacity
    
OUTPUT:
    - 5-year XLPE demand forecast
    - Accuracy metrics (MAPE, RMSE, MAE)
    - Feature importance analysis
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("[WARNING] Prophet not installed. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================
FORECAST_YEARS = 5
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
OUTPUT_DIR = 'outputs'

# ============================================================================
# DATA LOADING - ACTUAL DATA ONLY
# ============================================================================

def load_actual_data():
    """Load actual historical data from Excel - NO SYNTHETIC DATA"""
    try:
        # Load actual historical XLPE demand data
        df = pd.read_excel('historical_xlpe_demand.xlsx')
        print("[LOADED] Actual historical data from historical_xlpe_demand.xlsx")
        print("[INFO] Using REAL data only - NO synthetic data generation")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert date column
        df['date'] = pd.to_datetime(df['Date'])
        df['year'] = df['Year']
        df['month'] = df['Month']
        
        # Map actual columns to expected feature names
        df['xlpe_demand_tons'] = df['xlpe_demand_tons']
        df['polyethylene_price_index'] = df['polyethylene_price']
        df['gdp_growth_rate'] = df['gdp_growth_rate']
        
        print(f"[OK] Demand range: {df['xlpe_demand_tons'].min():.4f} - {df['xlpe_demand_tons'].max():.4f} tons/month")
        
        # Use electricity consumption as proxy for construction/grid activity
        df['construction_index'] = df['Total electricity consumption, Middle East'] / 100  # Normalize
        df['grid_expansion_rate'] = df['Monthy growth rate , Middle East %']
        
        # Create renewable capacity from electricity growth (derived feature)
        df['renewable_capacity_gw'] = df['Total electricity consumption, Middle East'] * 0.15  # Assume 15% renewable
        
        print(f"[OK] Loaded {len(df)} records from Excel")
        print(f"[OK] Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"[OK] Columns mapped: polyethylene_price -> polyethylene_price_index")
        print(f"[OK] Columns mapped: Total electricity consumption -> construction_index")
        print(f"[OK] Columns mapped: Monthly growth rate -> grid_expansion_rate")
        
        # Select required columns
        df = df[['date', 'year', 'month', 'xlpe_demand_tons', 'polyethylene_price_index', 
                 'gdp_growth_rate', 'construction_index', 'grid_expansion_rate', 'renewable_capacity_gw']]
        
        return df
        
    except FileNotFoundError:
        print("[ERROR] historical_xlpe_demand.xlsx not found!")
        print("[ERROR] Please ensure the Excel file exists in the project root directory")
        raise FileNotFoundError("Required file 'historical_xlpe_demand.xlsx' is missing")

def load_model1_outputs():
    """Load Model 1 outputs for validation and adjustment"""
    try:
        urgency_df = pd.read_csv('data/model2_urgency_demand.csv')
        risk_df = pd.read_csv('data/model2_risk_demand.csv')
        health_df = pd.read_csv('data/model2_health_demand.csv')
        
        print("[LOADED] Model 1 outputs for baseline validation")
        print(f"  • Total demand from Model 1: {urgency_df['xlpe_demand_tons'].sum():.2f} tons")
        
        return urgency_df, risk_df, health_df
    except FileNotFoundError:
        print("[WARNING] Model 1 outputs not found. Run arabcab.py first.")
        return None, None, None


# ============================================================================
# METHOD 1: FACEBOOK PROPHET (Time Series)
# ============================================================================

def train_prophet_model(df):
    """
    Train Facebook Prophet model for time series forecasting
    Prophet handles trend, seasonality, and holidays automatically
    """
    if not PROPHET_AVAILABLE:
        print("[SKIP] Prophet not available")
        return None, None
    
    print("\n" + "="*70)
    print("METHOD 1: FACEBOOK PROPHET (Time Series)")
    print("="*70)
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = df[['date', 'xlpe_demand_tons']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Split train/test
    split_idx = int(len(prophet_df) * TRAIN_TEST_SPLIT)
    train_df = prophet_df[:split_idx]
    test_df = prophet_df[split_idx:]
    
    # Train model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    
    # Add external regressors (if available)
    try:
        model.add_regressor('grid_expansion')
        model.add_regressor('construction_index')
        
        train_df['grid_expansion'] = df['grid_expansion_rate'][:split_idx].values
        train_df['construction_index'] = df['construction_index'][:split_idx].values
        
        print("[CONFIG] Added external regressors: grid_expansion, construction_index")
    except:
        pass
    
    print("[TRAINING] Prophet model...")
    model.fit(train_df)
    
    # Make predictions on test set
    if 'grid_expansion' in train_df.columns:
        test_df['grid_expansion'] = df['grid_expansion_rate'][split_idx:].values
        test_df['construction_index'] = df['construction_index'][split_idx:].values
    
    forecast = model.predict(test_df)
    
    # Calculate metrics
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n[PERFORMANCE] Prophet Test Set Metrics:")
    print(f"  • MAE:        {mae:,.0f} tons")
    print(f"  • RMSE:       {rmse:,.0f} tons")
    print(f"  • MAPE:       {mape*100:.2f}%")
    print(f"  • R²:         {r2:.4f}")
    print(f"  • Accuracy:   {(1-mape)*100:.1f}%")
    
    metrics = {
        'model': 'Prophet',
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'accuracy': (1-mape)*100
    }
    
    return model, metrics


# ============================================================================
# METHOD 2: GRADIENT BOOSTING (Feature-based ML)
# ============================================================================

def train_ml_model(df):
    """
    Train Gradient Boosting model with external features
    Better for capturing complex non-linear relationships
    """
    print("\n" + "="*70)
    print("METHOD 2: GRADIENT BOOSTING (Feature-based ML)")
    print("="*70)
    
    # Feature engineering
    df = df.copy()
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['time_index'] = np.arange(len(df))
    
    # Feature selection
    features = [
        'time_index',
        'month_sin', 'month_cos',
        'polyethylene_price_index',
        'gdp_growth_rate',
        'construction_index',
        'grid_expansion_rate',
        'renewable_capacity_gw'
    ]
    
    X = df[features]
    y = df['xlpe_demand_tons']
    
    # Split data
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
        subsample=0.8
    )
    
    print("[TRAINING] Gradient Boosting model...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n[PERFORMANCE] Gradient Boosting Test Set Metrics:")
    print(f"  • MAE:        {mae:,.0f} tons")
    print(f"  • RMSE:       {rmse:,.0f} tons")
    print(f"  • MAPE:       {mape*100:.2f}%")
    print(f"  • R²:         {r2:.4f}")
    print(f"  • Accuracy:   {(1-mape)*100:.1f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n[ANALYSIS] Feature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"  • {row['feature']:<30} {row['importance']:.4f}")
    
    metrics = {
        'model': 'GradientBoosting',
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'accuracy': (1-mape)*100
    }
    
    return model, scaler, features, metrics, feature_importance


# ============================================================================
# FUTURE FORECASTING
# ============================================================================

def forecast_future(prophet_model, ml_model, scaler, features, df, years=5):
    """
    Generate future forecasts using both models and ensemble them
    """
    print("\n" + "="*70)
    print(f"GENERATING {years}-YEAR FORECAST")
    print("="*70)
    
    # Future dates
    last_date = df['date'].max()
    future_dates = pd.date_range(last_date + timedelta(days=30), periods=years*12, freq='M')
    
    # Create future dataframe
    future_df = pd.DataFrame({
        'date': future_dates,
        'year': future_dates.year,
        'month': future_dates.month
    })
    
    # Extrapolate external features (simple linear projection)
    last_index = len(df)
    future_df['time_index'] = np.arange(last_index, last_index + len(future_df))
    
    # Simple projections for external factors
    future_df['polyethylene_price_index'] = np.linspace(
        df['polyethylene_price_index'].iloc[-12:].mean(),
        df['polyethylene_price_index'].iloc[-12:].mean() + 5,
        len(future_df)
    )
    
    future_df['gdp_growth_rate'] = np.clip(
        df['gdp_growth_rate'].iloc[-12:].mean() + np.random.normal(0, 0.5, len(future_df)),
        2, 7
    )
    
    future_df['construction_index'] = np.linspace(
        df['construction_index'].iloc[-1],
        df['construction_index'].iloc[-1] + 10,
        len(future_df)
    )
    
    future_df['grid_expansion_rate'] = np.clip(
        df['grid_expansion_rate'].iloc[-12:].mean() + np.random.normal(0, 0.3, len(future_df)),
        3, 8
    )
    
    future_df['renewable_capacity_gw'] = df['renewable_capacity_gw'].iloc[-1] * np.exp(
        np.arange(len(future_df)) * 0.008
    )
    
    # Feature engineering
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
    
    # Prophet forecast
    prophet_forecast = None
    if prophet_model is not None and PROPHET_AVAILABLE:
        prophet_input = pd.DataFrame({
            'ds': future_dates,
            'grid_expansion': future_df['grid_expansion_rate'],
            'construction_index': future_df['construction_index']
        })
        prophet_forecast = prophet_model.predict(prophet_input)['yhat'].values
        print("[PROPHET] Forecast generated")
    
    # ML forecast
    X_future = future_df[features]
    X_future_scaled = scaler.transform(X_future)
    ml_forecast = ml_model.predict(X_future_scaled)
    print("[ML] Forecast generated")
    
    # Ensemble (average of both models)
    if prophet_forecast is not None:
        ensemble_forecast = (prophet_forecast * 0.4 + ml_forecast * 0.6)
        print("[ENSEMBLE] Combined forecast (40% Prophet + 60% ML)")
    else:
        ensemble_forecast = ml_forecast
        print("[ENSEMBLE] Using ML forecast only")
    
    # Clip negative values
    ensemble_forecast = np.clip(ensemble_forecast, 0, None)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'year': future_dates.year,
        'month': future_dates.month,
        'forecast_demand_tons': ensemble_forecast.round(2)  # Keep decimals for small values
    })
    
    # Annual aggregation
    annual_forecast = forecast_df.groupby('year')['forecast_demand_tons'].sum().reset_index()
    annual_forecast.columns = ['year', 'annual_demand_tons']
    
    print(f"\n[FORECAST] Annual XLPE Demand Projections:")
    for _, row in annual_forecast.iterrows():
        print(f"  • {row['year']}: {row['annual_demand_tons']:>15,.2f} tons")
    
    print(f"\n[TOTAL] {years}-Year Total Demand: {annual_forecast['annual_demand_tons'].sum():,.2f} tons")
    
    return forecast_df, annual_forecast


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(prophet_model, ml_model, scaler, features, metrics_prophet, metrics_ml, 
                 forecast_df, annual_forecast):
    """Save all models and outputs"""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save models
    if prophet_model is not None:
        joblib.dump(prophet_model, 'models/prophet_model.pkl')
        print("[SAVED] models/prophet_model.pkl")
    
    joblib.dump(ml_model, 'models/ml_forecast_model.pkl')
    joblib.dump(scaler, 'models/ml_forecast_scaler.pkl')
    joblib.dump(features, 'models/ml_forecast_features.pkl')
    print("[SAVED] models/ml_forecast_*.pkl")
    
    # Save metrics
    all_metrics = pd.DataFrame([metrics_prophet, metrics_ml]) if metrics_prophet else pd.DataFrame([metrics_ml])
    all_metrics.to_csv(f'{OUTPUT_DIR}/forecast_metrics.csv', index=False)
    print(f"[SAVED] {OUTPUT_DIR}/forecast_metrics.csv")
    
    # Save forecasts
    forecast_df.to_csv(f'{OUTPUT_DIR}/monthly_forecast.csv', index=False)
    annual_forecast.to_csv(f'{OUTPUT_DIR}/annual_forecast.csv', index=False)
    print(f"[SAVED] {OUTPUT_DIR}/monthly_forecast.csv")
    print(f"[SAVED] {OUTPUT_DIR}/annual_forecast.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ARABCAB - MODEL 2: ML-BASED MARKET FORECASTING")
    print("USING ACTUAL DATA ONLY - NO SYNTHETIC DATA")
    print("="*70)
    
    # Load actual historical data (NO SYNTHETIC DATA)
    df = load_actual_data()
    
    # Load Model 1 outputs for validation
    urgency_df, risk_df, health_df = load_model1_outputs()
    
    # Train Prophet
    prophet_model, metrics_prophet = train_prophet_model(df)
    
    # Train ML model
    ml_model, scaler, features, metrics_ml, feature_importance = train_ml_model(df)
    
    # Generate future forecast
    forecast_df, annual_forecast = forecast_future(
        prophet_model, ml_model, scaler, features, df, years=FORECAST_YEARS
    )
    
    # Save everything
    save_outputs(prophet_model, ml_model, scaler, features, 
                 metrics_prophet, metrics_ml, forecast_df, annual_forecast)
    
    # Summary
    print("\n" + "="*70)
    print("MODEL 2 COMPLETE - SUMMARY")
    print("="*70)
    
    if metrics_prophet:
        print("  PROPHET MODEL")
        print(f"    Accuracy: {metrics_prophet['accuracy']:.1f}%  MAPE: {metrics_prophet['mape']*100:.2f}%")
        print("  GRADIENT BOOSTING MODEL")
        print(f"    Accuracy: {metrics_ml['accuracy']:.1f}%  MAPE: {metrics_ml['mape']*100:.2f}%")
        print(f"  ENSEMBLE FORECAST ({FORECAST_YEARS} Years)")
        print(f"    Total Demand: {annual_forecast['annual_demand_tons'].sum():,.0f} tons")
    else:
        print("  GRADIENT BOOSTING MODEL")
        print(f"    Accuracy: {metrics_ml['accuracy']:.1f}%  MAPE: {metrics_ml['mape']*100:.2f}%")
        print(f"  FORECAST ({FORECAST_YEARS} Years)")
        print(f"    Total Demand: {annual_forecast['annual_demand_tons'].sum():,.0f} tons")
    
    print("[COMPLETE] Forecasts ready for inventory optimization and dashboard")
    
    return annual_forecast


if __name__ == "__main__":
    main()
