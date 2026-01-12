"""
================================================================================
MODEL 1: CABLE HEALTH PREDICTION MODEL
================================================================================
ARABCAB Scientific Competition - AI-Based Demand Forecasting

PURPOSE:
    Predict cable health from utility inspection data using EXPLAINABLE models.
    
INPUT DATA (15-KV XLPE Cable.xlsx):
    - ID: Cable identifier
    - Age: Cable age in years
    - Partial Discharge: PD measurement (0-1 normalized)
    - Visual Condition: Good/Medium/Poor
    - Neutral Corrosion: Corrosion index (0-1)
    - Loading: Load value
    - Health Index: 1-5 scale (target variable)

MODELS:
    1. Health Index Prediction (Ridge Regression) - Predicts 1-5 health score
    2. Risk Level Classification (Logistic Regression) - Low/Medium/High/Critical

WHY LINEAR REGRESSION (Explainable AI)?
    - Coefficients show EXACTLY how each feature impacts predictions
    - Required for regulatory compliance in utilities
    - Judges can understand WHY the model makes decisions
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = '15-KV XLPE Cable.xlsx'
MODEL_DIR = 'models'
RANDOM_STATE = 42

# Features from real data
NUMERIC_FEATURES = ['Age', 'Partial Discharge', 'Neutral Corrosion', 'Loading']
CATEGORICAL_FEATURES = ['Visual Condition']

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(filepath):
    print("\n" + "="*60)
    print("STEP 1: LOADING REAL CABLE DATA")
    print("="*60)
    
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()  # Clean column names
    
    print(f"[OK] Loaded {len(df)} cable records from {filepath}")
    print(f"[OK] Columns: {df.columns.tolist()}")
    
    # One-hot encode Visual Condition
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
    
    # Get all feature columns
    feature_cols = NUMERIC_FEATURES.copy()
    for cat in CATEGORICAL_FEATURES:
        feature_cols.extend([c for c in df_encoded.columns if c.startswith(cat)])
    
    print(f"[OK] Features for model: {feature_cols}")
    
    # Add derived columns
    # Risk Level based on Health Index (1-5 scale)
    def get_risk_level(health):
        if health >= 5:
            return 'Low'
        elif health >= 4:
            return 'Medium'
        elif health >= 3:
            return 'High'
        else:
            return 'Critical'
    
    df['risk_level'] = df['Health Index'].apply(get_risk_level)
    df_encoded['risk_level'] = df['risk_level']
    
    # Replacement Urgency (years) - Health 1→1yr, Health 5→15yr
    df['replacement_urgency_years'] = ((df['Health Index'] / 5) * 15).clip(0.5, 15).round(1)
    df_encoded['replacement_urgency_years'] = df['replacement_urgency_years']
    
    # XLPE Demand estimation
    cable_length_km = 2.0  # Average 15kV cable length
    voltage_multiplier = 1.5  # MV cable
    urgency_factor = (15 - df['replacement_urgency_years']) / 15
    df['xlpe_demand_tons'] = (cable_length_km * voltage_multiplier * 0.5 * (1 + urgency_factor)).round(2)
    df_encoded['xlpe_demand_tons'] = df['xlpe_demand_tons']
    
    print(f"\n[STATS] Health Index Distribution:")
    print(df['Health Index'].value_counts().sort_index().to_string())
    
    print(f"\n[STATS] Risk Level Distribution:")
    print(df['risk_level'].value_counts().to_string())
    
    return df, df_encoded, feature_cols

# ============================================================================
# MODEL 1A: HEALTH INDEX (Ridge Regression)
# ============================================================================
def train_health_model(df_encoded, feature_cols):
    print("\n" + "="*60)
    print("STEP 2: TRAINING HEALTH INDEX MODEL (Ridge Regression)")
    print("="*60)
    
    X = df_encoded[feature_cols]
    y = df_encoded['Health Index']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    # Metrics
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'cv_r2': cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2').mean()
    }
    
    print(f"\n[PERFORMANCE] HEALTH INDEX MODEL PERFORMANCE:")
    print(f"   R² Score:     {metrics['r2']:.4f}")
    print(f"   MAE:          {metrics['mae']:.3f} (on 1-5 scale)")
    print(f"   RMSE:         {metrics['rmse']:.3f}")
    print(f"   Cross-Val R²: {metrics['cv_r2']:.4f}")
    
    # Feature importance (coefficients)
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_,
        'abs_importance': np.abs(model.coef_)
    }).sort_values('abs_importance', ascending=False)
    
    print("\n[ANALYSIS] FEATURE IMPORTANCE (Explainable AI):")
    print("   (+ improves health, - degrades health)\n")
    for _, row in coef_df.iterrows():
        sign = "+" if row['coefficient'] > 0 else "-"
        print(f"   {sign} {row['feature']:<25} {row['coefficient']:>8.4f}")
    
    return model, scaler, metrics, coef_df

# ============================================================================
# MODEL 1B: RISK LEVEL (Logistic Regression)
# ============================================================================
def train_risk_model(df_encoded, feature_cols):
    print("\n" + "="*60)
    print("STEP 3: TRAINING RISK CLASSIFIER (Logistic Regression)")
    print("="*60)
    
    X = df_encoded[feature_cols]
    le = LabelEncoder()
    y = le.fit_transform(df_encoded['risk_level'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                         random_state=RANDOM_STATE, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n[PERFORMANCE] RISK CLASSIFIER ACCURACY: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return model, scaler, le, accuracy

# ============================================================================
# MODEL 1C: REPLACEMENT URGENCY (Ridge Regression)
# ============================================================================
def train_urgency_model(df_encoded, feature_cols):
    print("\n" + "="*60)
    print("STEP 4: TRAINING URGENCY MODEL (Ridge Regression)")
    print("="*60)
    
    X = df_encoded[feature_cols]
    y = df_encoded['replacement_urgency_years']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    print(f"\n[PERFORMANCE] URGENCY MODEL PERFORMANCE:")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   MAE:      {metrics['mae']:.2f} years")
    print(f"   RMSE:     {metrics['rmse']:.2f} years")
    
    return model, scaler, metrics

# ============================================================================
# AGGREGATE DATA FOR MODEL 2 (Market Demand)
# ============================================================================
def aggregate_for_model2(df):
    print("\n" + "="*60)
    print("STEP 5: AGGREGATING DATA FOR MODEL 2 (Market Demand)")
    print("="*60)
    
    # Aggregation by Health Index
    health_demand = df.groupby('Health Index').agg({
        'xlpe_demand_tons': 'sum',
        'ID': 'count',
        'Age': 'mean',
        'Partial Discharge': 'mean',
        'Neutral Corrosion': 'mean'
    }).reset_index()
    health_demand.columns = ['health_index', 'total_xlpe_tons', 'cable_count', 
                              'avg_age', 'avg_pd', 'avg_corrosion']
    
    # Aggregation by Risk Level
    risk_demand = df.groupby('risk_level').agg({
        'xlpe_demand_tons': 'sum',
        'ID': 'count',
        'replacement_urgency_years': 'mean'
    }).reset_index()
    risk_demand.columns = ['risk_level', 'total_xlpe_tons', 'cable_count', 'avg_urgency']
    
    # Aggregation by Urgency Band
    df['urgency_band'] = pd.cut(
        df['replacement_urgency_years'],
        bins=[0, 3, 6, 10, 15],
        labels=['Immediate (0-3yr)', 'Short-term (3-6yr)', 
                'Medium-term (6-10yr)', 'Long-term (10-15yr)']
    )
    
    urgency_demand = df.groupby('urgency_band', observed=True).agg({
        'xlpe_demand_tons': 'sum',
        'ID': 'count'
    }).reset_index()
    urgency_demand.columns = ['urgency_band', 'xlpe_demand_tons', 'cable_count']
    
    # Save for Model 2
    health_demand.to_csv('data/model2_health_demand.csv', index=False)
    risk_demand.to_csv('data/model2_risk_demand.csv', index=False)
    urgency_demand.to_csv('data/model2_urgency_demand.csv', index=False)
    
    print("\n[STATS] XLPE DEMAND BY RISK LEVEL:")
    print(risk_demand.to_string(index=False))
    
    print("\n[STATS] XLPE DEMAND BY URGENCY:")
    print(urgency_demand.to_string(index=False))
    
    print(f"\n[STATS] TOTAL XLPE DEMAND: {df['xlpe_demand_tons'].sum():.2f} tons")
    
    print("\n[SAVED] data/model2_health_demand.csv")
    print("[SAVED] data/model2_risk_demand.csv")
    print("[SAVED] data/model2_urgency_demand.csv")
    
    return health_demand, risk_demand, urgency_demand

# ============================================================================
# SAVE MODELS
# ============================================================================
def save_models(health_model, health_scaler, health_coef,
                risk_model, risk_scaler, risk_encoder,
                urgency_model, urgency_scaler, feature_cols):
    print("\n" + "="*60)
    print("STEP 6: SAVING MODELS")
    print("="*60)
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Health model
    joblib.dump(health_model, f'{MODEL_DIR}/health_regressor.pkl')
    joblib.dump(health_scaler, f'{MODEL_DIR}/health_scaler.pkl')
    
    # Risk model
    joblib.dump(risk_model, f'{MODEL_DIR}/risk_classifier.pkl')
    joblib.dump(risk_scaler, f'{MODEL_DIR}/risk_scaler.pkl')
    joblib.dump(risk_encoder, f'{MODEL_DIR}/risk_encoder.pkl')
    
    # Urgency model
    joblib.dump(urgency_model, f'{MODEL_DIR}/urgency_regressor.pkl')
    joblib.dump(urgency_scaler, f'{MODEL_DIR}/urgency_scaler.pkl')
    
    # Feature names and coefficients
    joblib.dump(feature_cols, f'{MODEL_DIR}/feature_names.pkl')
    health_coef.to_csv(f'{MODEL_DIR}/health_coefficients.csv', index=False)
    
    print("[OK] All models saved to 'models/' directory")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*60)
    print("  ARABCAB - MODEL 1: CABLE HEALTH PREDICTION")
    print("  Using Real 15-KV XLPE Cable Data (2500 records)")
    print("="*60)
    
    # Load real data
    df, df_encoded, feature_cols = load_data(DATA_PATH)
    
    # Train models
    health_model, health_scaler, health_metrics, health_coef = train_health_model(df_encoded, feature_cols)
    risk_model, risk_scaler, risk_encoder, risk_acc = train_risk_model(df_encoded, feature_cols)
    urgency_model, urgency_scaler, urgency_metrics = train_urgency_model(df_encoded, feature_cols)
    
    # Aggregate for Model 2
    aggregate_for_model2(df)
    
    # Save models
    save_models(health_model, health_scaler, health_coef,
                risk_model, risk_scaler, risk_encoder,
                urgency_model, urgency_scaler, feature_cols)
    
    # Final summary
    print("\n" + "="*60)
    print("  MODEL 1 COMPLETE - SUMMARY")
    print("="*60)
    print(f"  HEALTH INDEX MODEL (Ridge Regression)")
    print(f"    R-squared: {health_metrics['r2']:.4f}")
    print(f"    MAE: {health_metrics['mae']:.3f} (1-5 scale)")
    print(f"  RISK CLASSIFIER (Logistic Regression)")
    print(f"    Accuracy: {risk_acc:.2%}")
    print(f"  URGENCY MODEL (Ridge Regression)")
    print(f"    R-squared: {urgency_metrics['r2']:.4f}")
    print(f"    MAE: {urgency_metrics['mae']:.2f} years")
    print("="*60)
    print("[COMPLETE] Model 1 outputs ready for Model 2 (Market Demand)")
    print("[COMPLETE] Dashboard can now use these models for predictions")
    
    return {'health': health_metrics, 'risk_acc': risk_acc, 'urgency': urgency_metrics}

if __name__ == "__main__":
    results = main()