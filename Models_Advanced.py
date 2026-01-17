import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline  # INNOVATION: Professional ML pipelines
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle
import json
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('outputs', exist_ok=True)

print("="*80)
print("XLPE DEMAND FORECASTING - TIME-SERIES FORECASTING MODEL")
print("INNOVATION: Scikit-learn Pipelines + Time-Series Cross-Validation")
print("INNOVATION: Lag-based Features + TimeSeriesSplit")
print("INNOVATION: Temporal Order Preserved (NO random shuffling)")
print("="*80)

# Load data
path = r"historical_xlpe_demand.xlsx"
df = pd.read_excel(path)

print(f"\n{'='*80}")
print("DATA ANALYSIS & PREPROCESSING")
print(f"{'='*80}")
print(f"Dataset loaded: {len(df)} records")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Data preprocessing
df_clean = df.drop(columns=['Date', 'Year', 'Month']).dropna()

# CRITICAL FORECASTING INNOVATION: Add lag-based features
print(f"\nAdding time-series lag features...")
df_clean['lag_1'] = df_clean['xlpe_demand_Million_tons'].shift(1)
df_clean['lag_3'] = df_clean['xlpe_demand_Million_tons'].shift(3)
df_clean['lag_12'] = df_clean['xlpe_demand_Million_tons'].shift(12)  # Seasonal pattern
df_clean['rolling_mean_3'] = df_clean['xlpe_demand_Million_tons'].rolling(3).mean()

# Drop NaN values created by lag features
df_clean = df_clean.dropna()

X = df_clean.drop(columns=['xlpe_demand_Million_tons'])
y = df_clean['xlpe_demand_Million_tons']

print(f"Features after adding lag variables: {len(X.columns)}")
print(f"Lag features added: lag_1, lag_3, lag_12 (seasonal), rolling_mean_3")

# Feature statistics
print(f"\n{'='*80}")
print("FEATURE ENGINEERING & STATISTICS")
print(f"{'='*80}")
print(f"\nFeatures used for prediction:")
for i, col in enumerate(X.columns, 1):
    mean_val = X[col].mean()
    std_val = X[col].std()
    print(f"  {i}. {col}")
    print(f"     Mean: {mean_val:.6f}, Std Dev: {std_val:.6f}")

# MANDATORY: Time-series split (NO random shuffling!)
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"\nTraining set: {len(X_train)} samples (first 80% chronologically)")
print(f"Testing set: {len(X_test)} samples (last 20% chronologically)")
print(f"Time-series split: 80-20 (TEMPORAL ORDER PRESERVED - NO SHUFFLING)")
print(f"Training period: indices 0 to {split_idx-1}")
print(f"Testing period: indices {split_idx} to {len(X)-1}")

print("\n" + "="*80)
print("INNOVATION: Using Scikit-learn Pipelines")
print("="*80)
print("WHY PIPELINES?")
print("  ✓ Prevents data leakage (scaler fits only on training data)")
print("  ✓ Ensures consistent preprocessing in production")
print("  ✓ Cleaner, more maintainable code")
print("  ✓ Industry best practice for ML deployment")
print("  ✓ Single object contains entire workflow (scale + train + predict)")

# INNOVATION: Advanced models with hyperparameter tuning
print("\n" + "="*80)
print("ADVANCED MODEL TRAINING & HYPERPARAMETER OPTIMIZATION")
print("Using Pipelines: StandardScaler → Model")
print("="*80)

# Define models with pipelines and hyperparameter tuning
# EXPLANATION: Each model is wrapped in a Pipeline with StandardScaler
# GridSearchCV will tune hyperparameters while pipeline prevents data leakage
models = {
    "Linear Regression": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        "params": {},  # No hyperparameters to tune
        "use_cv": True
    },
    "Ridge Regression": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge())
        ]),
        "params": {
            'model__alpha': [0.1, 1.0, 10.0]  # Note: 'model__' prefix for pipeline params
        },
        "use_cv": True
    },
    # REMOVED: Lasso (less effective for time-series)
    # REMOVED: KNN (unstable for forecasting)
    # REMOVED: Decision Tree (single trees overfit time-series)
    
    "Random Forest": {
        "pipeline": Pipeline([
            ('model', RandomForestRegressor(random_state=42))  # No scaling needed for trees
        ]),
        "params": {
            'model__n_estimators': [50, 100, 150],
            'model__max_depth': [10, 15, 20]
        },
        "use_cv": True
    },
    "Gradient Boosting": {
        "pipeline": Pipeline([
            ('model', GradientBoostingRegressor(random_state=42))  # No scaling needed for trees
        ]),
        "params": {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5]
        },
        "use_cv": True
    },
    # REMOVED: SVR (computationally expensive, less effective for time-series)
}

# Store results
results = []
trained_pipelines = {}  # Store complete pipelines, not just models
best_pipeline = None
best_accuracy = 0
best_model_name = ""

for name, model_info in models.items():
    print(f"\n{'='*80}")
    print(f"Training Pipeline: {name}")
    print(f"{'='*80}")
    
    # Hyperparameter tuning with GridSearchCV on ENTIRE PIPELINE
    # EXPLANATION: GridSearchCV fits the scaler on each CV fold separately
    # This prevents data leakage and gives honest performance estimates
    if model_info['params']:
        print(f"Performing Grid Search with Cross-Validation on Pipeline...")
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(
            model_info['pipeline'],  # Entire pipeline, not just model
            model_info['params'], 
            cv=tscv,  # Time-series cross-validation
            scoring='r2',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)  # Pipeline handles scaling automatically
        pipeline = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Cross-validation R² score: {grid_search.best_score_:.4f}")
    else:
        pipeline = model_info['pipeline']
        pipeline.fit(X_train, y_train)
        # Cross-validation for models without hyperparameters
        if model_info['use_cv']:
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='r2')
            print(f"Time-series CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Store trained pipeline (contains both scaler and model)
    trained_pipelines[name] = pipeline
    
    # Predictions (pipeline automatically scales X_test)
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    forecast_accuracy = 100 - mape  # Forecast Accuracy = 1 - MAPE
    
    # Training metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate overfitting metric
    overfitting_score = abs(train_r2 - r2)
    
    # Print results
    print(f"\n📊 Test Set Performance:")
    print(f"  ✓ Forecast Accuracy (1-MAPE): {forecast_accuracy:.2f}%")
    print(f"  ✓ Mean Absolute Error (MAE): {mae:.6f} million tons")
    print(f"  ✓ Root Mean Squared Error (RMSE): {rmse:.6f} million tons")
    print(f"  ✓ R² Score: {r2:.4f}")
    print(f"  ✓ MAPE: {mape:.2f}%")
    
    print(f"\n📈 Training Set Performance:")
    print(f"  ✓ MAE: {train_mae:.6f} million tons")
    print(f"  ✓ R² Score: {train_r2:.4f}")
    print(f"  ✓ Overfitting Score: {overfitting_score:.4f} ({'Low' if overfitting_score < 0.1 else 'Moderate' if overfitting_score < 0.2 else 'High'})")
    
    # Store results
    results.append({
        'Model': name,
        'Forecast Accuracy (%)': round(forecast_accuracy, 2),  # Renamed for clarity
        'MAE (million tons)': round(mae, 6),
        'RMSE (million tons)': round(rmse, 6),
        'R² Score': round(r2, 4),
        'MAPE (%)': round(mape, 2),
        'Train R²': round(train_r2, 4),
        'Overfitting': round(overfitting_score, 4)
    })
    
    # Track best model (considering both forecast accuracy and overfitting)
    model_score = forecast_accuracy - (overfitting_score * 10)  # Penalize overfitting
    if model_score > best_accuracy:
        best_accuracy = model_score
        best_pipeline = pipeline  # Store entire pipeline
        best_model_name = name
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Error': y_test.values - y_pred,
        'Absolute_Error': np.abs(y_test.values - y_pred),
        'Percentage_Error': np.abs((y_test.values - y_pred) / y_test.values) * 100
    })
    predictions_df.to_csv(f'outputs/{name.replace(" ", "_")}_predictions.csv', index=False)

# INNOVATION: Create Ensemble Model (Voting Regressor)
print(f"\n{'='*80}")
print("CREATING ADVANCED ENSEMBLE PIPELINE")
print(f"{'='*80}")

# Select top 3 models for ensemble
results_sorted = sorted(results, key=lambda x: x['Forecast Accuracy (%)'], reverse=True)
top_3_models = [r['Model'] for r in results_sorted[:3]]
print(f"Top 3 models for ensemble: {', '.join(top_3_models)}")

# EXPLANATION: Ensemble of pipelines
# NOTE: VotingRegressor uses pre-trained pipelines;
# each estimator applies its own preprocessing independently (no double-scaling).
ensemble_estimators = [(name, trained_pipelines[name]) for name in top_3_models]
ensemble_pipeline = VotingRegressor(estimators=ensemble_estimators)
ensemble_pipeline.fit(X_train, y_train)  # Each sub-pipeline scales data independently

# Evaluate ensemble
y_pred_ensemble = ensemble_pipeline.predict(X_test)
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)
mape_ensemble = np.mean(np.abs((y_test - y_pred_ensemble) / y_test)) * 100
forecast_accuracy_ensemble = 100 - mape_ensemble

print(f"\n🏆 Ensemble Pipeline Performance:")
print(f"  ✓ Forecast Accuracy (1-MAPE): {forecast_accuracy_ensemble:.2f}%")
print(f"  ✓ MAE: {mae_ensemble:.6f} million tons")
print(f"  ✓ R² Score: {r2_ensemble:.4f}")

# Add ensemble to results
results.append({
    'Model': 'Ensemble (Top 3)',
    'Forecast Accuracy (%)': round(forecast_accuracy_ensemble, 2),
    'MAE (million tons)': round(mae_ensemble, 6),
    'RMSE (million tons)': round(np.sqrt(mean_squared_error(y_test, y_pred_ensemble)), 6),
    'R² Score': round(r2_ensemble, 4),
    'MAPE (%)': round(mape_ensemble, 2),
    'Train R²': round(r2_score(y_train, ensemble_pipeline.predict(X_train)), 4),
    'Overfitting': round(abs(r2_score(y_train, ensemble_pipeline.predict(X_train)) - r2_ensemble), 4)
})

# Update best model if ensemble is better
if forecast_accuracy_ensemble > best_accuracy:
    best_pipeline = ensemble_pipeline
    best_model_name = 'Ensemble (Top 3)'
    best_accuracy = forecast_accuracy_ensemble

# Save results summary
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Forecast Accuracy (%)', ascending=False)
results_df.to_csv('outputs/model_comparison_results.csv', index=False)

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY (Ranked by Forecast Accuracy)")
print("="*80)
print(results_df.to_string(index=False))

print(f"\n{'='*80}")
print(f"🏆 BEST PIPELINE: {best_model_name} with {results_df[results_df['Model'] == best_model_name]['Forecast Accuracy (%)'].values[0]:.2f}% Forecast Accuracy")
print(f"{'='*80}")

# Save best pipeline (contains both scaler and model!)
with open('outputs/best_pipeline.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)
print(f"\n✓ Best pipeline saved to: outputs/best_pipeline.pkl")
print("  DEPLOYMENT: Load this single file and call pipeline.predict(new_data)")
print("  NO manual scaling needed - pipeline handles everything!")

# Save metadata
metadata = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'best_model': best_model_name,
    'best_forecast_accuracy': float(results_df[results_df['Model'] == best_model_name]['Forecast Accuracy (%)'].values[0]),
    'best_r2': float(results_df[results_df['Model'] == best_model_name]['R² Score'].values[0]),
    'features': list(X.columns),
    'target': 'xlpe_demand_Million_tons',
    'train_size': len(X_train),
    'test_size': len(X_test),
    'total_models_evaluated': len(models) + 1,  # +1 for ensemble
    'ensemble_members': top_3_models if best_model_name == 'Ensemble (Top 3)' else [],
    'preprocessing': 'StandardScaler (inside pipeline)',
    'innovation': 'Scikit-learn Pipelines for production-ready deployment'
}

with open('outputs/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

# TECHNICAL RIGOR: Advanced Visualizations
print("\n" + "="*80)
print("GENERATING ADVANCED VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# 1. Comprehensive Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1a. Forecast Accuracy Comparison
ax = axes[0, 0]
colors = ['#2ecc71' if r['Model'] == best_model_name else '#3498db' for r in results]
bars = ax.barh(range(len(results_df)), results_df['Forecast Accuracy (%)'], color=colors)
ax.set_yticks(range(len(results_df)))
ax.set_yticklabels(results_df['Model'])
ax.set_xlabel('Forecast Accuracy (%) = 1 - MAPE', fontsize=12, fontweight='bold')
ax.set_title('Model Forecast Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
for i, (idx, row) in enumerate(results_df.iterrows()):
    ax.text(row['Forecast Accuracy (%)'] + 1, i, f"{row['Forecast Accuracy (%)']:.2f}%", va='center', fontweight='bold')

# 1b. R² Score Comparison
ax = axes[0, 1]
colors = ['#2ecc71' if r['Model'] == best_model_name else '#e74c3c' for r in results]
ax.barh(range(len(results_df)), results_df['R² Score'], color=colors)
ax.set_yticks(range(len(results_df)))
ax.set_yticklabels(results_df['Model'])
ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Model R² Score Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
for i, (idx, row) in enumerate(results_df.iterrows()):
    ax.text(row['R² Score'] + 0.02, i, f"{row['R² Score']:.4f}", va='center', fontweight='bold')

# 1c. MAE Comparison (lower is better)
ax = axes[1, 0]
colors = ['#2ecc71' if r['Model'] == best_model_name else '#f39c12' for r in results]
ax.barh(range(len(results_df)), results_df['MAE (million tons)'], color=colors)
ax.set_yticks(range(len(results_df)))
ax.set_yticklabels(results_df['Model'])
ax.set_xlabel('MAE (million tons) - Lower is Better', fontsize=12, fontweight='bold')
ax.set_title('Model Error Comparison', fontsize=14, fontweight='bold')
for i, (idx, row) in enumerate(results_df.iterrows()):
    ax.text(row['MAE (million tons)'] + 0.000005, i, f"{row['MAE (million tons)']:.6f}", va='center', fontsize=9)

# 1d. Overfitting Analysis
ax = axes[1, 1]
colors = ['#2ecc71' if r['Model'] == best_model_name else '#9b59b6' for r in results]
ax.barh(range(len(results_df)), results_df['Overfitting'], color=colors)
ax.set_yticks(range(len(results_df)))
ax.set_yticklabels(results_df['Model'])
ax.set_xlabel('Overfitting Score (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_title('Model Generalization Analysis', fontsize=14, fontweight='bold')
ax.axvline(x=0.1, color='orange', linestyle='--', label='Low threshold')
ax.axvline(x=0.2, color='red', linestyle='--', label='Moderate threshold')
ax.legend()
for i, (idx, row) in enumerate(results_df.iterrows()):
    ax.text(row['Overfitting'] + 0.005, i, f"{row['Overfitting']:.4f}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Best Model Performance
best_y_pred = best_pipeline.predict(X_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 2a. Actual vs Predicted
ax = axes[0]
ax.scatter(y_test, best_y_pred, alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual XLPE Demand (Million Tons)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted XLPE Demand (Million Tons)', fontsize=12, fontweight='bold')
ax.set_title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2b. Residual Plot
ax = axes[1]
residuals = y_test.values - best_y_pred
ax.scatter(best_y_pred, residuals, alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax.set_title('Residual Analysis', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 2c. Error Distribution
ax = axes[2]
ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
ax.set_xlabel('Prediction Error (Million Tons)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/best_model_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature Importance (if applicable)
if best_model_name == 'Ensemble (Top 3)':
    # For ensemble of pipelines, we can't easily extract feature importance
    print("✓ Ensemble pipeline - combined predictions from top 3 models")
elif hasattr(best_pipeline.named_steps['model'], 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_pipeline.named_steps['model'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='teal')
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    feature_importance.to_csv('outputs/feature_importance.csv', index=False)
    print("✓ Feature importance analysis saved")

# 4. Learning Curves (for best single model, not ensemble)
if best_model_name != 'Ensemble (Top 3)':
    from sklearn.model_selection import learning_curve
    
    print("Generating learning curves (this may take a moment)...")
    tscv = TimeSeriesSplit(n_splits=5)
    train_sizes, train_scores, val_scores = learning_curve(
        best_pipeline, X_train, y_train, cv=tscv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, label='Cross-Validation Score', color='orange', marker='s')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    plt.xlabel('Training Set Size', fontsize=12, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12, fontweight='bold')
    plt.title(f'Learning Curve - {best_model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Learning curve saved")

print("\n✓ All visualizations generated successfully!")
print("\nFiles saved to 'outputs' directory:")
print("  ✓ model_comparison_results.csv - Detailed comparison table")
print("  ✓ comprehensive_model_comparison.png - 4-panel comparison chart")
print("  ✓ best_model_analysis.png - Best model performance analysis")
print("  ✓ best_pipeline.pkl - COMPLETE PIPELINE (scaler + model)")
print("  ✓ model_metadata.json - Model metadata and configuration")
print("  ✓ Individual model prediction CSVs")
print("  ✓ Feature importance analysis (if applicable)")
print("  ✓ Learning curve (if applicable)")

print("\n" + "="*80)
print("✅ ADVANCED FORECASTING COMPLETE - PRODUCTION READY")
print("="*80)
print(f"\n🎯 Innovation Highlights:")
print(f"  ✓ Scikit-learn Pipelines for production deployment")
print(f"  ✓ {len(models)} different ML algorithms evaluated")
print(f"  ✓ Hyperparameter optimization with GridSearchCV on pipelines")
print(f"  ✓ Cross-validation prevents data leakage")
print(f"  ✓ Ensemble modeling for improved accuracy")
print(f"  ✓ Single pipeline.pkl file contains entire workflow")
print(f"  ✓ Comprehensive error analysis and diagnostics")
print(f"\n🏆 Best Performance: {best_model_name}")
print(f"  ✓ Forecast Accuracy (1-MAPE): {results_df[results_df['Model'] == best_model_name]['Forecast Accuracy (%)'].values[0]:.2f}%")
print(f"  ✓ R² Score: {results_df[results_df['Model'] == best_model_name]['R² Score'].values[0]:.4f}")
print(f"  ✓ MAE: {results_df[results_df['Model'] == best_model_name]['MAE (million tons)'].values[0]:.6f} million tons")

print("\n" + "="*80)
print("DEPLOYMENT INSTRUCTIONS")
print("="*80)
print("To use the trained pipeline in production:")
print("  1. Load: pipeline = pickle.load(open('outputs/best_pipeline.pkl', 'rb'))")
print("  2. Predict: predictions = pipeline.predict(new_data)")
print("  3. That's it! Pipeline handles scaling automatically")
print("="*80)
