import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import json
from datetime import datetime
import os

# Create output directory
os.makedirs('outputs', exist_ok=True)

print("="*80)
print("XLPE DEMAND FORECASTING - AI-BASED MODEL EVALUATION")
print("="*80)

# Load data
path = r"historical_xlpe_demand.xlsx"
df = pd.read_excel(path)

print(f"\nDataset loaded: {len(df)} records")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Data preprocessing
df_clean = df.drop(columns=['Date', 'Year', 'Month']).dropna()
X = df_clean.drop(columns=['xlpe_demand_Million_tons'])
y = df_clean['xlpe_demand_Million_tons']

print(f"\nFeatures used for prediction:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Store results
results = []
best_model = None
best_accuracy = 0
best_model_name = ""

print("\n" + "="*80)
print("MODEL TRAINING AND EVALUATION")
print("="*80)

for name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Training: {name}")
    print(f"{'='*80}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    
    # Training metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Print results
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Mean Absolute Error: {mae:.6f} million tons")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    print(f"\nTraining Set Performance:")
    print(f"  MAE: {train_mae:.6f} million tons")
    print(f"  R² Score: {train_r2:.4f}")
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy (%)': round(accuracy, 2),
        'MAE (million tons)': round(mae, 6),
        'R² Score': round(r2, 4),
        'MAPE (%)': round(mape, 2),
        'Train MAE': round(train_mae, 6),
        'Train R²': round(train_r2, 4)
    })
    
    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
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

# Save results summary
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/model_comparison_results.csv', index=False)

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(results_df.to_string(index=False))

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name} with {best_accuracy:.2f}% Accuracy")
print(f"{'='*80}")

# Save best model
with open('outputs/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\nBest model saved to: outputs/best_model.pkl")

# Save metadata
metadata = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'best_model': best_model_name,
    'best_accuracy': best_accuracy,
    'features': list(X.columns),
    'target': 'xlpe_demand_Million_tons',
    'train_size': len(X_train),
    'test_size': len(X_test)
}

with open('outputs/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

# Create visualizations
print("\nGenerating visualizations...")

# 1. Model Comparison Plot
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(results_df))
plt.bar(x_pos, results_df['Accuracy (%)'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xticks(x_pos, results_df['Model'], rotation=45, ha='right')
plt.ylim(0, 100)
for i, v in enumerate(results_df['Accuracy (%)']):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=300)
plt.close()

# 2. Actual vs Predicted for Best Model
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_best, alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual XLPE Demand (Million Tons)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted XLPE Demand (Million Tons)', fontsize=12, fontweight='bold')
plt.title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/actual_vs_predicted.png', dpi=300)
plt.close()

# 3. Error Distribution
errors = y_test.values - y_pred_best
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error (Million Tons)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Error Distribution', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/error_distribution.png', dpi=300)
plt.close()

# 4. Feature Importance (for tree-based models)
if best_model_name in ["Decision Tree", "Random Forest"]:
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300)
    plt.close()
    
    feature_importance.to_csv('outputs/feature_importance.csv', index=False)

print("\nAll outputs saved to 'outputs' directory:")
print("  - model_comparison_results.csv")
print("  - Individual model predictions (CSV)")
print("  - best_model.pkl")
print("  - model_metadata.json")
print("  - Visualization plots (PNG)")

print("\n" + "="*80)
print("FORECASTING COMPLETE")
print("="*80)