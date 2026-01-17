import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


print("MODELS PREDICTION FOR THE NEXT MONTH (Risk-Aware + Hedging)")
print("___" * 30)
path = r"C:\Users\Omar\Downloads\Monthly_XLPE_Data - Sheet1 (1).csv"
df = pd.read_csv(path)
if 'Date' in df.columns: df = df.drop(columns=['Date'])
if 'Year' in df.columns: df = df.drop(columns=['Year'])
df_clean = df.dropna()
target_col = 'xlpe_demand_Million_tons'
price_col = 'polyethylene_price'
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]
split_point = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Collect predictions from all 100 trees to measure volatility
all_tree_preds = np.array([tree.predict(X_test.values) for tree in model.estimators_])
# Forecast (Mean)
y_pred = np.mean(all_tree_preds, axis=0)
# Risk (Standard Deviation)
uncertainty = np.std(all_tree_preds, axis=0)
# Dynamic Safety Stock (95% Confidence -> 1.96 Sigma)
safety_stock = 1.96 * uncertainty

# Logic: Buy 5% more if Price < 3-Month Average
current_prices = X_test[price_col].values
rolling_avg_price = X_test[price_col].rolling(window=3).mean().bfill().values

hedging_factors = []
for p, avg in zip(current_prices, rolling_avg_price):
    if p < avg * 0.95:
        hedging_factors.append(1.05) # PRICE DIP -> Buy Extra
    elif p > avg * 1.05:
        hedging_factors.append(0.95) # PRICE SPIKE -> Buy Less
    else:
        hedging_factors.append(1.0)  # Standard

hedging_factors = np.array(hedging_factors)

# Final Order Calculation
final_orders = (y_pred + safety_stock) * hedging_factors
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
accuracy = 100 - mape
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model: Risk-Aware Random Forest")
print(f"  > Accuracy: {accuracy:.2f}%")
print(f"  > Error (MAE): {mae:.4f} tons")
print(f"  > R2 Score: {r2:.4f}")
print("___" * 30)

results = pd.DataFrame({
    'Actual': y_test.values,
    'Forecast': y_pred,
    'Safety_Stock': safety_stock,
    'Price': current_prices,
    'Strategy_Factor': hedging_factors,
    'Final_Order': final_orders
})
print(results.tail())
