import pandas as pd
import numpy as np

# 1. Historical Annual Data (2000 - Jan 2024)
years = list(range(2000, 2025))
gdp_values = [6.365211221, 11.962754725, 2.086215238, 4.699606884, 7.940703352, 
              5.288409258, 5.988150195, 5.233011329, 4.376540555, 1.118258303, 
              4.809665191, 3.842264154, 4.248199272, 2.417173305, 3.122064294,
              2.770165524, 4.498466766, 2.076518433, 2.399952291, 1.591371607, 
              -3.58499749, 5.036313617, 6.446528922, 2.259207061, 2.112324854]

# Calculate historical volatility (year-over-year changes)
historical_changes = np.diff(gdp_values)
mean_change = np.mean(historical_changes)
std_change = np.std(historical_changes)

# 2. Resample 2000-2023 to Monthly using Cubic Interpolation
df_hist = pd.DataFrame({'gdp_growth': gdp_values}, 
                       index=pd.date_range(start='2000-01-01', periods=len(gdp_values), freq='YS'))
monthly_hist = df_hist.resample('MS').interpolate(method='cubic')

# 3. Bootstrap Simulation for Feb 2024 - Dec 2025 (23 months)
num_months = 23
num_sims = 1000
last_real_val = gdp_values[-1]

simulations = np.zeros((num_sims, num_months))

for i in range(num_sims):
    current_val = last_real_val
    for m in range(num_months):
        # Sample a random historical change and scale to monthly
        shock = np.random.normal(mean_change/12, std_change/12)
        current_val += shock
        simulations[i, m] = current_val

# Calculate the mean path from all simulations
mean_forecast_path = np.mean(simulations, axis=0)

# 4. Combine Historical and Bootstrap Forecast
forecast_dates = pd.date_range(start='2024-02-01', periods=num_months, freq='MS')
df_forecast = pd.DataFrame({'gdp_growth': mean_forecast_path}, index=forecast_dates)

# Merge everything
final_df = pd.concat([monthly_hist, df_forecast])

# Format for Final CSV
final_df['Year'] = final_df.index.year
final_df['Month'] = final_df.index.month
final_df = final_df[['Year', 'Month', 'gdp_growth']]

# 5. SAVE AS CSV AND CALL FINAL
final_df.to_csv('data/FINAL_GDP_BOOTSTRAP.csv', index=False)

print("File saved successfully as FINAL_GDP_BOOTSTRAP.csv")
print(final_df.tail(15)) # Previewing the Bootstrap months