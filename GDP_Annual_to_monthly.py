import pandas as pd
import numpy as np

# 1. Your historical annual data
years = list(range(2000, 2025))
gdp_values = [6.365211221, 11.962754725, 2.086215238, 4.699606884, 7.940703352, 
              5.288409258, 5.988150195, 5.233011329, 4.376540555, 1.118258303, 
              4.809665191, 3.842264154, 4.248199272, 2.417173305, 3.122064294,
              2.770165524, 4.498466766, 2.076518433, 2.399952291, 1.591371607, 
              -3.58499749, 5.036313617, 6.446528922, 2.259207061, 2.112324854]

df = pd.DataFrame({'year': years, 'gdp_growth': gdp_values})
df['date'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
df = df.set_index('date')

# 2. Resample and Interpolate (Cubic Spline)
monthly_gdp = df.resample('MS').interpolate(method='cubic')

# 3. Filter to stop exactly at December 2024
monthly_gdp = monthly_gdp[:'2024-12-01'].copy()

# 4. Create separate columns for Year and Month
monthly_gdp['Year'] = monthly_gdp.index.year
monthly_gdp['Month'] = monthly_gdp.index.month

# 5. Reorder columns: Year, Month, then the GDP Growth rate
output = monthly_gdp[['Year', 'Month', 'gdp_growth']]

# Save to CSV
output.to_csv('data/historical_gdp_final.csv', index=False)

# Preview
print("First 5 rows:")
print(output.head())
print("\nLast 5 rows:")
print(output.tail())