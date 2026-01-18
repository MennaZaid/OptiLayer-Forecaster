# Infrastructure-Based Scenario Forecasting - Implementation Guide

## Overview

This guide explains the infrastructure-based scenario forecasting system integrated into the XLPE cable demand forecasting model. The system allows you to estimate total XLPE cable requirements by combining model-predicted market demand with planned infrastructure deployment.

---

## Methodology

### Cable Mass Calculation Formula

```
Total Cable Mass ≈ Infrastructure Area (km²) × Avg. Cable Density (km/km²) × Avg. Cable Weight (t/km)
```

**For 2026 Infrastructure Expansion:**
- Infrastructure Area: 8,779.9 km²
- Cable Density: ~100 km/km² (medium-voltage distribution)
- Cable Weight: ~1 ton/km (medium-voltage XLPE cables)
- **Result: 878,000 tons (0.878 million tons)**

*Source: ngoclancable.com - Industry standard for medium-voltage XLPE cables*

---

## Implementation Details

### 1. Infrastructure Proxy Features

Three new features were added to capture infrastructure-related demand patterns:

#### **a) Construction Output Index**
- **Calculation:** `100 + (GDP growth rate × 2.5)`
- **Purpose:** Captures construction activity that drives cable demand
- **Training Mean:** 109.69

#### **b) Urbanization Rate**
- **Calculation:** `(Electricity consumption / Max consumption) × 100`
- **Purpose:** Proxies urban expansion requiring grid infrastructure
- **Training Mean:** 54.09%

#### **c) Infrastructure Investment**
- **Calculation:** 3-month rolling average of XLPE demand
- **Purpose:** Captures momentum in infrastructure spending
- **Training Mean:** 0.266 million tons

### 2. Scenario Forecasting Function

The `predict_with_infrastructure_scenario()` function integrates infrastructure input with model predictions:

```python
scenario_results = predict_with_infrastructure_scenario(
    pipeline=best_pipeline,              # Trained ensemble model
    base_features=scenario_2026_features, # Economic indicators
    infrastructure_tons=878_000,          # Planned deployment
    feature_names=list(X.columns),       # Feature order
    scenario_name="2026 Infrastructure Expansion"
)
```

**Function Logic:**
1. Converts infrastructure tons to million tons (match dataset units)
2. Updates infrastructure proxy features based on planned deployment
   - Infrastructure investment = input × 0.8 (accounting for deployment phase-in)
   - Construction index = baseline × 1.15 (high activity period)
   - Urbanization rate = baseline × 1.10 (urban expansion)
3. Generates model prediction for market demand
4. Adds infrastructure contribution to get total demand

### 3. 2026 Baseline Assumptions

Economic indicators projected for 2026:

| Indicator | Value | Assumption |
|-----------|-------|------------|
| GDP Growth Rate | 3.5% | Middle East economic projection |
| Polyethylene Price | +2% from 2025 | Modest price increase |
| Electricity Consumption | +5% from 2025 | Annual growth trend |
| Infrastructure Area | 8,779.9 km² | Planned grid expansion |

---

## Results Summary

### 2026 Infrastructure Scenario Forecast

| Component | Value (Million Tons) | Value (Tons) | Percentage |
|-----------|---------------------|--------------|------------|
| **Base Market Demand** | 0.388848 | 388,848 | 30.7% |
| **Infrastructure Contribution** | 0.878000 | 878,000 | 69.3% |
| **TOTAL PROJECTED DEMAND** | **1.266848** | **1,266,848** | **100%** |

### Key Insights

1. **Infrastructure Dominance**: Infrastructure represents 69.3% of total 2026 demand
   - Indicates major grid expansion project
   - Requires careful supply chain planning

2. **Market Demand**: Model predicts 388,848 tons of baseline market demand
   - Based on GDP growth, electricity consumption trends
   - Excludes the infrastructure mega-project

3. **Total Demand Increase**: 1.27 million tons represents ~4.7× increase from current levels
   - Requires significant production capacity
   - Potential supply chain constraints to address

---

## Output Files Generated

### 1. **scenario_forecast_2026.csv**
Structured data with all scenario components:
```
Scenario, Base_Demand_Million_Tons, Infrastructure_Million_Tons, 
Total_Demand_Million_Tons, Total_Demand_Tons, Infrastructure_Percentage, Model_Used
```

### 2. **scenario_forecast_2026_report.txt**
Detailed text report with:
- Methodology explanation
- Forecast results breakdown
- Demand component percentages
- All assumptions documented

### 3. **scenario_forecast_2026.png**
Visual representation with:
- Pie chart: Demand breakdown (Market vs Infrastructure)
- Bar chart: Component values and total demand

---

## Using the System

### For Standard Forecasting (No Infrastructure)

```python
import pickle
import pandas as pd

# Load trained pipeline
pipeline = pickle.load(open('outputs/best_pipeline.pkl', 'rb'))

# Prepare feature data (12 features)
new_data = pd.DataFrame({
    'polyethylene_price': [8500],
    'Average Inflation percent index of Dollar': [3.0],
    'gdp_growth_rate': [3.5],
    'Total electricity consumption, Middle East': [1100000],
    'Monthy growth rate , Middle East %': [0.5],
    'construction_output_index': [108.75],
    'urbanization_rate': [58.5],
    'infrastructure_investment': [0.27],
    'lag_1': [0.325],
    'lag_3': [0.320],
    'lag_12': [0.315],
    'rolling_mean_3': [0.323]
})

# Get prediction
prediction = pipeline.predict(new_data)[0]
print(f"Predicted XLPE Demand: {prediction:.6f} million tons")
```

### For Infrastructure Scenario Forecasting

```python
# Define scenario features (from Models_Advanced.py)
scenario_features = {
    'polyethylene_price': 6426.51,
    'gdp_growth_rate': 3.5,
    # ... all 12 features
}

# Call scenario function (defined in Models_Advanced.py)
scenario_results = predict_with_infrastructure_scenario(
    pipeline=pipeline,
    base_features=scenario_features,
    infrastructure_tons=878_000,  # Your planned infrastructure
    feature_names=list(X.columns),
    scenario_name="Custom Scenario"
)

# Access results
print(f"Total Demand: {scenario_results['total_demand_tons']:,.0f} tons")
print(f"Infrastructure %: {scenario_results['infrastructure_percentage']:.1f}%")
```

---

## Customization Options

### Adjusting Infrastructure Input

For different infrastructure projects:

```python
# Example: Smaller project (400,000 tons)
infrastructure_tons = 400_000

# Example: Larger project (1.5 million tons)
infrastructure_tons = 1_500_000

# Recalculate scenario
scenario_results = predict_with_infrastructure_scenario(
    pipeline=best_pipeline,
    base_features=your_features,
    infrastructure_tons=infrastructure_tons,
    feature_names=feature_names,
    scenario_name="Alternative Scenario"
)
```

### Adjusting Proxy Feature Scaling

In the `predict_with_infrastructure_scenario()` function:

```python
# Current settings (conservative)
scenario_features['infrastructure_investment'] = infrastructure_million_tons * 0.8
scenario_features['construction_output_index'] *= 1.15
scenario_features['urbanization_rate'] *= 1.10

# More aggressive (faster deployment impact)
scenario_features['infrastructure_investment'] = infrastructure_million_tons * 1.0
scenario_features['construction_output_index'] *= 1.25
scenario_features['urbanization_rate'] *= 1.15
```

---

## Validation & Monitoring

### Key Metrics to Track

1. **Forecast Accuracy Validation**
   - Compare Q1-Q2 2026 actuals vs predictions
   - Recalibrate if MAE exceeds 5%

2. **Infrastructure Deployment Progress**
   - Track actual km² deployed vs planned
   - Adjust infrastructure_tons proportionally

3. **Economic Indicator Monitoring**
   - Update GDP growth if significantly different
   - Track polyethylene price fluctuations
   - Monitor electricity consumption trends

### Recommended Update Frequency

- **Monthly:** Update lag features (lag_1, lag_3) with latest actuals
- **Quarterly:** Reassess economic assumptions (GDP, prices)
- **Semi-Annually:** Retrain model with new historical data

---

## Integration with Ensemble Approach

The infrastructure scenario forecasting leverages the ensemble model's advantages:

### Why Ensemble for Infrastructure Scenarios?

1. **Lower Variance (70% reduction)**
   - More stable predictions across different infrastructure scales
   - Critical when infrastructure represents 69% of total demand

2. **Better Generalization (11% lower overfitting)**
   - Infrastructure scenarios represent "unseen" patterns
   - Ensemble's robustness handles extrapolation better

3. **Algorithm Diversity**
   - Random Forest (60%): Captures non-linear infrastructure impacts
   - Ridge Regression (40%): Models linear economic trends
   - Combined: Balanced view of infrastructure-driven demand

---

## Limitations & Assumptions

### Model Limitations

1. **Historical Data Scope**: Trained on 2001-2025 data (300 records)
   - Infrastructure deployments of this scale may be unprecedented
   - Model extrapolates beyond historical experience

2. **Proxy Feature Approximation**: Infrastructure proxies are synthetic
   - In production, use actual infrastructure planning data
   - Replace proxies with real metrics (cement consumption, CapEx)

3. **Linear Combination Assumption**: Infrastructure contribution added linearly
   - Assumes no interaction effects between market and infrastructure demand
   - May underestimate synergies (e.g., infrastructure driving market growth)

### Key Assumptions

1. **Cable Specifications**: Medium-voltage XLPE cables (100 km/km², 1 t/km)
   - Actual mix may include high-voltage (heavier) or low-voltage (lighter)
   - Adjust infrastructure_tons based on actual cable specifications

2. **Deployment Timeline**: Full deployment within 2026
   - If phased over multiple years, distribute infrastructure_tons accordingly

3. **Economic Projections**: GDP 3.5%, electricity +5%
   - Monitor actuals and update as deviations occur

---

## Technical Implementation Notes

### Feature Engineering Pipeline

```
Raw Data → Infrastructure Proxies → Lag Features → Time-Series Split → Model Training
```

**Feature Count:** 12 total features
- 5 economic indicators (polyethylene price, inflation, GDP, electricity, growth rate)
- 3 infrastructure proxies (construction index, urbanization, investment)
- 4 temporal features (lag_1, lag_3, lag_12, rolling_mean_3)

### Model Architecture

- **Type:** Voting Regressor (weighted ensemble)
- **Components:** Random Forest (60.2%) + Ridge Regression (39.8%)
- **Preprocessing:** StandardScaler (embedded in pipeline)
- **Cross-Validation:** TimeSeriesSplit (5 folds)
- **Performance:** 98.60% forecast accuracy on historical data

---

## Business Recommendations

### Supply Chain Implications

1. **Capacity Planning**
   - Current production handles ~270,000 tons/year
   - 2026 requires 1,266,848 tons (4.7× increase)
   - **Action:** Negotiate multi-year contracts with suppliers

2. **Inventory Strategy**
   - Infrastructure projects have fixed deadlines
   - **Action:** Build 6-month safety stock for infrastructure component

3. **Pricing Strategy**
   - Infrastructure represents 69% of demand (leverage for volume discounts)
   - **Action:** Separate pricing for infrastructure vs market demand

### Risk Mitigation

1. **Infrastructure Delay Risk**
   - Mega-projects often face delays
   - **Action:** Monthly infrastructure progress monitoring
   - **Contingency:** Flexible market demand absorption

2. **Raw Material Availability**
   - Polyethylene is key input
   - **Action:** Secure long-term polyethylene supply agreements
   - **Monitor:** Global polyethylene capacity and Middle East refinery output

3. **Demand Concentration Risk**
   - 69% of demand from single source
   - **Action:** Diversify customer base for market demand component
   - **Strategy:** Expand into adjacent markets (data centers, EV charging)

---

## Future Enhancements

### Recommended Improvements

1. **Real Infrastructure Data Integration**
   - Replace proxies with actual: cement consumption, construction permits, CapEx budgets
   - Source: Government infrastructure databases, World Bank projects

2. **Multi-Year Scenario Planning**
   - Extend to 2027-2030 with different infrastructure deployment phases
   - Model infrastructure ramp-up/ramp-down curves

3. **Sensitivity Analysis**
   - Test infrastructure range: 500K - 1.5M tons
   - Analyze break-even points for production capacity expansion

4. **Geographic Segmentation**
   - Break down by country/region within Middle East
   - Account for different infrastructure priorities

5. **Cable Type Differentiation**
   - Separate models for low/medium/high voltage
   - Different weight assumptions per cable type

---

## Contact & Support

For questions about the infrastructure forecasting implementation:

- **Model Performance**: Check `model_comparison_results.csv`
- **Scenario Details**: Review `scenario_forecast_2026_report.txt`
- **Visual Analysis**: Examine `scenario_forecast_2026.png`

**Model Version:** Enhanced with Infrastructure Proxies (January 2026)
**Ensemble Accuracy:** 98.60%
**Last Updated:** 2026-01-18
