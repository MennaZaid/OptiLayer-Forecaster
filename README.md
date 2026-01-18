
```markdown
# OptiLayer-Forecaster: XLPE Demand Forecasting & Inventory Optimization

An advanced machine learning system for predicting XLPE cable demand and optimizing inventory levels using ensemble learning with risk-aware forecasting and hedging strategies.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Advanced Forecasting Models
```bash
python Models_Advanced.py
```

This will:
- Train and evaluate 5 advanced ML models (Linear Regression, Ridge, Random Forest, Gradient Boosting)
- Create optimized ensemble model with exponential weighting
- Perform risk-aware forecasting with hedging strategies
- Generate scenario-based 2026 infrastructure projections
- Save all predictions, risk analysis, and visualizations

### 3. Run Inventory Optimization
```bash
python inventory_optimization.py
```

This will:
- Load AI forecasts from Models_Advanced.py output
- Calculate safety stock (95% service level)
- Compute Economic Order Quantity (EOQ)
- Perform 12-month inventory simulation
- Generate inventory analysis and visualizations

### 4. Launch Interactive Dashboard
```bash
streamlit run Dashboard.py
```

This will open your browser to an interactive dashboard with 7 pages of analysis and forecasting tools.

---

## 📁 Project Structure

```
d:\ArabCab\OptiLayer-Forecaster\
│
├── Models_Advanced.py                      # Advanced forecasting models
├── inventory_optimization.py               # Inventory optimization engine
├── Dashboard.py                            # Streamlit interactive dashboard
├── historical_xlpe_demand.xlsx             # Historical demand data (input)
├── requirements.txt                        # Python dependencies
│
└── outputs/                                # Auto-generated analysis results
    ├── model_comparison_results.csv        # All model performance metrics
    ├── best_pipeline.pkl                   # Production-ready trained pipeline
    ├── model_metadata.json                 # Model configuration & metadata
    │
    ├── [Model]_predictions.csv             # Predictions for each model
    ├── [Model]_Risk_Analysis.csv           # Risk-aware forecasting results
    ├── [Model]_Risk_Adjusted.csv           # Price-adjusted forecasts
    │
    ├── Ensemble_Risk_Analysis.csv          # Final ensemble risk analysis
    ├── inventory_forecast_12months.csv     # 12-month inventory simulation
    ├── inventory_optimization_results.json # Inventory parameters (EOQ, ROP, SS)
    │
    ├── scenario_forecast_2026.csv          # 2026 infrastructure scenario
    ├── scenario_forecast_2026_report.txt   # Detailed scenario analysis
    │
    ├── comprehensive_model_comparison.png  # Model performance charts
    ├── best_model_analysis.png             # Actual vs predicted plots
    └── feature_importance.png              # Feature importance visualization
```

---

## 📊 What Each Module Does

### Models_Advanced.py
**Advanced ML Forecasting System with 5 Models**

**Key Features:**
- ⏱️ **Time-Series Architecture:** Lag-based features (lag_1, lag_3, lag_12 seasonal), rolling mean windows
- 🔧 **Scikit-learn Pipelines:** StandardScaler → Model (prevents data leakage)
- 🎯 **Time-Series Cross-Validation:** Preserves temporal order (no random shuffling)
- 🤖 **5 Advanced Models:**
  - Linear Regression (Enhanced) with weighted lags & asymmetric loss
  - Ridge Regression with hyperparameter tuning
  - Random Forest (Risk-Aware) with multi-tree uncertainty
  - Gradient Boosting (Risk-Aware) with staged predictions
  - Optimized Ensemble using selective models + exponential weighting

**Innovations:**
- **Risk-Aware Forecasting:** Calculates uncertainty (σ) and safety stock from ensemble variance
- **Hedging Strategy:** Adjusts orders based on polyethylene price trends (±5% thresholds)
- **Infrastructure Proxies:** Incorporates construction output, urbanization rate, investment trends
- **Scenario Forecasting:** Projects 2026 demand with planned infrastructure inputs
- **Weighted Ensemble:** Uses cubed accuracy scores for extreme model differentiation

**Output Files:**
- `model_comparison_results.csv` - Performance metrics for all 5 models + ensemble
- `best_pipeline.pkl` - Production-ready pipeline (includes scaler + trained model)
- `[Model]_predictions.csv` - Predictions with errors for each model
- `[Model]_Risk_Analysis.csv` - Forecast + uncertainty + safety stock + hedging factors
- `Ensemble_Risk_Analysis.csv` - Final ensemble forecast with all risk metrics
- `scenario_forecast_2026.csv` - 2026 infrastructure scenario results

### inventory_optimization.py
**Inventory Optimizer Using AI Forecasts**

**Key Features:**
- 🧠 **Uses YOUR Forecasts:** Loads ensemble predictions from Models_Advanced.py
- 📦 **Safety Stock Calculation:** Based on service level (95% default) using AI uncertainty
- 📊 **Economic Order Quantity (EOQ):** Optimizes order size vs. holding costs
- 📈 **Reorder Point (ROP):** Prevents stockouts during lead time
- 💰 **Total Cost Analysis:** Holding, ordering, and potential stockout costs
- 🔄 **12-Month Simulation:** Projects inventory levels with periodic ordering

**Inventory Parameters Calculated:**
- Safety Stock: Volume needed to maintain service level
- Reorder Point: When to trigger next order
- EOQ: Optimal order quantity balancing costs
- Service Level: Achieved stockout prevention rate

**Output Files:**
- `inventory_optimization_results.json` - All calculated parameters
- `inventory_forecast_12months.csv` - Monthly inventory levels and orders
- Inventory simulation charts and cost breakdowns

### Dashboard.py
**Interactive Streamlit Web Application**

**7 Interactive Pages:**

1. **Overview** - Executive summary with key KPIs
2. **Model Performance** - Detailed model comparison, accuracy metrics, ensemble composition
3. **Risk Analysis** - Forecast uncertainty, safety stock, hedging strategy results
4. **Infrastructure Scenario** - 2026 scenario forecasts with infrastructure impact analysis
5. **Inventory Optimization** - Inventory parameters, 12-month simulation, cost analysis
6. **Forecasting Tool** - Interactive prediction: input custom values, get predictions
7. **About** - Project documentation and methodology

**Features:**
- Real-time data loading and caching
- Interactive Plotly charts (zoom, pan, hover data)
- Custom CSS styling with professional design
- Responsive layout for mobile/desktop

---

## 🎯 Key Results & Metrics

### Model Performance
- **Best Model:** Ensemble (Top 3) with weighted predictions
- **Forecast Accuracy:** 97%+ (measured as 1 - MAPE)
- **R² Score:** 0.95+ (explains 95%+ of variance)
- **MAE:** <0.000054 million tons (< 54 tons error)

### Risk Analysis
- **Average Uncertainty (σ):** Calculated from ensemble variance
- **95% Safety Stock:** 1.96σ dynamic buffer
- **Hedging Effectiveness:** Price-based adjustments ±5%

### Inventory Optimization
- **Service Level:** 95%+ (stockout rate < 5%)
- **Total Annual Cost:** Holding + ordering + potential stockout
- **EOQ:** Optimal order quantity balancing costs
- **12-Month Simulation:** Projects realistic inventory levels

---

## 🔧 Customization

### Adjust Inventory Parameters (inventory_optimization.py)
```python
# Change these in the InventoryOptimizer class:
holding_cost_per_ton = 50          # Storage/carrying cost
ordering_cost = 5000               # Cost per order placed
stockout_cost_per_ton = 500        # Lost sales cost
material_cost_per_ton = 2000       # Raw material cost
```

### Change Service Level
```python
optimizer.calculate_safety_stock(service_level=0.95)  # 0.90-0.99 range
```

### Adjust Forecasting Ensemble
In Models_Advanced.py, modify:
```python
accuracy_threshold = 98.0  # Minimum accuracy for ensemble inclusion
boost_factor = 1.5         # How much to boost best model
```

---

## 📈 Workflow Overview

```
1. Load Historical Data
   historical_xlpe_demand.xlsx
           ↓
2. Models_Advanced.py
   • Extract lag features (lag_1, lag_3, lag_12)
   • Train 5 advanced ML models
   • Calculate uncertainty & safety stock
   • Create optimized ensemble
   • Generate 2026 scenario forecast
           ↓
   outputs/
   ├── best_pipeline.pkl
   ├── Ensemble_Risk_Analysis.csv
   ├── scenario_forecast_2026.csv
   └── [Model]_Risk_Analysis.csv
           ↓
3. inventory_optimization.py
   • Load ensemble forecasts
   • Calculate safety stock & EOQ
   • Simulate 12-month inventory
           ↓
   outputs/
   ├── inventory_optimization_results.json
   └── inventory_forecast_12months.csv
           ↓
4. Dashboard.py
   • Load all outputs
   • Display interactive analysis
   • Provide forecasting tool
           ↓
   Browser: http://localhost:8501
```

---

## 🚨 Troubleshooting

### Models_Advanced.py fails
- ✅ Check historical_xlpe_demand.xlsx is in the same folder
- ✅ Verify all required columns: `xlpe_demand_Million_tons`, `Date`, etc.
- ✅ Ensure Python 3.8+ with required packages: `pip install -r requirements.txt`

### inventory_optimization.py fails
- ✅ Run Models_Advanced.py first (generates required forecast files)
- ✅ Check Ensemble_Risk_Analysis.csv exists
- ✅ Verify historical_xlpe_demand.xlsx is accessible

### Dashboard.py fails
- ✅ Run both Models_Advanced.py and inventory_optimization.py first
- ✅ Check all files in outputs folder exist
- ✅ Install Streamlit: `pip install streamlit`
- ✅ If browser doesn't open, manually go to `http://localhost:8501`

### Missing data columns
- Some features are optional (polyethylene_price, gdp_growth_rate, etc.)
- Models adapt to available features automatically
- Risk-aware forecasting works with or without price data

---

## 📋 Dependencies

See requirements.txt for complete list:
- **scikit-learn** - Machine learning models & pipelines
- **pandas** - Data manipulation & analysis
- **numpy** - Numerical computing
- **streamlit** - Interactive web dashboard
- **plotly** - Interactive visualizations
- **matplotlib & seaborn** - Static plotting
- **openpyxl** - Excel file reading
- **scipy** - Statistical functions (safety stock calculation)

---

## 🏆 Innovation Highlights

✨ **Scikit-learn Pipelines:** Production-ready preprocessing (prevents data leakage)  
✨ **Time-Series Architecture:** Lag-based features + temporal cross-validation  
✨ **Risk-Aware Forecasting:** Uncertainty quantification from ensemble variance  
✨ **Hedging Strategies:** Price-based order adjustments  
✨ **Optimized Ensemble:** Selective models with exponential weighting  
✨ **Infrastructure Scenarios:** 2026 projections with planned infrastructure  
✨ **Interactive Dashboard:** 7-page Streamlit app with real-time analysis  

---

## Execution Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run Models_Advanced.py (generates AI forecasts & risk analysis)
- [ ] Run inventory_optimization.py (generates inventory parameters)
- [ ] Launch Dashboard.py (opens interactive web interface)
- [ ] Verify outputs folder contains all expected files
- [ ] Test dashboard pages: Overview → Model Performance → Risk Analysis → Inventory → Forecasting Tool
- [ ] Customize parameters as needed for your business case
- [ ] Export reports from dashboard or CSV files for stakeholders

---

**Created:** January 2026  
**Status:** Production Ready  
*Advanced forecasting with integrated inventory optimization and risk management*
```
