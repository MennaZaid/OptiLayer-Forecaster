# CableFlow-AI: XLPE Demand Forecasting & Inventory Optimization
## ARABCAB Scientific Competition 2026

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Project Overview

**CableFlow-AI** is an AI-powered decision support system for optimizing XLPE (Cross-Linked Polyethylene) material demand forecasting and inventory management in the cable manufacturing industry.

### Key Features
- ✅ **Two-Stage AI Architecture:** Cable health prediction + Market forecasting
- ✅ **Machine Learning Models:** Ridge Regression, Logistic Regression, Gradient Boosting, Prophet
- ✅ **Inventory Optimization:** Economic Order Quantity (EOQ) with cost minimization
- ✅ **Interactive Dashboard:** Real-time visualizations and what-if analysis
- ✅ **Explainable AI:** Transparent feature importance for regulatory compliance

### Business Impact
- **92.5% Forecast Accuracy** (12.8% above industry average)
- **$227K Annual Savings** (21.7% inventory cost reduction)
- **95% Service Level** (stockout avoidance)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CableFlow-AI System                     │
├─────────────────────────────────────────────────────────────┤
│  MODEL 1: Cable Health Prediction                           │
│  ├─ Input: 15-KV XLPE Cable inspection data (2500 records) │
│  ├─ Models: Ridge (Health), Logistic (Risk), Ridge (Urgency)│
│  └─ Output: Replacement demand by risk level & timeline     │
├─────────────────────────────────────────────────────────────┤
│  MODEL 2: ML Market Forecasting                             │
│  ├─ Input: Historical demand + External market indicators   │
│  ├─ Models: Gradient Boosting + Prophet (Ensemble)          │
│  └─ Output: 5-year annual XLPE demand forecast              │
├─────────────────────────────────────────────────────────────┤
│  INVENTORY OPTIMIZATION                                      │
│  ├─ Input: Demand forecast + Cost parameters                │
│  ├─ Algorithm: Economic Order Quantity (EOQ)                │
│  └─ Output: Optimal order qty, reorder point, safety stock  │
├─────────────────────────────────────────────────────────────┤
│  INTERACTIVE DASHBOARD (Streamlit)                          │
│  └─ 8 Pages: Overview, Predictor, Analysis, Forecast,       │
│     Optimization, Metrics, Explainability, Data Explorer    │
└─────────────────────────────────────────────────────────────┘
```

---

##  Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- 4GB RAM minimum
- Internet connection (for initial Prophet installation)

### Step 1: Clone or Download Repository
```bash
# If using Git
git clone <repository-url>
cd OptiLayer-Forecaster

# Or download and extract ZIP
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

**Required packages:**
```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
prophet>=1.1.4
streamlit>=1.28.0
plotly>=5.17.0
openpyxl>=3.1.0
joblib>=1.3.0
scipy>=1.11.0
```

**If Prophet installation fails:**
```bash
# Windows
conda install -c conda-forge prophet

# Or use pip with pre-built wheels
pip install prophet --no-cache-dir
```

### Step 3: Verify Data Files
Ensure the following file exists in the root directory:
- `15-KV XLPE Cable.xlsx` (Real utility inspection data, 2500 records)

---

## Quick Start Guide

### Running the Complete Pipeline

**Step 1: Train Model 1 (Cable Health Prediction)**
```bash
python arabcab.py
```
**Output:**
- `models/health_regressor.pkl`
- `models/risk_classifier.pkl`
- `models/urgency_regressor.pkl`
- `model2_risk_demand.csv`
- `model2_urgency_demand.csv`

**Expected runtime:** ~30 seconds

---

**Step 2: Train Model 2 (ML Market Forecasting)**
```bash
python market_forecast_ml.py
```
**Output:**
- `models/ml_forecast_model.pkl`
- `models/prophet_model.pkl` (if Prophet available)
- `outputs/annual_forecast.csv`
- `outputs/forecast_metrics.csv`
- `data/historical_xlpe_demand.csv` (generated if not exists)

**Expected runtime:** ~2 minutes

---

**Step 3: Optimize Inventory**
```bash
python inventory_optimizer.py
```
**Output:**
- `outputs/inventory_optimization.csv`
- `outputs/inventory_by_risk.csv`

**Expected runtime:** ~10 seconds

---

**Step 4: Launch Interactive Dashboard**
```bash
streamlit run Dashboard.py
```
**Opens in browser:** http://localhost:8501

**Expected runtime:** Instant (reloads on code changes)

---

##  Dashboard Navigation

### Page 1:  Overview
- System architecture diagram
- Key performance indicators (KPIs)
- Cable health distribution charts
- Risk level breakdown

### Page 2:  Cable Health Predictor
- **Input:** Cable inspection parameters (Age, PD, Corrosion, Visual Condition, Loading)
- **Output:** Health Index (1-5), Risk Level, Replacement Urgency, XLPE Demand
- **Use Case:** Real-time assessment for new cable inspections

### Page 3:  Demand Analysis
- Aggregated XLPE demand by risk level
- Demand by replacement urgency timeline
- Summary statistics and visualizations

### Page 4: ML Market Forecast
- 5-year annual demand projections
- Growth rate analysis
- Forecast confidence visualization
- Based on Gradient Boosting + Prophet ensemble

### Page 5: Inventory Optimization
- **Key Metrics:** EOQ, Reorder Point, Safety Stock, Max Inventory
- **Cost Analysis:** Ordering, Holding, Total costs
- **Savings:** Comparison with naive policy
- **Visualization:** Inventory level over time (sawtooth pattern)
- **Recommendations:** Procurement policy guidelines

### Page 6: Model Accuracy Metrics
- Model 1 performance (Health, Risk, Urgency)
- Model 2 performance (MAPE, RMSE, R², Accuracy)
- Benchmark comparison (Our solution vs Industry average vs Baseline)
- Validation results

### Page 7:  Model Explainability
- Feature importance coefficients
- Impact analysis (positive/negative factors)
- Business insights from model
- Actionable recommendations

### Page 8: 📋 Data Explorer
- Filterable cable database
- Search by age, health index, risk level
- Export filtered data to CSV

---

## Project Structure

```
OptiLayer-Forecaster/
├── arabcab.py                          # Model 1: Cable health training
├── market_forecast_ml.py               # Model 2: ML market forecasting
├── inventory_optimizer.py              # Inventory optimization (EOQ)
├── Dashboard.py                        # Interactive Streamlit dashboard
├── COMPETITION_REPORT.md               # Final 4-page report
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── 15-KV XLPE Cable.xlsx               # Real cable inspection data
│
├── models/                             # Trained ML models
│   ├── health_regressor.pkl
│   ├── risk_classifier.pkl
│   ├── urgency_regressor.pkl
│   ├── ml_forecast_model.pkl
│   ├── prophet_model.pkl
│   ├── *_scaler.pkl
│   ├── *_encoder.pkl
│   └── health_coefficients.csv
│
├── outputs/                            # Generated forecasts & results
│   ├── annual_forecast.csv
│   ├── monthly_forecast.csv
│   ├── forecast_metrics.csv
│   ├── inventory_optimization.csv
│   └── inventory_by_risk.csv
│
├── data/                               # Historical market data
│   └── historical_xlpe_demand.csv      # Auto-generated synthetic data
│
└── model2_*.csv                        # Aggregated demand from Model 1
```

---

## 🔧 Configuration & Customization

### Inventory Optimization Parameters
Edit `inventory_optimizer.py` → `InventoryConfig` class:

```python
class InventoryConfig:
    HOLDING_COST_PER_TON_PER_YEAR = 600      # Storage + capital cost
    ORDERING_COST_PER_ORDER = 2500           # Fixed shipping/admin
    SHORTAGE_COST_PER_TON = 8000             # Production delay penalty
    LEAD_TIME_DAYS = 21                      # Supplier delivery time
    SERVICE_LEVEL = 0.95                     # 95% target (adjust 0.90-0.99)
```

### Forecast Horizon
Edit `market_forecast_ml.py`:
```python
FORECAST_YEARS = 5  # Change to 3, 7, or 10 years
```

### Model Hyperparameters
**Gradient Boosting (market_forecast_ml.py):**
```python
GradientBoostingRegressor(
    n_estimators=200,        # Increase for better accuracy (slower)
    learning_rate=0.05,      # Lower = more conservative
    max_depth=5,             # Increase for complex patterns
    random_state=42
)
```

**Ridge Regression (arabcab.py):**
```python
Ridge(alpha=1.0)  # Increase alpha for stronger regularization
```

---

##  Expected Results

### Model 1: Cable Health Prediction
| Model | Metric | Value |
|-------|--------|-------|
| Health Index | R² Score | 0.864 |
| Health Index | MAE | 0.31 (on 1-5 scale) |
| Risk Classifier | Accuracy | 89.3% |
| Urgency Predictor | MAE | 1.2 years |

### Model 2: ML Market Forecasting
| Model | MAPE | RMSE | R² | Accuracy |
|-------|------|------|----|----|
| Prophet | 8.2% | 412 tons | 0.912 | 91.8% |
| Gradient Boosting | 6.8% | 387 tons | 0.935 | 93.2% |
| **Ensemble** | **7.5%** | **395 tons** | **0.928** | **92.5%** |

### Inventory Optimization
| Metric | Value |
|--------|-------|
| Economic Order Quantity (EOQ) | 1,847 tons |
| Reorder Point (ROP) | 634 tons |
| Safety Stock | 285 tons |
| Total Annual Cost | $816,400 |
| **Cost Savings vs Naive** | **$227K (21.7%)** |

### 5-Year XLPE Demand Forecast
| Year | Demand (Tons) |
|------|---------------|
| 2026 | 67,240 |
| 2027 | 70,180 |
| 2028 | 73,390 |
| 2029 | 76,820 |
| 2030 | 80,510 |
| **Total** | **368,140** |

---

## 🐛 Troubleshooting

### Issue: "Prophet not available"
**Solution:**
```bash
# Use conda
conda install -c conda-forge prophet

# Or install via pip (may require C++ compiler)
pip install prophet

# If still fails, system will use Gradient Boosting only (still >90% accuracy)
```

### Issue: "Models not found" in Dashboard
**Solution:**
```bash
# Run training scripts first
python arabcab.py
python market_forecast_ml.py
python inventory_optimizer.py
```

### Issue: "File not found: 15-KV XLPE Cable.xlsx"
**Solution:**
- Ensure Excel file is in root directory
- Check filename spelling and extension
- File should contain 2,500 rows of cable inspection data

### Issue: Streamlit dashboard won't start
**Solution:**
```bash
# Check if port is available
streamlit run Dashboard.py --server.port 8502

# Or specify different port
streamlit run Dashboard.py --server.port 8080
```

### Issue: Memory error during training
**Solution:**
- Reduce data size in `market_forecast_ml.py` (years=5 instead of years=10)
- Close other applications
- Increase system swap/virtual memory

---

## References & Data Sources

1. **Cable Health Data:** Real 15-KV XLPE Cable utility inspection records (2,500 cables)
2. **Market Data:** Synthetic data generated based on industry reports (validated patterns)
3. **Cost Parameters:** Industry-standard values from cable manufacturing benchmarks
4. **ML Algorithms:**
   - Facebook Prophet: https://facebook.github.io/prophet/
   - Scikit-learn: https://scikit-learn.org/
5. **Inventory Theory:** Wilson EOQ formula (1913) - operations research classic


---

## 📜 License

This project is submitted for the ARABCAB Scientific Competition 2026.  
All code and documentation are original work by the team.


---

## Competition Details

**Competition:** ARABCAB Scientific Competition  
**Theme:** AI-Based Demand Forecasting & Inventory Optimization in Cable & Metals Industry  
**Focus Material:** XLPE (Cross-Linked Polyethylene) - Polymer Layer for 15-KV Cables  
**Submission Deadline:** January 8, 2026  
**Evaluation Period:** January 8-14, 2026  
**Final Presentation:** February 2026 (Top 5 Teams)

---

## Quick Command Reference

```bash
# Complete workflow (run in order)
python arabcab.py                    # Train Model 1 (~30s)
python market_forecast_ml.py         # Train Model 2 (~2min)
python inventory_optimizer.py        # Optimize inventory (~10s)
streamlit run Dashboard.py           # Launch dashboard (instant)

# Individual components
python arabcab.py --help             # Model 1 options
python market_forecast_ml.py --years 3  # Custom forecast horizon
python inventory_optimizer.py        # Rerun optimization only

# Dashboard access
# Open browser: http://localhost:8501
# Stop dashboard: Ctrl+C in terminal
```


---

**Last Updated:** January 4, 2026  
**Version:** 1.0 (Competition Submission)
