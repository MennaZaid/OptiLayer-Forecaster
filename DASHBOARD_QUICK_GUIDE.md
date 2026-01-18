# Dashboard Quick Reference Guide

## Getting Started

### **Launch Dashboard**
```bash
cd "d:\ArabCab\OptiLayer-Forecaster"
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## Page Navigation

### **1. Overview** 📊
**What it shows:** High-level summary of forecasting system

**Key Metrics:**
- Best Model & Accuracy (98.60%)
- R² Score (0.9974)
- 2026 Total Demand (1.267M tons) - if scenario exists
- Service Level (95%) - if no scenario

**Use when:** You need a quick health check of the system

---

### **2. Model Performance** 📈
**What it shows:** Detailed comparison of all 7 models tested

**Key Features:**
- Accuracy bar chart (95-100% range for better visualization)
- Overfitting analysis (train vs test comparison)
- Ensemble composition pie chart (60% RF, 40% Ridge)
- Best model metrics (4 metrics: accuracy, R², MAE, RMSE, overfitting)

**Overfitting Interpretation:**
- < 0.1: Low (excellent generalization) - GREEN
- 0.1-0.2: Moderate (acceptable) - YELLOW
- > 0.2: High (may need attention) - RED

**Use when:** 
- Justifying model selection decisions
- Understanding why ensemble is best (lower overfitting, higher R²)
- Explaining model performance to stakeholders

---

### **3. Risk Analysis** ⚠️ (NEW)
**What it shows:** Uncertainty quantification for predictions

**How to use:**
1. Select model from dropdown (Linear Regression, Random Forest, Gradient Boosting, Ensemble)
2. View 4 risk metrics:
   - Avg Forecast: Mean prediction value
   - Avg Uncertainty (σ): Standard deviation of predictions
   - Avg Safety Stock: 1.96σ for 95% confidence
   - Price Adjustments: How often hedging is applied

**Visualization:**
- Black line: Actual demand (historical truth)
- Blue line: Predicted demand (model forecast)
- Light blue bands: 95% confidence intervals (±1.96σ)

**Interpreting Uncertainty:**
- Narrow bands: High confidence, stable predictions
- Wide bands: Lower confidence, more variability
- Ensemble has narrowest bands (70% variance reduction)

**Use when:**
- Setting safety stock levels
- Understanding prediction reliability
- Risk assessment for large purchases
- Comparing model confidence (Ensemble vs single models)

---

### **4. Infrastructure Scenario** 🏗️ (NEW)
**What it shows:** 2026 forecast with 878K ton infrastructure project

**Key Metrics:**
- Base Market Demand: 0.389M tons (from ensemble model)
- Infrastructure Contribution: 0.878M tons (planned deployment)
- Total 2026 Demand: 1.267M tons (69.3% infrastructure)

**Visualizations:**
1. **Pie Chart:** Shows 69.3% infrastructure / 30.7% market split
2. **Bar Chart:** Component breakdown (Base, Infrastructure, Total)

**Methodology Section:**
```
Total Cable = Area (km²) × Cable Density (km/km²) × Cable Weight (t/km)
Example: 8,779.9 km² × 100 km/km² × 1 t/km = 878,000 tons
```

**Business Implications:**
- Revenue opportunity: $1.9B @ $1,500/ton
- Scale-up required: 4.7× current capacity (270K → 1,267K tons/year)
- Lead time: 6-12 months for capacity expansion
- Multi-sourcing strategy recommended

**Download Report:**
- Click "Download Full Scenario Report" button
- Get detailed text report with all calculations

**Use when:**
- Planning for large infrastructure projects
- Capacity planning (12-18 months ahead)
- Budget forecasting for major contracts
- Supply chain readiness assessment

---

### **5. Inventory Optimization** 📦
**What it shows:** EOQ, safety stock, reorder point calculations

**Key Results:**
- Optimal Order Quantity: 62,100 tons
- Safety Stock: 16,600 tons (95% service level)
- Reorder Point: 21,600 tons
- Annual Savings: $18-73M

**Cost Breakdown:**
- Total Annual Cost: Holding + Ordering + Shortage
- Cost components shown in pie chart

**12-Month Forecast:**
- Table with monthly predictions
- Visual trend chart
- Cumulative demand tracking

**Use when:**
- Setting reorder points in inventory system
- Calculating optimal order sizes
- Justifying inventory investments
- Reducing holding costs while maintaining service level

---

### **6. Forecasting Tool** 🔮
**What it shows:** Interactive prediction generator

**Standard Forecasting (without infrastructure):**
1. Adjust economic indicators using sliders:
   - GDP Growth Rate (%)
   - Polyethylene Price Index
   - Electricity Consumption (TWh)
   - (and 12 other features)

2. Click "Generate Forecast"

3. View 3 metrics:
   - Predicted XLPE Demand
   - vs Historical Average (%)
   - Inventory Status (SAFE/WATCH/ORDER)

**Infrastructure Scenario Forecasting (NEW):**
1. Check "Include planned infrastructure deployment"

2. Choose input method:
   - **Direct (tons):** Enter infrastructure amount
     - Default: 878,000 tons (2026 project)
   - **Calculate from Area:** 
     - Infrastructure Area (km²): 8,779.9
     - Cable Density (km/km²): 100 (slider: 50-200)
     - Cable Weight (tons/km): 1.0 (slider: 0.5-2.0)
     - Auto-calculates total

3. Click "Generate Forecast"

4. View 4 metrics:
   - Base Market Demand
   - Infrastructure Contribution
   - Total Demand
   - Growth vs Avg

5. View bar chart showing breakdown

**Use when:**
- Testing "what-if" scenarios (e.g., "What if GDP grows by 5%?")
- Forecasting for infrastructure projects
- Quick predictions without running full model
- Demonstrating forecasting capability to stakeholders

---

### **7. About** ℹ️
**What it shows:** Complete project documentation and innovation showcase

**Sections:**
1. **Project Objectives:** Problem statement and goals
2. **Machine Learning Innovation:** Technical approach details
3. **Key Features:** What the system can do
4. **Performance Metrics:** All quantified results
5. **Technologies Used:** Tech stack and models evaluated
6. **Innovation Highlights:** 4 key innovations explained
7. **Documentation:** Links to all guides
8. **Competitive Advantages:** 7 differentiators listed

**Use when:**
- Onboarding new team members
- Writing competition reports
- Explaining system to non-technical stakeholders
- Understanding full project scope

---

## Common Workflows

### **Workflow 1: Generate Standard Forecast**
1. Go to **Forecasting Tool**
2. Keep default values or adjust economic indicators
3. Click "Generate Forecast"
4. Review prediction and inventory status
5. Compare to historical average

**Time:** 30 seconds

---

### **Workflow 2: Assess Infrastructure Project**
1. Go to **Infrastructure Scenario**
2. Review 2026 forecast (1.267M tons total)
3. Note infrastructure percentage (69.3%)
4. Review business implications ($1.9B opportunity)
5. Download full report for stakeholders

**Time:** 2 minutes

---

### **Workflow 3: Generate Infrastructure Forecast**
1. Go to **Forecasting Tool**
2. Adjust economic indicators for target year
3. Check "Include planned infrastructure deployment"
4. Choose calculation method:
   - **If you know total tons:** Select "Direct (tons)", enter value
   - **If you know area:** Select "Calculate from Area", enter km²
5. Adjust cable density (50-200 km/km²) based on:
   - Low voltage residential: 50-75 km/km²
   - Medium voltage distribution: 75-125 km/km²
   - High voltage grid: 125-200 km/km²
6. Adjust cable weight (0.5-2.0 t/km) based on cable type:
   - Light XLPE (low voltage): 0.5-0.8 t/km
   - Standard XLPE (medium voltage): 0.8-1.2 t/km
   - Heavy XLPE (high voltage): 1.2-2.0 t/km
7. Click "Generate Forecast"
8. Review 4-metric output and bar chart

**Time:** 3 minutes

---

### **Workflow 4: Model Justification for Report**
1. Go to **Overview** - Note ensemble accuracy (98.60%)
2. Go to **Model Performance** - View overfitting (0.0048, Low)
3. View ensemble composition (60/40 RF/Ridge)
4. Go to **Risk Analysis** - Compare Ensemble vs Random Forest uncertainty
5. Note 70% variance reduction
6. Go to **About** - Copy Innovation Highlights section
7. Compile justification: Higher accuracy + Lower overfitting + Lower variance

**Time:** 5 minutes

---

### **Workflow 5: Risk Assessment for Large Purchase**
1. Go to **Risk Analysis**
2. Select "Ensemble" model
3. Note average uncertainty (σ ≈ 0.00225 M tons = 2,250 tons)
4. Calculate 95% CI: ±1.96σ = ±4,410 tons
5. For 100,000 ton order: 95.6K - 104.4K tons range
6. Set safety stock at upper bound (104,410 tons)
7. Monitor price adjustments for hedging opportunities

**Time:** 3 minutes

---

### **Workflow 6: Inventory Policy Update**
1. Go to **Inventory Optimization**
2. Note current policies:
   - EOQ: 62,100 tons
   - Safety Stock: 16,600 tons
   - Reorder Point: 21,600 tons
3. Review 12-month forecast table
4. Go to **Risk Analysis** - Check uncertainty bands
5. If bands are narrow → Can reduce safety stock
6. If bands are wide → Keep or increase safety stock
7. Update inventory management system with new policies

**Time:** 5 minutes

---

## Tips & Tricks

### **Performance Tips**
- Dashboard loads all data once at startup (cached)
- Switching between pages is instant (no reloading)
- If data seems outdated, refresh browser (F5)

### **Interpretation Tips**
- **Green deltas:** Good news (lower cost, higher accuracy)
- **Red deltas:** Action needed or informational
- **Percentages in deltas:** Contextual information, not change

### **Visualization Tips**
- Hover over chart elements for exact values
- Click legend items to show/hide data series
- Drag to zoom, double-click to reset

### **Forecasting Tool Tips**
- Use sliders for quick adjustments
- Type exact values for precision
- Feature names are auto-formatted (underscores → spaces, title case)

### **Infrastructure Calculator Tips**
- Default values (8779.9 km², 100 km/km², 1 t/km) are for 2026 project
- Density 100 km/km² is typical for medium-voltage distribution
- Weight 1 t/km is standard for medium XLPE cables
- Calculate conservatively (round up) for safety

---

## Troubleshooting

### **Dashboard won't start**
```bash
# Check if port 8501 is in use
netstat -ano | findstr :8501

# Kill existing process (replace PID)
taskkill /PID <PID> /F

# Restart dashboard
streamlit run dashboard.py
```

### **Missing data warning**
If you see "No scenario forecast available" or "No risk analysis files found":

1. Run the main model file:
```bash
python Models_Advanced.py
```

2. Wait for completion (5-10 minutes)

3. Refresh dashboard (F5 in browser)

### **Incorrect values displayed**
1. Check if `Models_Advanced.py` was run recently
2. Verify outputs folder contains latest files
3. Check file timestamps in outputs folder
4. Re-run model if files are old

### **Slow performance**
- Close other browser tabs
- Restart dashboard
- Clear browser cache
- Check system resources (CPU, RAM)

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `R` | Re-run dashboard (refresh data) |
| `F5` | Refresh browser (reload page) |
| `Ctrl + C` (in terminal) | Stop dashboard |
| `Ctrl + Shift + R` | Hard refresh (clear cache) |

---

## File Dependencies

Dashboard requires these files in `outputs/` folder:

**Required (Dashboard won't load without these):**
- `model_comparison_results.csv` - Model performance data
- `inventory_optimization_results.json` - EOQ, safety stock, reorder point
- `inventory_forecast_12months.csv` - 12-month predictions
- `model_metadata.json` - Best model info, features, ensemble weights

**Optional (Dashboard works without these):**
- `scenario_forecast_2026.csv` - Infrastructure scenario (shows warning if missing)
- `Linear_Regression_risk_analysis.csv` - Risk metrics (shows warning if missing)
- `Random_Forest_risk_analysis.csv` - Risk metrics (shows warning if missing)
- `Gradient_Boosting_risk_analysis.csv` - Risk metrics (shows warning if missing)
- `Ensemble_risk_analysis.csv` - Risk metrics (shows warning if missing)
- `scenario_forecast_2026_report.txt` - Full scenario report (download button disabled if missing)
- `scenario_forecast_2026.png` - Scenario chart (not used in dashboard)

---

## Best Practices

### **Daily Use**
1. Start dashboard at beginning of day
2. Check Overview for system health
3. Use Forecasting Tool for quick predictions
4. Monitor Risk Analysis weekly

### **Weekly Review**
1. Review Model Performance (check for degradation)
2. Analyze Risk Analysis trends
3. Update inventory policies if needed
4. Generate infrastructure scenarios for upcoming projects

### **Monthly Planning**
1. Review 12-month forecast in Inventory Optimization
2. Assess infrastructure projects in pipeline
3. Use Forecasting Tool for next quarter scenarios
4. Update stakeholders with About page highlights

### **Quarterly Reports**
1. Compile metrics from Overview
2. Extract visualizations from Model Performance
3. Document risk assessment from Risk Analysis
4. Include infrastructure scenarios for planned projects
5. Use About page for executive summary

---

## Support & Contact

**Documentation:**
- `DASHBOARD_ENHANCEMENTS.md` - Complete changelog
- `ENHANCED_SOLUTION_GUIDE.md` - Technical implementation
- `INFRASTRUCTURE_SCENARIO_FORECASTING_GUIDE.md` - Infrastructure methodology
- `FORECASTING_IMPROVEMENTS.md` - Model enhancements explained

**Competition:** ARABCAB Scientific Competition  
**Industry:** Cable & Metals Manufacturing  
**Geography:** Egypt • Bahrain • UAE

---

**Dashboard Version:** 2.0  
**Last Updated:** 2024  
**Status:** ✅ Production Ready
