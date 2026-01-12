"""
================================================================================
INVENTORY OPTIMIZATION MODULE
================================================================================
ARABCAB Scientific Competition - Critical Missing Component

PURPOSE:
    Optimize XLPE inventory levels to minimize total costs while maintaining
    service levels. Uses demand forecasts from Model 1 & 2.

KEY OUTPUTS:
    - Economic Order Quantity (EOQ)
    - Reorder Point (ROP)
    - Safety Stock
    - Total Cost Analysis
    - Cost Savings vs Current Policy
================================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import joblib

# ============================================================================
# CONFIGURATION - Tunable Parameters
# ============================================================================
class InventoryConfig:
    """Industry-standard parameters for cable manufacturing"""
    
    # Cost parameters (USD)
    HOLDING_COST_PER_TON_PER_YEAR = 600  # Storage, insurance, capital cost
    ORDERING_COST_PER_ORDER = 2500  # Fixed cost: paperwork, shipping, inspection
    SHORTAGE_COST_PER_TON = 8000  # Lost production, rush orders, penalties
    
    # Operational parameters
    LEAD_TIME_DAYS = 21  # Average supplier lead time (3 weeks)
    WORKING_DAYS_PER_YEAR = 250
    SERVICE_LEVEL = 0.95  # 95% service level (industry standard)
    
    # Risk multipliers by urgency
    URGENCY_MULTIPLIERS = {
        'Critical': 1.5,      # Higher safety stock for critical cables
        'High': 1.3,
        'Medium': 1.0,
        'Low': 0.8
    }

# ============================================================================
# CORE INVENTORY OPTIMIZATION FUNCTIONS
# ============================================================================

def calculate_eoq(annual_demand, ordering_cost, holding_cost_per_unit):
    """
    Economic Order Quantity (Wilson Formula)
    
    EOQ = sqrt((2 * D * S) / H)
    where:
        D = Annual demand
        S = Ordering cost per order
        H = Holding cost per unit per year
    """
    if annual_demand <= 0:
        return 0
    
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
    return round(eoq, 2)


def calculate_safety_stock(avg_daily_demand, demand_std, lead_time_days, service_level):
    """
    Safety Stock = Z * σ_L * sqrt(L)
    
    where:
        Z = Z-score for service level (e.g., 1.65 for 95%)
        σ_L = Standard deviation of daily demand
        L = Lead time in days
    """
    if demand_std <= 0 or avg_daily_demand <= 0:
        return avg_daily_demand * lead_time_days * 0.2  # 20% buffer
    
    z_score = norm.ppf(service_level)
    safety_stock = z_score * demand_std * np.sqrt(lead_time_days)
    return round(safety_stock, 2)


def calculate_reorder_point(avg_daily_demand, lead_time_days, safety_stock):
    """
    Reorder Point = (Average Daily Demand × Lead Time) + Safety Stock
    """
    rop = (avg_daily_demand * lead_time_days) + safety_stock
    return round(rop, 2)


def calculate_total_cost(annual_demand, order_quantity, holding_cost, 
                        ordering_cost, safety_stock, shortage_probability=0.05):
    """
    Total Inventory Cost = Ordering Cost + Holding Cost + Shortage Cost
    
    TC = (D/Q) * S + (Q/2) * H + SS * H + Shortage_Cost
    """
    if order_quantity <= 0:
        return float('inf')
    
    # Number of orders per year
    num_orders = annual_demand / order_quantity
    
    # Costs
    ordering_cost_total = num_orders * ordering_cost
    holding_cost_total = (order_quantity / 2 + safety_stock) * holding_cost
    shortage_cost_total = shortage_probability * annual_demand * InventoryConfig.SHORTAGE_COST_PER_TON * 0.1
    
    total_cost = ordering_cost_total + holding_cost_total + shortage_cost_total
    
    return round(total_cost, 2)


# ============================================================================
# OPTIMIZATION FOR XLPE DEMAND
# ============================================================================

def optimize_xlpe_inventory(demand_forecast_df, config=InventoryConfig):
    """
    Main optimization function for XLPE inventory
    
    Input: DataFrame with columns ['year', 'demand_tons']
    Output: Optimization results with EOQ, ROP, safety stock, costs
    """
    
    print("\n" + "="*70)
    print("XLPE INVENTORY OPTIMIZATION")
    print("="*70)
    
    # Calculate demand statistics
    annual_demand = demand_forecast_df['demand_tons'].mean()
    demand_std = demand_forecast_df['demand_tons'].std()
    
    if demand_std == 0 or np.isnan(demand_std):
        demand_std = annual_demand * 0.15  # Assume 15% coefficient of variation
    
    daily_demand = annual_demand / config.WORKING_DAYS_PER_YEAR
    daily_std = demand_std / np.sqrt(config.WORKING_DAYS_PER_YEAR)
    
    print(f"\n[INPUT] Demand Statistics:")
    print(f"  • Annual Demand:        {annual_demand:,.1f} tons/year")
    print(f"  • Daily Demand (avg):   {daily_demand:.2f} tons/day")
    print(f"  • Demand Std Dev:       {demand_std:.2f} tons")
    print(f"  • Coefficient of Var:   {(demand_std/annual_demand)*100:.1f}%")
    
    # Calculate Economic Order Quantity
    eoq = calculate_eoq(
        annual_demand,
        config.ORDERING_COST_PER_ORDER,
        config.HOLDING_COST_PER_TON_PER_YEAR
    )
    
    # Calculate Safety Stock
    safety_stock = calculate_safety_stock(
        daily_demand,
        daily_std,
        config.LEAD_TIME_DAYS,
        config.SERVICE_LEVEL
    )
    
    # Calculate Reorder Point
    rop = calculate_reorder_point(
        daily_demand,
        config.LEAD_TIME_DAYS,
        safety_stock
    )
    
    # Calculate Total Costs
    total_cost = calculate_total_cost(
        annual_demand,
        eoq,
        config.HOLDING_COST_PER_TON_PER_YEAR,
        config.ORDERING_COST_PER_ORDER,
        safety_stock
    )
    
    # Calculate number of orders per year
    num_orders = annual_demand / eoq if eoq > 0 else 0
    order_frequency_days = config.WORKING_DAYS_PER_YEAR / num_orders if num_orders > 0 else 0
    
    # Maximum inventory level
    max_inventory = eoq + safety_stock
    
    # Average inventory
    avg_inventory = (eoq / 2) + safety_stock
    
    # Calculate cost breakdown
    ordering_cost_annual = num_orders * config.ORDERING_COST_PER_ORDER
    holding_cost_annual = avg_inventory * config.HOLDING_COST_PER_TON_PER_YEAR
    
    # Compare with naive policy (order when stock reaches 0)
    naive_order_qty = annual_demand / 12  # Monthly ordering
    naive_cost = calculate_total_cost(
        annual_demand,
        naive_order_qty,
        config.HOLDING_COST_PER_TON_PER_YEAR,
        config.ORDERING_COST_PER_ORDER,
        0  # No safety stock in naive approach
    ) + (annual_demand * 0.1 * config.SHORTAGE_COST_PER_TON)  # Assume 10% shortage
    
    cost_savings = naive_cost - total_cost
    cost_savings_percent = (cost_savings / naive_cost) * 100
    
    # Print results
    print(f"\n[OPTIMIZATION RESULTS]")
    print(f"  " + "="*50)
    print(f"  Economic Order Quantity (EOQ):  {eoq:>10,.0f} tons")
    print(f"  Safety Stock:                   {safety_stock:>10,.0f} tons")
    print(f"  Reorder Point (ROP):            {rop:>10,.0f} tons")
    print(f"  Maximum Inventory:              {max_inventory:>10,.0f} tons")
    print(f"  " + "="*50)
    
    print(f"\n[OPERATIONAL METRICS]")
    print(f"  • Orders per Year:      {num_orders:.1f}")
    print(f"  • Order Frequency:      Every {order_frequency_days:.0f} days")
    print(f"  • Service Level:        {config.SERVICE_LEVEL*100:.0f}%")
    print(f"  • Lead Time:            {config.LEAD_TIME_DAYS} days")
    
    print(f"\n[COST ANALYSIS]")
    print(f"  • Ordering Cost:        ${ordering_cost_annual:>10,.0f}/year")
    print(f"  • Holding Cost:         ${holding_cost_annual:>10,.0f}/year")
    print(f"  • Total Inventory Cost: ${total_cost:>10,.0f}/year")
    print(f"  • Naive Policy Cost:    ${naive_cost:>10,.0f}/year")
    print(f"  • Cost Savings:         ${cost_savings:>10,.0f}/year ({cost_savings_percent:.1f}%)")
    
    # Return results dictionary
    results = {
        'eoq': eoq,
        'safety_stock': safety_stock,
        'reorder_point': rop,
        'max_inventory': max_inventory,
        'avg_inventory': avg_inventory,
        'num_orders_per_year': num_orders,
        'order_frequency_days': order_frequency_days,
        'total_cost': total_cost,
        'ordering_cost': ordering_cost_annual,
        'holding_cost': holding_cost_annual,
        'cost_savings_vs_naive': cost_savings,
        'cost_savings_percent': cost_savings_percent,
        'annual_demand': annual_demand,
        'daily_demand': daily_demand,
        'service_level': config.SERVICE_LEVEL
    }
    
    return results


def optimize_by_risk_level(risk_demand_df, config=InventoryConfig):
    """
    Optimize inventory separately for each risk level
    Allocates different safety stock levels based on urgency
    """
    print("\n" + "="*70)
    print("RISK-STRATIFIED INVENTORY OPTIMIZATION")
    print("="*70)
    
    results_by_risk = {}
    total_cost = 0
    
    for _, row in risk_demand_df.iterrows():
        risk_level = row['risk_level']
        demand = row['total_xlpe_tons']
        
        # Create mini forecast
        forecast_df = pd.DataFrame({
            'year': [2026],
            'demand_tons': [demand]
        })
        
        # Adjust service level by risk
        adjusted_config = InventoryConfig()
        if risk_level == 'Critical':
            adjusted_config.SERVICE_LEVEL = 0.99  # 99% for critical
        elif risk_level == 'High':
            adjusted_config.SERVICE_LEVEL = 0.97
        else:
            adjusted_config.SERVICE_LEVEL = 0.95
        
        print(f"\n--- {risk_level} Risk Cables (Service Level: {adjusted_config.SERVICE_LEVEL*100:.0f}%) ---")
        
        result = optimize_xlpe_inventory(forecast_df, adjusted_config)
        results_by_risk[risk_level] = result
        total_cost += result['total_cost']
    
    print(f"\n[TOTAL] Combined Inventory Cost: ${total_cost:,.0f}/year")
    
    return results_by_risk, total_cost


# ============================================================================
# SAVE OPTIMIZATION RESULTS
# ============================================================================

def save_optimization_results(results, filename='outputs/inventory_optimization.csv'):
    """Save optimization results to CSV for dashboard"""
    import os
    
    os.makedirs('outputs', exist_ok=True)
    
    df = pd.DataFrame([results])
    df.to_csv(filename, index=False)
    print(f"\n[SAVED] {filename}")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution: Load forecast data and optimize inventory
    """
    print("\n" + "="*70)
    print("ARABCAB - INVENTORY OPTIMIZATION MODULE")
    print("="*70)
    
    try:
        # Try to load Model 2 forecast
        forecast_df = pd.read_csv('outputs/model2_adjusted_forecast.csv')
        # Use adjusted demand column (last column)
        if 'Adjusted_XLPE_Demand_Tons' in forecast_df.columns:
            forecast_df = forecast_df[['Year', 'Adjusted_XLPE_Demand_Tons']].rename(
                columns={'Year': 'year', 'Adjusted_XLPE_Demand_Tons': 'demand_tons'}
            )
        else:
            # Fallback: use last two columns
            forecast_df = forecast_df.iloc[:, [0, -1]]
            forecast_df.columns = ['year', 'demand_tons']
    except FileNotFoundError:
        print("[WARNING] Model 2 forecast not found. Using Model 1 aggregated data.")
        # Use aggregated demand from Model 1
        try:
            risk_df = pd.read_csv('data/model2_risk_demand.csv')
            total_demand = risk_df['total_xlpe_tons'].sum()
            forecast_df = pd.DataFrame({
                'year': range(2026, 2031),
                'demand_tons': [total_demand] * 5
            })
        except FileNotFoundError:
            print("[ERROR] No demand data found. Run arabcab.py first.")
            return None
    
    # Perform optimization
    results = optimize_xlpe_inventory(forecast_df)
    
    # Save results
    save_optimization_results(results)
    
    # Risk-stratified optimization (if risk data available)
    try:
        risk_df = pd.read_csv('data/model2_risk_demand.csv')
        risk_results, total_cost = optimize_by_risk_level(risk_df)
        
        # Save risk-stratified results
        risk_results_df = pd.DataFrame(risk_results).T
        risk_results_df.to_csv('outputs/inventory_by_risk.csv')
        print("[SAVED] outputs/inventory_by_risk.csv")
        
    except Exception as e:
        print(f"[SKIP] Risk-stratified optimization: {e}")
    
    print("\n" + "="*70)
    print("INVENTORY OPTIMIZATION COMPLETE")
    print("="*70)
    print("\n[NEXT STEPS]")
    print("  1. Review optimization results in outputs/")
    print("  2. Run Dashboard.py to visualize inventory recommendations")
    print("  3. Adjust InventoryConfig parameters based on business needs")
    
    return results


if __name__ == "__main__":
    main()
