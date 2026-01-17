import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os


print("="*80)
print("INVENTORY OPTIMIZATION MODULE")
print("="*80)

# Load historical data
df = pd.read_excel('historical_xlpe_demand.xlsx')
df_clean = df.drop(columns=['Date', 'Year', 'Month']).dropna()

# Calculate demand statistics
xlpe_demand = df_clean['xlpe_demand_Million_tons']
avg_demand = xlpe_demand.mean()
std_demand = xlpe_demand.std()
min_demand = xlpe_demand.min()
max_demand = xlpe_demand.max()

print(f"\nHistorical Demand Statistics:")
print(f"  Average: {avg_demand:.6f} million tons")
print(f"  Std Dev: {std_demand:.6f} million tons")
print(f"  Min: {min_demand:.6f} million tons")
print(f"  Max: {max_demand:.6f} million tons")

# Inventory optimization parameters
class InventoryOptimizer:
    def __init__(self, avg_demand, std_demand, lead_time_days=30, service_level=0.95):
        """
        avg_demand: Average monthly demand (million tons)
        std_demand: Standard deviation of demand (million tons)
        lead_time_days: Lead time for procurement (days)
        service_level: Desired service level (95% = 0.95)
        """
        self.avg_demand = avg_demand
        self.std_demand = std_demand
        self.lead_time_months = lead_time_days / 30  # Convert to months
        self.service_level = service_level
        
        # Cost parameters (adjustable based on company data)
        self.holding_cost_per_ton = 50  # USD per ton per month
        self.ordering_cost = 5000  # USD per order
        self.stockout_cost_per_ton = 500  # USD per ton shortage
        self.material_cost_per_ton = 2000  # USD per ton
        
    def calculate_safety_stock(self):
        """Calculate safety stock using service level"""
        from scipy import stats
        z_score = stats.norm.ppf(self.service_level)
        safety_stock = z_score * self.std_demand * np.sqrt(self.lead_time_months)
        return safety_stock
    
    def calculate_reorder_point(self):
        """Calculate reorder point (ROP)"""
        lead_time_demand = self.avg_demand * self.lead_time_months
        safety_stock = self.calculate_safety_stock()
        rop = lead_time_demand + safety_stock
        return rop
    
    def calculate_economic_order_quantity(self):
        """Calculate EOQ using Wilson formula"""
        annual_demand = self.avg_demand * 12  # Convert to annual
        eoq = np.sqrt((2 * annual_demand * self.ordering_cost) / 
                      (self.holding_cost_per_ton))
        return eoq
    
    def calculate_total_inventory_cost(self, order_quantity, safety_stock):
        """Calculate total inventory cost"""
        annual_demand = self.avg_demand * 12
        
        # Ordering cost
        num_orders = annual_demand / order_quantity
        total_ordering_cost = num_orders * self.ordering_cost
        
        # Holding cost
        avg_inventory = (order_quantity / 2) + safety_stock
        total_holding_cost = avg_inventory * self.holding_cost_per_ton * 12
        
        # Material cost
        total_material_cost = annual_demand * self.material_cost_per_ton
        
        return {
            'ordering_cost': total_ordering_cost,
            'holding_cost': total_holding_cost,
            'material_cost': total_material_cost,
            'total_cost': total_ordering_cost + total_holding_cost + total_material_cost
        }
    
    def optimize(self):
        """Run optimization and return recommendations"""
        safety_stock = self.calculate_safety_stock()
        rop = self.calculate_reorder_point()
        eoq = self.calculate_economic_order_quantity()
        costs = self.calculate_total_inventory_cost(eoq, safety_stock)
        
        # Maximum inventory level
        max_inventory = eoq + safety_stock
        
        return {
            'safety_stock': safety_stock,
            'reorder_point': rop,
            'economic_order_quantity': eoq,
            'max_inventory_level': max_inventory,
            'costs': costs,
            'service_level': self.service_level * 100
        }

# Run optimization
optimizer = InventoryOptimizer(
    avg_demand=avg_demand,
    std_demand=std_demand,
    lead_time_days=30,
    service_level=0.95
)

results = optimizer.optimize()

print(f"\n{'='*80}")
print("INVENTORY OPTIMIZATION RESULTS")
print(f"{'='*80}")
print(f"\nKey Inventory Metrics:")
print(f"  Safety Stock: {results['safety_stock']:.6f} million tons ({results['safety_stock']*1000:.2f} tons)")
print(f"  Reorder Point: {results['reorder_point']:.6f} million tons ({results['reorder_point']*1000:.2f} tons)")
print(f"  Economic Order Quantity (EOQ): {results['economic_order_quantity']:.6f} million tons ({results['economic_order_quantity']*1000:.2f} tons)")
print(f"  Maximum Inventory Level: {results['max_inventory_level']:.6f} million tons ({results['max_inventory_level']*1000:.2f} tons)")
print(f"  Target Service Level: {results['service_level']:.1f}%")

print(f"\nAnnual Cost Analysis:")
print(f"  Ordering Cost: ${results['costs']['ordering_cost']:,.2f}")
print(f"  Holding Cost: ${results['costs']['holding_cost']:,.2f}")
print(f"  Material Cost: ${results['costs']['material_cost']:,.2f}")
print(f"  Total Annual Cost: ${results['costs']['total_cost']:,.2f}")

# Simulate inventory levels over time with forecasted demand
print(f"\n{'='*80}")
print("INVENTORY SIMULATION - NEXT 12 MONTHS")
print(f"{'='*80}")

# Use latest data point for next month forecast
X_features = df_clean.drop(columns=['xlpe_demand_Million_tons'])
latest_features = X_features.iloc[-1:].copy()

# Generate forecasts for next 12 months (with slight variations)
forecast_months = []
inventory_levels = []
current_inventory = results['max_inventory_level']  # Start with max inventory

for month in range(1, 13):
    # Simulate demand for the month (no model prediction)
    demand_variation = np.random.normal(0, std_demand * 0.1)
    adjusted_demand = max(0, avg_demand + demand_variation)
    
    # Check if reorder needed
    if current_inventory <= results['reorder_point']:
        order_quantity = results['economic_order_quantity']
        current_inventory += order_quantity
        reorder_flag = True
    else:
        reorder_flag = False
    
    # Subtract demand
    current_inventory -= adjusted_demand
    
    # Ensure non-negative
    stockout = max(0, -current_inventory)
    current_inventory = max(0, current_inventory)
    
    forecast_months.append({
        'Month': month,
        'Forecasted_Demand': adjusted_demand,
        'Starting_Inventory': current_inventory + adjusted_demand,
        'Ending_Inventory': current_inventory,
        'Reorder_Triggered': reorder_flag,
        'Order_Quantity': order_quantity if reorder_flag else 0,
        'Stockout': stockout
    })
    
    inventory_levels.append(current_inventory)

forecast_df = pd.DataFrame(forecast_months)
forecast_df.to_csv('outputs/inventory_forecast_12months.csv', index=False)

print(forecast_df.to_string(index=False))

# Calculate performance metrics
total_stockouts = forecast_df['Stockout'].sum()
total_orders = forecast_df['Reorder_Triggered'].sum()
avg_inventory = forecast_df['Ending_Inventory'].mean()

print(f"\n{'='*80}")
print("SIMULATION RESULTS (12 Months)")
print(f"{'='*80}")
print(f"  Average Inventory Level: {avg_inventory:.6f} million tons")
print(f"  Total Number of Orders: {total_orders}")
print(f"  Total Stockouts: {total_stockouts:.6f} million tons")
print(f"  Service Level Achieved: {((1 - total_stockouts/forecast_df['Forecasted_Demand'].sum())*100):.2f}%")

# Save optimization results
optimization_summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'safety_stock_million_tons': float(results['safety_stock']),
    'safety_stock_tons': float(results['safety_stock'] * 1000),
    'reorder_point_million_tons': float(results['reorder_point']),
    'reorder_point_tons': float(results['reorder_point'] * 1000),
    'eoq_million_tons': float(results['economic_order_quantity']),
    'eoq_tons': float(results['economic_order_quantity'] * 1000),
    'max_inventory_million_tons': float(results['max_inventory_level']),
    'max_inventory_tons': float(results['max_inventory_level'] * 1000),
    'service_level_percent': float(results['service_level']),
    'annual_costs': {k: float(v) for k, v in results['costs'].items()},
    'simulation_12months': {
        'avg_inventory_million_tons': float(avg_inventory),
        'total_orders': int(total_orders),
        'total_stockouts_million_tons': float(total_stockouts),
        'service_level_achieved': float((1 - total_stockouts/forecast_df['Forecasted_Demand'].sum())*100)
    }
}

with open('outputs/inventory_optimization_results.json', 'w') as f:
    json.dump(optimization_summary, f, indent=4)

# Visualization
print("\nGenerating inventory visualization...")

# 1. Inventory Level Simulation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Inventory levels over time
ax1.plot(forecast_df['Month'], forecast_df['Ending_Inventory'], 'b-', linewidth=2, marker='o', label='Inventory Level')
ax1.axhline(y=results['reorder_point'], color='r', linestyle='--', linewidth=2, label=f'Reorder Point ({results["reorder_point"]:.4f})')
ax1.axhline(y=results['safety_stock'], color='orange', linestyle='--', linewidth=2, label=f'Safety Stock ({results["safety_stock"]:.4f})')
ax1.fill_between(forecast_df['Month'], 0, results['safety_stock'], alpha=0.2, color='orange')
ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
ax1.set_ylabel('Inventory Level (Million Tons)', fontsize=12, fontweight='bold')
ax1.set_title('Inventory Level Simulation - Next 12 Months', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Demand vs Inventory
ax2.plot(forecast_df['Month'], forecast_df['Forecasted_Demand'], 'g-', linewidth=2, marker='s', label='Forecasted Demand')
ax2.plot(forecast_df['Month'], forecast_df['Ending_Inventory'], 'b-', linewidth=2, marker='o', label='Ending Inventory')
ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
ax2.set_ylabel('Million Tons', fontsize=12, fontweight='bold')
ax2.set_title('Demand vs Inventory', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/inventory_simulation.png', dpi=300)
plt.close()

# 2. Cost breakdown pie chart
fig, ax = plt.subplots(figsize=(10, 8))
cost_labels = ['Ordering Cost', 'Holding Cost', 'Material Cost']
cost_values = [
    results['costs']['ordering_cost'],
    results['costs']['holding_cost'],
    results['costs']['material_cost']
]
colors = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.05, 0.05, 0.05)

ax.pie(cost_values, explode=explode, labels=cost_labels, colors=colors,
       autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Annual Inventory Cost Breakdown', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/cost_breakdown.png', dpi=300)
plt.close()

print("\nInventory optimization complete!")
print("\nOutputs saved:")
print("  - outputs/inventory_optimization_results.json")
print("  - outputs/inventory_forecast_12months.csv")
print("  - outputs/inventory_simulation.png")
print("  - outputs/cost_breakdown.png")

print("\n" + "="*80)
print("INVENTORY OPTIMIZATION COMPLETE")
print("="*80)
