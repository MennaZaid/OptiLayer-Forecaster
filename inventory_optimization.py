import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from datetime import datetime

print("=" * 80)
print("INVENTORY OPTIMIZER: USING EXISTING AI FORECASTS")
print("=" * 80)


class InventoryOptimizer:
    def __init__(self, forecast_path='outputs/Ensemble_Risk_Analysis.csv',
                 historical_path='historical_xlpe_demand.xlsx'):
        """
        Uses YOUR AI forecasts - doesn't forecast anything new.
        
        Parameters:
        -----------
        forecast_path : str
            Path to your AI forecast outputs (Ensemble_Risk_Analysis.csv)
        historical_path : str
            Path to historical demand data (for demand std dev)
        """
        # Load YOUR existing AI forecasts
        self.load_ai_forecasts(forecast_path)
        
        # Load historical data for demand statistics
        self.load_historical_stats(historical_path)
        
        print(f"✓ Loaded {len(self.forecast_data)} AI forecasts")
        print(f"✓ Latest forecast: {self.latest_forecast['Forecast']:.6f}M tons")
        print(f"✓ AI uncertainty (σ): {self.latest_forecast['Uncertainty_Sigma']:.6f}")
    
    def load_ai_forecasts(self, path):
        """Load YOUR existing AI forecasts"""
        try:
            self.forecast_data = pd.read_csv(path)
            self.latest_forecast = self.forecast_data.iloc[-1].to_dict()
        except FileNotFoundError:
            print(f"❌ Error: AI forecast file not found: {path}")
            print("Run Models_Advanced.py first to generate forecasts")
            raise
    
    def load_historical_stats(self, path):
        """Load historical demand for statistics"""
        try:
            df = pd.read_excel(path)
            demand = df['xlpe_demand_Million_tons'].dropna()
            self.historical_mean = demand.mean()
            self.historical_std = demand.std()
        except:
            print(f"⚠️  Could not load historical data, using AI uncertainty only")
            self.historical_std = self.latest_forecast.get('Uncertainty_Sigma', 0.05)
    
    def calculate_safety_stock(self, service_level=0.95, lead_time_days=30, 
                               use_ai_uncertainty=True):
        """
        Calculate safety stock using YOUR AI's uncertainty or historical std dev.
        
        Parameters:
        -----------
        service_level : float
            Target service level (0.95 = 95%)
        lead_time_days : int
            Lead time in days
        use_ai_uncertainty : bool
            True: Use AI forecast uncertainty (recommended)
            False: Use historical demand std dev
        """
        # Z-score for service level
        z_score = stats.norm.ppf(service_level)
        
        # Use AI uncertainty if available and requested
        if use_ai_uncertainty and 'Uncertainty_Sigma' in self.latest_forecast:
            demand_std = self.latest_forecast['Uncertainty_Sigma']
            uncertainty_source = "AI Forecast Uncertainty"
        else:
            demand_std = self.historical_std
            uncertainty_source = "Historical Demand Std Dev"
        
        # Convert lead time to months
        lead_time_months = lead_time_days / 30.0
        
        # Safety stock formula
        safety_stock = z_score * demand_std * np.sqrt(lead_time_months)
        
        return {
            'safety_stock': safety_stock,
            'z_score': z_score,
            'demand_std': demand_std,
            'lead_time_months': lead_time_months,
            'uncertainty_source': uncertainty_source,
            'service_level': service_level * 100
        }
    
    def calculate_eoq(self, annual_demand, ordering_cost, holding_cost_per_unit):
        """
        Economic Order Quantity (EOQ) - standard formula.
        
        Parameters:
        -----------
        annual_demand : float
            Annual demand in units
        ordering_cost : float
            Cost per order ($)
        holding_cost_per_unit : float
            Holding cost per unit per year ($/unit/year)
        
        Note: Convert holding cost to per year if needed
        """
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        
        return {
            'eoq': eoq,
            'annual_demand': annual_demand,
            'ordering_cost': ordering_cost,
            'holding_cost': holding_cost_per_unit,
            'orders_per_year': annual_demand / eoq if eoq > 0 else 0,
            'time_between_orders': 365 / (annual_demand / eoq) if eoq > 0 else 0
        }
    
    def optimize(self, user_inputs):
        """
        Main optimization using YOUR AI forecasts + user inputs.
        
        Parameters:
        -----------
        user_inputs : dict
            {
                'material_cost': 2350,      # $/ton
                'holding_cost': 75,         # $/ton/month
                'ordering_cost': 7500,      # $/order
                'stockout_cost': 1200,      # $/ton
                'service_level': 0.95,      # 95%
                'lead_time_days': 30        # days
            }
        """
        print(f"\n{'='*80}")
        print("RUNNING OPTIMIZATION WITH YOUR AI FORECASTS")
        print(f"{'='*80}")
        
        # 1. Get YOUR AI forecast
        ai_forecast = self.latest_forecast['Forecast']
        print(f"📊 Using YOUR AI forecast: {ai_forecast:.6f} million tons/month")
        
        # 2. Calculate safety stock
        safety_info = self.calculate_safety_stock(
            service_level=user_inputs['service_level'],
            lead_time_days=user_inputs['lead_time_days'],
            use_ai_uncertainty=True
        )
        
        print(f"✓ Safety stock: {safety_info['safety_stock']:.6f}M tons")
        print(f"  (Using {safety_info['uncertainty_source']})")
        
        # 3. Calculate annual demand from AI forecast
        monthly_demand = ai_forecast
        annual_demand = monthly_demand * 12
        
        # 4. Calculate EOQ
        # Convert holding cost to annual rate if needed
        holding_cost_per_ton_year = user_inputs['holding_cost'] * 12
        
        eoq_info = self.calculate_eoq(
            annual_demand=annual_demand,
            ordering_cost=user_inputs['ordering_cost'],
            holding_cost_per_unit=holding_cost_per_ton_year
        )
        
        print(f"✓ EOQ: {eoq_info['eoq']:.6f}M tons")
        print(f"  Orders/year: {eoq_info['orders_per_year']:.1f}")
        
        # 5. Calculate reorder point
        lead_time_months = user_inputs['lead_time_days'] / 30.0
        lead_time_demand = monthly_demand * lead_time_months
        reorder_point = lead_time_demand + safety_info['safety_stock']
        
        # 6. Calculate costs
        ordering_cost_total = eoq_info['orders_per_year'] * user_inputs['ordering_cost']
        
        avg_inventory = (eoq_info['eoq'] / 2) + safety_info['safety_stock']
        holding_cost_total = avg_inventory * user_inputs['holding_cost'] * 12
        
        material_cost_total = annual_demand * user_inputs['material_cost']
        
        # Simplified stockout cost
        stockout_probability = 1 - user_inputs['service_level']
        expected_stockouts = self.historical_std * stockout_probability * np.sqrt(lead_time_months)
        stockout_cost_total = expected_stockouts * user_inputs['stockout_cost'] * 12
        
        total_cost = ordering_cost_total + holding_cost_total + material_cost_total + stockout_cost_total
        
        # 7. Compile results
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ai_forecast_used': {
                'monthly_demand': ai_forecast,
                'annual_demand': annual_demand,
                'uncertainty_sigma': self.latest_forecast.get('Uncertainty_Sigma', 0),
                'hedging_factor': self.latest_forecast.get('Hedging_Factor', 1.0)
            },
            'inventory_policy': {
                'safety_stock': safety_info['safety_stock'],
                'reorder_point': reorder_point,
                'economic_order_quantity': eoq_info['eoq'],
                'max_inventory_level': eoq_info['eoq'] + safety_info['safety_stock'],
                'lead_time_demand': lead_time_demand,
                'lead_time_days': user_inputs['lead_time_days']
            },
            'cost_analysis': {
                'total_annual_cost': total_cost,
                'ordering_cost': ordering_cost_total,
                'holding_cost': holding_cost_total,
                'material_cost': material_cost_total,
                'stockout_cost': stockout_cost_total,
                'avg_inventory': avg_inventory,
                'orders_per_year': eoq_info['orders_per_year']
            },
            'performance_metrics': {
                'service_level_percent': user_inputs['service_level'] * 100,
                'inventory_turnover': annual_demand / avg_inventory if avg_inventory > 0 else 0,
                'avg_inventory_cover_days': (avg_inventory / monthly_demand) * 30,
                'expected_stockouts': expected_stockouts
            },
            'user_inputs_used': user_inputs,
            'calculation_notes': {
                'safety_stock_source': safety_info['uncertainty_source'],
                'holding_cost_basis': 'per month, converted to annual for EOQ',
                'demand_source': 'AI Ensemble Forecast'
            }
        }
        
        return results
    
    def run_what_if_scenario(self, base_inputs, scenario_changes):
        """
        Run what-if scenario analysis.
        
        Example:
        --------
        scenario_changes = {
            'material_cost': 1.2,  # 20% increase
            'holding_cost': 0.8    # 20% decrease
        }
        """
        print(f"\n🔍 Running What-If Scenario")
        
        # Apply changes
        scenario_inputs = base_inputs.copy()
        for key, multiplier in scenario_changes.items():
            if key in scenario_inputs and key != 'service_level':
                scenario_inputs[key] = scenario_inputs[key] * multiplier
            elif key == 'service_level':
                scenario_inputs[key] = scenario_inputs[key] * multiplier
        
        # Run optimization
        scenario_results = self.optimize(scenario_inputs)
        
        return {
            'scenario_changes': scenario_changes,
            'results': scenario_results
        }
    
    def save_results(self, results, output_dir='outputs'):
        """Save optimization results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON
        json_path = os.path.join(output_dir, 'inventory_recommendations.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save summary CSV
        summary_data = {
            'Metric': [
                'AI Monthly Forecast (M tons)',
                'AI Uncertainty (σ)',
                'Safety Stock (M tons)',
                'Reorder Point (M tons)',
                'EOQ (M tons)',
                'Total Annual Cost ($)',
                'Service Level (%)',
                'Orders per Year',
                'Avg Inventory (M tons)'
            ],
            'Value': [
                results['ai_forecast_used']['monthly_demand'],
                results['ai_forecast_used']['uncertainty_sigma'],
                results['inventory_policy']['safety_stock'],
                results['inventory_policy']['reorder_point'],
                results['inventory_policy']['economic_order_quantity'],
                results['cost_analysis']['total_annual_cost'],
                results['performance_metrics']['service_level_percent'],
                results['cost_analysis']['orders_per_year'],
                results['cost_analysis']['avg_inventory']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, 'inventory_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        
        print(f"\n✅ Results saved:")
        print(f"  • {json_path}")
        print(f"  • {csv_path}")
        
        return {
            'json': json_path,
            'csv': csv_path
        }

# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("="*80)
    
    # Create optimizer (uses YOUR existing forecasts)
    optimizer = InventoryOptimizer()
    
    # Example user inputs (from dashboard)
    example_inputs = {
        'material_cost': 2350,      # $/ton
        'holding_cost': 75,         # $/ton/month
        'ordering_cost': 7500,      # $/order
        'stockout_cost': 1200,      # $/ton
        'service_level': 0.95,      # 95%
        'lead_time_days': 30        # days
    }
    
    # Run optimization using YOUR AI forecasts
    results = optimizer.optimize(example_inputs)
    
    # Save results
    optimizer.save_results(results)
    
    print(f"\n📊 KEY RECOMMENDATIONS:")
    print(f"  Safety Stock: {results['inventory_policy']['safety_stock']:.4f}M tons")
    print(f"  Reorder Point: {results['inventory_policy']['reorder_point']:.4f}M tons")
    print(f"  EOQ: {results['inventory_policy']['economic_order_quantity']:.4f}M tons")
    print(f"  Total Cost: ${results['cost_analysis']['total_annual_cost']:,.0f}")
    
