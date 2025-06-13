"""
Plan A, Step 4: Net kWh Cost Calculation
=======================================

This module implements Step 4 of Plan A: computing the net cost per kWh
using the specific formula that combines antimatter production costs with
conversion efficiency losses.

Formula:
Cost_kWh ≈ ($6.25×10¹³ /g) / (2.5×10¹⁰ kWh/g × η) ~ $2,500/η per kWh

Where:
- η is the total conversion efficiency (0-1)
- Production cost: $6.25×10¹³ per gram ($62.5 trillion/gram) 
- Energy density: 2.5×10¹⁰ kWh per gram theoretical

Benchmarked against WEST tokamak baseline (February 12, 2025).
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os

class NetCostCalculator:
    """Net kWh cost calculator for Plan A Step 4"""
    
    def __init__(self):
        # Constants from the formula
        self.antimatter_cost_per_gram = 6.25e13  # $62.5 trillion per gram
        self.theoretical_energy_per_gram = 2.5e10  # kWh per gram theoretical
        self.cost_efficiency_factor = 2500.0  # $2,500/η coefficient
        
        # WEST baseline for comparison
        self.west_baseline = {
            'confinement_time_s': 1337.0,
            'plasma_temperature_c': 50e6,
            'heating_power_w': 2e6,
            'total_energy_kwh': (2e6 * 1337.0) / 3.6e6,  # 742.78 kWh
            'date': '2025-02-12'
        }
        
        # Market thresholds ($/kWh)
        self.market_thresholds = {
            'grid_competitive': 0.10,
            'premium_residential': 0.30,
            'premium_industrial': 1.00,
            'space_applications': 1000.00,
            'ultra_premium': 10000.00
        }
        
        # Conversion efficiency data from Step 3
        self.conversion_methods = {
            'tpv_system': {
                'name': 'TPV Full System',
                'base_efficiency': 0.05,
                'polymer_enhanced': 0.075,
                'description': 'Real-world thermophotovoltaic'
            },
            'tpv_lab': {
                'name': 'TPV Laboratory',
                'base_efficiency': 0.35,
                'polymer_enhanced': 0.525,
                'description': 'Best laboratory conditions'
            },
            'thermionic': {
                'name': 'Thermionic',
                'base_efficiency': 0.15,
                'polymer_enhanced': 0.195,
                'description': 'Thermionic conversion'
            },
            'direct': {
                'name': 'Direct Conversion',
                'base_efficiency': 0.25,
                'polymer_enhanced': 0.50,
                'description': 'Theoretical optimum'
            }
        }
    
    def calculate_net_cost_per_kwh(self, efficiency: float) -> Dict[str, float]:
        """
        Calculate net cost per kWh using the Step 4 formula
        
        Args:
            efficiency: Total conversion efficiency (0-1)
            
        Returns:
            Cost analysis results
        """
        if efficiency <= 0:
            return {
                'efficiency': efficiency,
                'cost_per_kwh_formula': float('inf'),
                'cost_per_kwh_detailed': float('inf'),
                'theoretical_energy_kwh_per_gram': self.theoretical_energy_per_gram,
                'production_cost_per_gram': self.antimatter_cost_per_gram
            }
        
        # Step 4 formula: Cost ≈ $2,500/η per kWh
        cost_formula = self.cost_efficiency_factor / efficiency
        
        # Detailed calculation for verification
        realistic_energy_per_gram = self.theoretical_energy_per_gram * efficiency
        cost_detailed = self.antimatter_cost_per_gram / realistic_energy_per_gram
        
        return {
            'efficiency': efficiency,
            'cost_per_kwh_formula': cost_formula,
            'cost_per_kwh_detailed': cost_detailed,
            'theoretical_energy_kwh_per_gram': self.theoretical_energy_per_gram,
            'realistic_energy_kwh_per_gram': realistic_energy_per_gram,
            'production_cost_per_gram': self.antimatter_cost_per_gram,
            'formula_verification': abs(cost_formula - cost_detailed) < 1e-6
        }
    
    def analyze_conversion_methods(self) -> Dict[str, Dict]:
        """Analyze cost for all conversion methods from Step 3"""
        results = {}
        
        for method_key, method_data in self.conversion_methods.items():
            # Standard efficiency
            std_cost = self.calculate_net_cost_per_kwh(method_data['base_efficiency'])
            
            # Polymer-enhanced efficiency
            enh_cost = self.calculate_net_cost_per_kwh(method_data['polymer_enhanced'])
            
            results[method_key] = {
                'name': method_data['name'],
                'description': method_data['description'],
                'base_efficiency': method_data['base_efficiency'],
                'polymer_efficiency': method_data['polymer_enhanced'],
                'standard_cost': std_cost,
                'enhanced_cost': enh_cost,
                'cost_improvement_factor': std_cost['cost_per_kwh_formula'] / enh_cost['cost_per_kwh_formula']
            }
        
        return results
    
    def market_viability_analysis(self, max_efficiency: float = 0.8) -> Dict:
        """
        Analyze market viability across efficiency range
        
        Args:
            max_efficiency: Maximum efficiency to consider
            
        Returns:
            Market viability analysis
        """
        # Generate efficiency range
        efficiency_values = np.linspace(0.001, max_efficiency, 1000)
        
        results = {
            'efficiency_values': efficiency_values.tolist(),
            'costs_per_kwh': [],
            'market_accessibility': {threshold: [] for threshold in self.market_thresholds.keys()},
            'required_efficiencies': {},
            'west_comparison': {}
        }
        
        # Calculate costs for each efficiency
        for eta in efficiency_values:
            cost = self.calculate_net_cost_per_kwh(eta)['cost_per_kwh_formula']
            results['costs_per_kwh'].append(cost)
            
            # Check market accessibility
            for threshold_name, threshold_cost in self.market_thresholds.items():
                accessible = cost <= threshold_cost
                results['market_accessibility'][threshold_name].append(accessible)
        
        # Find required efficiencies for each market threshold
        for threshold_name, threshold_cost in self.market_thresholds.items():
            required_eta = self.cost_efficiency_factor / threshold_cost
            if required_eta <= 1.0:
                results['required_efficiencies'][threshold_name] = {
                    'efficiency_required': required_eta,
                    'achievable': required_eta <= max_efficiency,
                    'threshold_cost': threshold_cost
                }
            else:
                results['required_efficiencies'][threshold_name] = {
                    'efficiency_required': required_eta,
                    'achievable': False,
                    'threshold_cost': threshold_cost,
                    'note': 'Requires >100% efficiency (impossible)'
                }
        
        # WEST comparison
        west_cost_per_kwh = 0.10  # Approximate grid cost
        west_energy_per_dollar = 1.0 / west_cost_per_kwh  # 10 kWh per dollar
        
        # Find efficiency where antimatter matches WEST economics
        required_eta_west = self.cost_efficiency_factor / west_cost_per_kwh
        
        results['west_comparison'] = {
            'west_cost_per_kwh': west_cost_per_kwh,
            'west_energy_kwh': self.west_baseline['total_energy_kwh'],
            'efficiency_for_west_parity': required_eta_west,
            'west_parity_achievable': required_eta_west <= 1.0,
            'efficiency_gap': required_eta_west - max_efficiency if required_eta_west > max_efficiency else 0
        }
        
        return results
    
    def step4_demonstration_analysis(self) -> Dict:
        """Run the complete Step 4 demonstration analysis"""
        print("=" * 80)
        print("PLAN A, STEP 4: NET kWh COST CALCULATION")
        print("Formula: Cost ≈ $2,500/η per kWh")
        print("=" * 80)
        print()
        
        print("FORMULA VERIFICATION:")
        print("-" * 30)
        print(f"Antimatter cost: ${self.antimatter_cost_per_gram:.2e}/gram")
        print(f"Theoretical energy: {self.theoretical_energy_per_gram:.2e} kWh/gram")
        print(f"Cost factor: ${self.cost_efficiency_factor:.0f}/η")
        print()
        
        # Test the specific example from the prompt: η = 0.5
        eta_example = 0.5
        cost_example = self.calculate_net_cost_per_kwh(eta_example)
        
        print(f"EXAMPLE CALCULATION (η = {eta_example}):")
        print("-" * 40)
        print(f"Cost per kWh = ${self.cost_efficiency_factor:.0f}/{eta_example} = ${cost_example['cost_per_kwh_formula']:.0f}/kWh")
        print(f"Detailed calculation: ${cost_example['cost_per_kwh_detailed']:.0f}/kWh")
        print(f"Formula verification: {'✓ Passed' if cost_example['formula_verification'] else '✗ Failed'}")
        print()
        
        # Compare with market thresholds
        print("MARKET THRESHOLD COMPARISON:")
        print("-" * 35)
        for threshold_name, threshold_cost in self.market_thresholds.items():
            factor_above = cost_example['cost_per_kwh_formula'] / threshold_cost
            print(f"  {threshold_name.replace('_', ' ').title()}: "
                  f"${threshold_cost:.2f}/kWh → {factor_above:.0f}× too expensive")
        print()
        
        # Analyze all conversion methods
        print("CONVERSION METHOD ANALYSIS:")
        print("-" * 35)
        method_results = self.analyze_conversion_methods()
        
        for method_key, data in method_results.items():
            print(f"\\n{data['name']} ({data['description']}):")
            print(f"  Base efficiency: {data['base_efficiency']*100:.1f}% → ${data['standard_cost']['cost_per_kwh_formula']:.0f}/kWh")
            print(f"  Polymer enhanced: {data['polymer_efficiency']*100:.1f}% → ${data['enhanced_cost']['cost_per_kwh_formula']:.0f}/kWh")
            print(f"  Improvement factor: {data['cost_improvement_factor']:.1f}×")
            
            # Check if any method reaches market thresholds
            best_cost = data['enhanced_cost']['cost_per_kwh_formula']
            viable_markets = [name for name, threshold in self.market_thresholds.items() 
                            if best_cost <= threshold]
            
            if viable_markets:
                print(f"  Viable for: {', '.join(viable_markets)}")
            else:
                print("  Not viable for any market segment")
        
        # Market viability analysis
        print()
        print("REQUIRED EFFICIENCIES FOR MARKET ACCESS:")
        print("-" * 45)
        viability = self.market_viability_analysis()
        
        for threshold_name, data in viability['required_efficiencies'].items():
            eta_req = data['efficiency_required']
            achievable = data['achievable']
            threshold_cost = data['threshold_cost']
            
            if eta_req <= 1.0:
                print(f"  {threshold_name.replace('_', ' ').title()}: "
                      f"η ≥ {eta_req:.3f} ({eta_req*100:.1f}%) - "
                      f"{'Potentially achievable' if achievable else 'Beyond current technology'}")
            else:
                print(f"  {threshold_name.replace('_', ' ').title()}: "
                      f"η ≥ {eta_req:.1f} (>{eta_req*100:.0f}%) - Physically impossible")
        
        # WEST comparison
        print()
        print("WEST TOKAMAK COMPARISON:")
        print("-" * 30)
        west_data = viability['west_comparison']
        print(f"WEST baseline energy: {self.west_baseline['total_energy_kwh']:.1f} kWh")
        print(f"WEST equivalent cost: ~${west_data['west_cost_per_kwh']:.2f}/kWh")
        print(f"Required efficiency for parity: η = {west_data['efficiency_for_west_parity']:.1f}")
        print(f"Parity achievable: {'No - requires >100% efficiency' if not west_data['west_parity_achievable'] else 'Theoretically possible'}")
        
        if not west_data['west_parity_achievable']:
            print(f"Efficiency gap: {west_data['efficiency_gap']:.1f} (impossible)")
        
        # Key conclusions
        print()
        print("KEY CONCLUSIONS:")
        print("-" * 20)
        print("1. Even at η = 50%, cost is $5,000/kWh (50,000× grid prices)")
        print("2. Grid parity requires impossible efficiency (η = 25,000)")
        print("3. Space applications require η ≥ 2.5 (250% - impossible)")
        print("4. Ultra-premium markets require η ≥ 0.25 (25% - barely achievable)")
        print("5. Fundamental physics limits make economic viability impossible")
        
        # Return comprehensive results
        return {
            'formula_verification': cost_example,
            'conversion_methods': method_results,
            'market_viability': viability,
            'west_comparison': west_data,
            'conclusions': {
                'economically_viable': False,
                'best_case_cost': min([data['enhanced_cost']['cost_per_kwh_formula'] 
                                     for data in method_results.values()]),
                'best_case_method': min(method_results.keys(), 
                                      key=lambda k: method_results[k]['enhanced_cost']['cost_per_kwh_formula']),
                'grid_parity_impossible': True,
                'space_applications_impossible': True
            }
        }
    
    def generate_step4_visualization(self, results: Dict, save_path: Optional[str] = None):
        """Generate comprehensive Step 4 visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Cost vs Efficiency (Step 4 Formula)
        efficiency_range = np.linspace(0.01, 0.8, 100)
        costs = [self.cost_efficiency_factor / eta for eta in efficiency_range]
        
        ax1.loglog(efficiency_range * 100, costs, 'b-', linewidth=3, label='$2,500/η Formula')
        
        # Add market threshold lines
        for threshold_name, threshold_cost in self.market_thresholds.items():
            if threshold_cost < 1e6:  # Only show reasonable thresholds
                ax1.axhline(y=threshold_cost, linestyle='--', alpha=0.7,
                           label=f'{threshold_name.replace("_", " ").title()} (${threshold_cost:.0f})')
        
        # Mark conversion methods
        method_results = results['conversion_methods']
        for method_key, data in method_results.items():
            eta_std = data['base_efficiency'] * 100
            eta_enh = data['polymer_efficiency'] * 100
            cost_std = data['standard_cost']['cost_per_kwh_formula']
            cost_enh = data['enhanced_cost']['cost_per_kwh_formula']
            
            ax1.scatter(eta_std, cost_std, s=100, alpha=0.7, label=f'{data["name"]} (Standard)')
            ax1.scatter(eta_enh, cost_enh, s=100, alpha=0.7, marker='^', 
                       label=f'{data["name"]} (Enhanced)')
        
        ax1.set_xlabel('Conversion Efficiency (%)')
        ax1.set_ylabel('Cost per kWh ($)')
        ax1.set_title('Step 4: Net Cost Formula ($2,500/η)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Market accessibility analysis
        viability = results['market_viability']
        
        # Show required efficiencies for market access
        markets = []
        required_effs = []
        achievable_colors = []
        
        for threshold_name, data in viability['required_efficiencies'].items():
            if data['efficiency_required'] <= 5.0:  # Only show reasonable requirements
                markets.append(threshold_name.replace('_', '\\n'))
                required_effs.append(data['efficiency_required'] * 100)
                achievable_colors.append('green' if data['achievable'] else 'red')
        
        bars = ax2.bar(markets, required_effs, color=achievable_colors, alpha=0.7)
        ax2.set_ylabel('Required Efficiency (%)')
        ax2.set_title('Efficiency Requirements for Market Access')
        ax2.set_yscale('log')
        
        # Add 100% efficiency line
        ax2.axhline(y=100, color='black', linestyle='-', linewidth=2, label='Physical Limit (100%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Conversion method comparison
        methods = list(method_results.keys())
        method_names = [method_results[m]['name'] for m in methods]
        std_costs = [method_results[m]['standard_cost']['cost_per_kwh_formula'] for m in methods]
        enh_costs = [method_results[m]['enhanced_cost']['cost_per_kwh_formula'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax3.bar(x - width/2, std_costs, width, label='Standard', alpha=0.7)
        ax3.bar(x + width/2, enh_costs, width, label='Polymer Enhanced', alpha=0.7)
        
        ax3.set_xlabel('Conversion Method')
        ax3.set_ylabel('Cost per kWh ($)')
        ax3.set_title('Cost Comparison by Method')
        ax3.set_yscale('log')
        ax3.set_xticks(x)
        ax3.set_xticklabels([name.replace(' ', '\\n') for name in method_names])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Economic gap visualization
        best_cost = results['conclusions']['best_case_cost']
        
        gaps = {}
        for threshold_name, threshold_cost in self.market_thresholds.items():
            if threshold_cost >= 0.1:  # Skip sub-grid thresholds
                gap = best_cost / threshold_cost
                gaps[threshold_name.replace('_', '\\n')] = gap
        
        gap_names = list(gaps.keys())
        gap_values = list(gaps.values())
        colors = ['red' if gap > 1000 else 'orange' if gap > 100 else 'yellow' if gap > 10 else 'green' 
                 for gap in gap_values]
        
        bars = ax4.bar(gap_names, gap_values, color=colors, alpha=0.7)
        ax4.set_ylabel('Cost Factor Above Threshold')
        ax4.set_title('Economic Gap Analysis (Best Case)')
        ax4.set_yscale('log')
        
        # Add reference line at 1× (break-even)
        ax4.axhline(y=1, color='green', linestyle='--', linewidth=2, label='Break-even')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, gap in zip(bars, gap_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{gap:.0f}×', ha='center', va='bottom', rotation=90)
        
        plt.tight_layout()
        plt.suptitle(f'Plan A Step 4: Net kWh Cost Analysis\\n'
                    f'Formula: Cost ≈ $2,500/η per kWh (WEST Baseline: {self.west_baseline["total_energy_kwh"]:.1f} kWh)',
                    fontsize=14, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Step 4 visualization saved to: {save_path}")
        
        plt.show()

def run_step4_analysis():
    """Run the complete Plan A Step 4 analysis"""
    print("Starting Plan A Step 4: Net kWh Cost Calculation...")
    print()
    
    # Create output directory
    output_dir = "plan_a_step4_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize calculator
    calculator = NetCostCalculator()
    
    # Run demonstration analysis
    results = calculator.step4_demonstration_analysis()
    
    print()
    print("=" * 80)
    print("GENERATING STEP 4 VISUALIZATION...")
    print("=" * 80)
    
    # Generate visualization
    plot_path = os.path.join(output_dir, "step4_net_cost_analysis.png")
    calculator.generate_step4_visualization(results, plot_path)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "step4_net_cost_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {results_file}")
    print()
    print("=" * 80)
    print("STEP 4 ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("FINAL CONCLUSION:")
    print("The Step 4 formula Cost ≈ $2,500/η per kWh demonstrates that")
    print("even perfect polymer enhancement cannot overcome the fundamental")
    print("economic barriers imposed by antimatter production costs and")
    print("thermodynamic conversion limits. Grid parity remains impossible.")
    
    return results

if __name__ == "__main__":
    # Run the complete Step 4 analysis
    results = run_step4_analysis()
