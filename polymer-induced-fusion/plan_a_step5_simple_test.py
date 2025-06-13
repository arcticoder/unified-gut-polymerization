"""
Simple test for Plan A Step 5 reactor design functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import json
import os

def simple_reactor_test():
    """Run a simplified version of the reactor design test"""
    print("=" * 60)
    print("PLAN A STEP 5: REACTOR DESIGN - SIMPLE TEST")
    print("=" * 60)
    
    # Basic parameters
    mu_values = np.linspace(1.0, 10.0, 10)
    energy_values = np.linspace(1022, 5000, 10)  # keV
    
    # Calculate cross-section enhancement
    print("\\nCross-section enhancement with polymer scale:")
    for mu in [1.0, 5.0, 10.0]:
        polymer_factor = 1.0 + 0.5 * np.log(1 + mu) + 0.2 * mu**0.6
        print(f"  μ = {mu:.1f}: Enhancement factor = {polymer_factor:.2f}")
    
    # Energy conversion efficiency
    print("\\nEnergy conversion efficiency:")
    gamma_to_heat = 0.95
    heat_to_electric_base = 0.35
    
    for mu in [1.0, 5.0, 10.0]:
        thermal_enhancement = 1 + 0.2 * np.log(1 + mu)
        electric_enhancement = 1 + 0.15 * np.log(1 + mu)
        
        overall_efficiency = (gamma_to_heat * heat_to_electric_base * 
                            thermal_enhancement * electric_enhancement * 0.9)
        
        print(f"  μ = {mu:.1f}: Overall efficiency = {overall_efficiency:.2f}")
    
    # Cost analysis
    print("\\nCost analysis (simplified):")
    antimatter_cost_per_gram = 6.25e13  # $62.5 trillion/gram
    target_cost_per_kwh = 0.10
    
    # For 1 kg antimatter over 20 years
    antimatter_mass_kg = 1e-6  # 1 mg
    production_cost = antimatter_mass_kg * 1000 * antimatter_cost_per_gram
    
    # Energy yield: E = mc²
    energy_yield_j = antimatter_mass_kg * constants.c**2
    energy_yield_kwh = energy_yield_j / 3.6e6
    
    # Cost per kWh (production only)
    cost_per_kwh_production = production_cost / energy_yield_kwh
    
    print(f"  Antimatter mass: {antimatter_mass_kg*1e6:.1f} mg")
    print(f"  Energy yield: {energy_yield_kwh:.2e} kWh")
    print(f"  Production cost: ${production_cost:.2e}")
    print(f"  Cost/kWh (production only): ${cost_per_kwh_production:.2f}")
    print(f"  Target cost: ${target_cost_per_kwh:.2f}/kWh")
    
    if cost_per_kwh_production < target_cost_per_kwh:
        print("  → TARGET ACHIEVED!")
    else:
        reduction_needed = cost_per_kwh_production / target_cost_per_kwh
        print(f"  → Need {reduction_needed:.0f}x cost reduction")
    
    # Parameter space visualization
    print("\\nGenerating parameter space visualization...")
    
    mu_grid, energy_grid = np.meshgrid(mu_values, energy_values, indexing='ij')
    
    # Calculate yield matrix
    yield_matrix = np.zeros_like(mu_grid)
    for i, mu in enumerate(mu_values):
        for j, energy in enumerate(energy_values):
            if energy > 1022:  # Above pair production threshold
                polymer_factor = 1.0 + 0.5 * np.log(1 + mu) + 0.2 * mu**0.6
                base_cross_section = 6.65e-25 * np.log(energy/1022) * (1 - 1022/energy)
                enhanced_cross_section = base_cross_section * polymer_factor
                yield_matrix[i, j] = enhanced_cross_section * 1e15  # arbitrary flux
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contourf(energy_grid, mu_grid, yield_matrix, levels=20, cmap='viridis')
    plt.colorbar(label='Pair Production Yield')
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('Polymer Scale μ')
    plt.title('Pair Production Yield vs (E, μ)')
    
    plt.subplot(1, 2, 2)
    # Cost landscape (simplified)
    cost_matrix = np.zeros_like(mu_grid)
    for i, mu in enumerate(mu_values):
        for j, energy in enumerate(energy_values):
            # Simplified cost model
            base_cost = 1000  # Arbitrary units
            efficiency = 0.3 * (1 + 0.2 * np.log(1 + mu))
            cost_matrix[i, j] = base_cost / efficiency
    
    plt.contourf(energy_grid, mu_grid, cost_matrix, levels=20, cmap='coolwarm')
    plt.colorbar(label='Cost per kWh (arbitrary units)')
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('Polymer Scale μ')
    plt.title('Cost Landscape vs (E, μ)')
    
    plt.tight_layout()
    
    # Save results
    output_dir = "plan_a_step5_reactor_design"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "reactor_parameter_space.png"), dpi=150)
    print(f"  Visualization saved to: {output_dir}/reactor_parameter_space.png")
    
    # Save data
    results = {
        'test_type': 'simplified_reactor_design',
        'mu_values': mu_values.tolist(),
        'energy_values': energy_values.tolist(),
        'yield_matrix': yield_matrix.tolist(),
        'cost_matrix': cost_matrix.tolist(),
        'key_findings': {
            'production_cost_per_kwh': cost_per_kwh_production,
            'target_cost_per_kwh': target_cost_per_kwh,
            'cost_reduction_needed': cost_per_kwh_production / target_cost_per_kwh,
            'max_yield': float(np.max(yield_matrix)),
            'min_cost': float(np.min(cost_matrix))
        }
    }
    
    results_file = os.path.join(output_dir, "simple_reactor_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Results saved to: {results_file}")
    
    print("\\n" + "=" * 60)
    print("SIMPLE REACTOR TEST COMPLETE")
    print("Key finding: Antimatter production cost is the limiting factor")
    print("Need breakthrough cost reductions for economic viability")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    results = simple_reactor_test()
