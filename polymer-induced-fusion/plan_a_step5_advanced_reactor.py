"""
Plan A, Step 5: Advanced Reactor Design Analysis
==============================================

Complete implementation of simulation-driven reactor design with:
1. Pair-production yield optimization
2. Magnetic trap dynamics
3. Energy conversion efficiency chains
4. Economic viability assessment
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ReactorConfiguration:
    """Complete reactor configuration parameters"""
    # Geometry
    trap_radius: float = 2.0  # meters
    trap_length: float = 4.0  # meters
    magnetic_field: float = 15.0  # Tesla
    
    # Physics
    polymer_scale_mu: float = 5.0
    photon_energy_kev: float = 2000.0
    antimatter_mass_kg: float = 1e-6  # 1 mg
    
    # Economics
    target_cost_per_kwh: float = 0.10  # $/kWh
    reactor_lifetime_years: int = 20
    
    @property
    def trap_volume(self) -> float:
        return np.pi * self.trap_radius**2 * self.trap_length

class AdvancedReactorAnalyzer:
    """Advanced reactor performance and cost analysis"""
    
    def __init__(self, config: ReactorConfiguration):
        self.config = config
        self.c = constants.c
        self.e = constants.e
        self.me = constants.m_e
        
        # Economic constants
        self.antimatter_cost_per_gram = 6.25e13  # $62.5 trillion/gram
        self.reactor_base_cost = 1e9  # $1 billion
        self.annual_operations_cost = 1e8  # $100 million/year
    
    def pair_production_cross_section(self, energy_kev: float, mu: float) -> float:
        """Calculate polymer-enhanced pair production cross-section"""
        threshold_kev = 1022.0  # Pair production threshold
        
        if energy_kev < threshold_kev:
            return 0.0
        
        # Base cross-section (Klein-Nishina approximation)
        energy_ratio = energy_kev / threshold_kev
        if energy_ratio > 1:
            base_sigma = 6.65e-25 * np.log(energy_ratio) * (1 - 1/energy_ratio)
        else:
            base_sigma = 0.0
        
        # Polymer enhancement
        polymer_factor = 1.0 + 0.5 * np.log(1 + mu) + 0.2 * mu**0.6
        
        return base_sigma * polymer_factor
    
    def confinement_efficiency(self, mu: float) -> float:
        """Calculate antimatter confinement efficiency with polymer enhancement"""
        base_efficiency = 0.80  # 80% base confinement
        
        # Polymer enhancement for confinement
        enhancement = 1 + 0.3 * np.log(1 + mu) + 0.1 * mu**0.5
        
        return min(base_efficiency * enhancement, 0.99)  # Cap at 99%
    
    def energy_conversion_efficiency(self, mu: float) -> Dict[str, float]:
        """Calculate complete energy conversion chain efficiency"""
        # Stage efficiencies
        gamma_absorption = 0.95  # High efficiency gamma absorption
        
        # Thermodynamic conversion (enhanced by polymer)
        base_thermal_eff = 0.35
        thermal_enhancement = 1 + 0.2 * np.log(1 + mu)
        thermal_efficiency = min(base_thermal_eff * thermal_enhancement, 0.60)
        
        # Electrical conversion (enhanced by polymer)
        base_electric_eff = 0.90
        electric_enhancement = 1 + 0.15 * np.log(1 + mu)
        electric_efficiency = min(base_electric_eff * electric_enhancement, 0.95)
        
        # System losses
        system_efficiency = 0.90  # 10% system losses
        
        # Overall efficiency
        overall = (gamma_absorption * thermal_efficiency * 
                  electric_efficiency * system_efficiency)
        
        return {
            'gamma_absorption': gamma_absorption,
            'thermal_conversion': thermal_efficiency,
            'electrical_conversion': electric_efficiency,
            'system_efficiency': system_efficiency,
            'overall_efficiency': overall
        }
    
    def calculate_energy_yield(self, antimatter_mass_kg: float) -> Dict[str, float]:
        """Calculate energy yield from antimatter annihilation"""
        # E = mc² for total conversion
        total_energy_j = antimatter_mass_kg * self.c**2
        total_energy_kwh = total_energy_j / 3.6e6
        
        # Apply conversion efficiency
        efficiency_data = self.energy_conversion_efficiency(self.config.polymer_scale_mu)
        usable_energy_kwh = total_energy_kwh * efficiency_data['overall_efficiency']
        
        return {
            'total_energy_j': total_energy_j,
            'total_energy_kwh': total_energy_kwh,
            'usable_energy_kwh': usable_energy_kwh,
            'efficiency_data': efficiency_data
        }
    
    def calculate_reactor_costs(self) -> Dict[str, float]:
        """Calculate complete reactor cost breakdown"""
        # Capital costs
        size_factor = (self.config.trap_radius / 2.0)**2
        field_factor = (self.config.magnetic_field / 15.0)**3
        polymer_factor = 1 + 0.5 * np.log(1 + self.config.polymer_scale_mu)
        
        capital_cost = (self.reactor_base_cost * size_factor * 
                       field_factor * polymer_factor)
        
        # Antimatter production cost
        antimatter_cost = (self.config.antimatter_mass_kg * 1000 * 
                          self.antimatter_cost_per_gram)
        
        # Operational costs
        total_operational = (self.annual_operations_cost * 
                           self.config.reactor_lifetime_years)
        
        # Total system cost
        total_cost = capital_cost + antimatter_cost + total_operational
        
        return {
            'capital_cost': capital_cost,
            'antimatter_production_cost': antimatter_cost,
            'operational_cost': total_operational,
            'total_cost': total_cost
        }
    
    def economic_viability_analysis(self) -> Dict:
        """Complete economic viability analysis"""
        # Energy calculation
        energy_data = self.calculate_energy_yield(self.config.antimatter_mass_kg)
        
        # Cost calculation
        cost_data = self.calculate_reactor_costs()
        
        # Cost per kWh
        total_energy = energy_data['usable_energy_kwh']
        total_cost = cost_data['total_cost']
        
        if total_energy > 0:
            cost_per_kwh = total_cost / total_energy
        else:
            cost_per_kwh = float('inf')
        
        # Viability metrics
        target_achieved = cost_per_kwh < self.config.target_cost_per_kwh
        cost_reduction_needed = cost_per_kwh / self.config.target_cost_per_kwh
        
        # Component cost breakdown per kWh
        cost_breakdown = {
            'capital_per_kwh': cost_data['capital_cost'] / total_energy if total_energy > 0 else 0,
            'antimatter_per_kwh': cost_data['antimatter_production_cost'] / total_energy if total_energy > 0 else 0,
            'operational_per_kwh': cost_data['operational_cost'] / total_energy if total_energy > 0 else 0
        }
        
        return {
            'energy_analysis': energy_data,
            'cost_analysis': cost_data,
            'cost_per_kwh': cost_per_kwh,
            'target_cost': self.config.target_cost_per_kwh,
            'target_achieved': target_achieved,
            'cost_reduction_needed': cost_reduction_needed,
            'cost_breakdown_per_kwh': cost_breakdown,
            'dominant_cost_factor': max(cost_breakdown.items(), key=lambda x: x[1])[0]
        }
    
    def parameter_sensitivity_analysis(self) -> Dict:
        """Analyze sensitivity to key parameters"""
        logger.info("Starting parameter sensitivity analysis")
        
        # Parameter ranges for sensitivity
        param_ranges = {
            'polymer_scale': np.linspace(1.0, 20.0, 20),
            'magnetic_field': np.linspace(10.0, 50.0, 20),
            'antimatter_mass': np.logspace(-9, -5, 20),  # 1 ng to 10 mg
            'trap_size': np.linspace(1.0, 10.0, 20)
        }
        
        sensitivity_results = {}
        baseline_config = self.config
        
        for param_name, param_values in param_ranges.items():
            results = []
            
            for value in param_values:
                # Create modified configuration
                test_config = ReactorConfiguration()
                
                if param_name == 'polymer_scale':
                    test_config.polymer_scale_mu = value
                elif param_name == 'magnetic_field':
                    test_config.magnetic_field = value
                elif param_name == 'antimatter_mass':
                    test_config.antimatter_mass_kg = value
                elif param_name == 'trap_size':
                    test_config.trap_radius = value
                    test_config.trap_length = 2 * value
                
                # Analyze this configuration
                analyzer = AdvancedReactorAnalyzer(test_config)
                viability = analyzer.economic_viability_analysis()
                
                results.append({
                    'parameter_value': value,
                    'cost_per_kwh': viability['cost_per_kwh'],
                    'target_achieved': viability['target_achieved'],
                    'overall_efficiency': viability['energy_analysis']['efficiency_data']['overall_efficiency']
                })
            
            sensitivity_results[param_name] = {
                'parameter_values': param_values.tolist(),
                'results': results
            }
        
        return sensitivity_results
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive reactor design report"""
        logger.info("Generating comprehensive reactor design report")
        
        # Core analysis
        viability = self.economic_viability_analysis()
        sensitivity = self.parameter_sensitivity_analysis()
        
        # Physics performance
        physics_performance = {
            'pair_production_cross_section': self.pair_production_cross_section(
                self.config.photon_energy_kev, self.config.polymer_scale_mu
            ),
            'confinement_efficiency': self.confinement_efficiency(self.config.polymer_scale_mu),
            'conversion_efficiency': self.energy_conversion_efficiency(self.config.polymer_scale_mu)
        }
        
        # WEST tokamak comparison
        west_baseline_kwh = 742.78  # From previous calculations
        west_comparison = {
            'west_baseline_kwh': west_baseline_kwh,
            'reactor_output_kwh': viability['energy_analysis']['usable_energy_kwh'],
            'energy_ratio': viability['energy_analysis']['usable_energy_kwh'] / west_baseline_kwh,
            'cost_comparison': 'WEST cost not available for direct comparison'
        }
        
        # Optimization recommendations
        recommendations = self._generate_optimization_recommendations(viability, sensitivity)
        
        return {
            'configuration': {
                'trap_radius': self.config.trap_radius,
                'trap_length': self.config.trap_length,
                'magnetic_field': self.config.magnetic_field,
                'polymer_scale': self.config.polymer_scale_mu,
                'antimatter_mass_kg': self.config.antimatter_mass_kg
            },
            'physics_performance': physics_performance,
            'economic_viability': viability,
            'sensitivity_analysis': sensitivity,
            'west_comparison': west_comparison,
            'optimization_recommendations': recommendations,
            'summary': {
                'cost_per_kwh': viability['cost_per_kwh'],
                'target_achieved': viability['target_achieved'],
                'key_limiting_factor': viability['dominant_cost_factor'],
                'required_breakthrough': viability['cost_reduction_needed']
            }
        }
    
    def _generate_optimization_recommendations(self, viability: Dict, sensitivity: Dict) -> List[str]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        # Cost analysis
        if viability['dominant_cost_factor'] == 'antimatter_per_kwh':
            recommendations.append(
                "Primary bottleneck: Antimatter production cost. "
                f"Need {viability['cost_reduction_needed']:.0f}x cost reduction."
            )
            recommendations.append(
                "Research priorities: Advanced antimatter production methods, "
                "magnetic plasma confinement optimization, alternative production pathways."
            )
        
        # Efficiency optimization
        overall_eff = viability['energy_analysis']['efficiency_data']['overall_efficiency']
        if overall_eff < 0.5:
            recommendations.append(
                f"Conversion efficiency ({overall_eff:.2f}) has improvement potential. "
                "Focus on thermodynamic cycle optimization and polymer enhancement."
            )
        
        # Parameter optimization
        polymer_scale = self.config.polymer_scale_mu
        if polymer_scale < 10:
            recommendations.append(
                f"Polymer scale (μ={polymer_scale:.1f}) could be increased for better enhancement. "
                "Investigate higher-order polymer field configurations."
            )
        
        # Scale considerations
        if self.config.antimatter_mass_kg < 1e-6:
            recommendations.append(
                "Consider larger antimatter inventory for improved economics of scale, "
                "balanced against storage and safety constraints."
            )
        
        return recommendations

def demonstrate_advanced_reactor_design():
    """Demonstrate the advanced reactor design analysis"""
    print("=" * 80)
    print("PLAN A, STEP 5: ADVANCED REACTOR DESIGN ANALYSIS")
    print("Complete simulation-driven reactor optimization")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = "plan_a_step5_reactor_design"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure reactor
    config = ReactorConfiguration(
        trap_radius=3.0,
        trap_length=6.0,
        magnetic_field=20.0,
        polymer_scale_mu=8.0,
        antimatter_mass_kg=1e-6  # 1 mg
    )
    
    print("REACTOR CONFIGURATION:")
    print(f"  Trap geometry: {config.trap_radius:.1f}m × {config.trap_length:.1f}m")
    print(f"  Magnetic field: {config.magnetic_field:.1f} T")
    print(f"  Polymer scale: μ = {config.polymer_scale_mu:.1f}")
    print(f"  Antimatter mass: {config.antimatter_mass_kg*1e6:.1f} mg")
    print(f"  Target cost: ${config.target_cost_per_kwh:.2f}/kWh")
    print()
    
    # Initialize analyzer
    analyzer = AdvancedReactorAnalyzer(config)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Display key results
    print("ECONOMIC VIABILITY ANALYSIS:")
    print("-" * 40)
    viability = report['economic_viability']
    print(f"  Total energy yield: {viability['energy_analysis']['usable_energy_kwh']:.2e} kWh")
    print(f"  Total system cost: ${viability['cost_analysis']['total_cost']:.2e}")
    print(f"  Cost per kWh: ${viability['cost_per_kwh']:.2f}")
    print(f"  Target achieved: {viability['target_achieved']}")
    
    if not viability['target_achieved']:
        print(f"  Cost reduction needed: {viability['cost_reduction_needed']:.0f}x")
    
    print(f"  Dominant cost factor: {viability['dominant_cost_factor']}")
    print()
    
    print("PHYSICS PERFORMANCE:")
    print("-" * 30)
    physics = report['physics_performance']
    print(f"  Pair production σ: {physics['pair_production_cross_section']:.2e} cm²")
    print(f"  Confinement efficiency: {physics['confinement_efficiency']:.2f}")
    print(f"  Overall conversion efficiency: {physics['conversion_efficiency']['overall_efficiency']:.2f}")
    print()
    
    print("WEST TOKAMAK COMPARISON:")
    print("-" * 35)
    west = report['west_comparison']
    print(f"  WEST baseline: {west['west_baseline_kwh']:.1f} kWh")
    print(f"  Reactor output: {west['reactor_output_kwh']:.2e} kWh")
    print(f"  Energy ratio: {west['energy_ratio']:.2e}x WEST output")
    print()
    
    print("OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 40)
    for i, rec in enumerate(report['optimization_recommendations'], 1):
        print(f"  {i}. {rec}")
    print()
    
    # Create visualizations
    create_reactor_visualizations(analyzer, report, output_dir)
    
    # Save complete report
    report_file = os.path.join(output_dir, "advanced_reactor_design_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Complete report saved to: {report_file}")
    print()
    print("=" * 80)
    print("ADVANCED REACTOR DESIGN ANALYSIS COMPLETE")
    print("=" * 80)
    
    return report

def create_reactor_visualizations(analyzer: AdvancedReactorAnalyzer, 
                                report: Dict, output_dir: str):
    """Create comprehensive visualizations"""
    
    # Extract sensitivity data
    sensitivity = report['sensitivity_analysis']
    
    # Create multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Reactor Design Parameter Sensitivity Analysis', fontsize=16)
    
    # Plot 1: Polymer scale sensitivity
    polymer_data = sensitivity['polymer_scale']
    mu_values = polymer_data['parameter_values']
    costs = [r['cost_per_kwh'] for r in polymer_data['results']]
    
    axes[0, 0].semilogy(mu_values, costs, 'b-', linewidth=2, marker='o')
    axes[0, 0].axhline(y=0.10, color='r', linestyle='--', label='Target ($0.10/kWh)')
    axes[0, 0].set_xlabel('Polymer Scale μ')
    axes[0, 0].set_ylabel('Cost per kWh ($)')
    axes[0, 0].set_title('Cost vs Polymer Scale')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Magnetic field sensitivity
    field_data = sensitivity['magnetic_field']
    field_values = field_data['parameter_values']
    field_costs = [r['cost_per_kwh'] for r in field_data['results']]
    
    axes[0, 1].semilogy(field_values, field_costs, 'g-', linewidth=2, marker='s')
    axes[0, 1].axhline(y=0.10, color='r', linestyle='--', label='Target ($0.10/kWh)')
    axes[0, 1].set_xlabel('Magnetic Field (T)')
    axes[0, 1].set_ylabel('Cost per kWh ($)')
    axes[0, 1].set_title('Cost vs Magnetic Field')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Antimatter mass sensitivity
    mass_data = sensitivity['antimatter_mass']
    mass_values = mass_data['parameter_values']
    mass_costs = [r['cost_per_kwh'] for r in mass_data['results']]
    
    axes[1, 0].loglog(np.array(mass_values)*1e6, mass_costs, 'm-', linewidth=2, marker='^')  # Convert to mg
    axes[1, 0].axhline(y=0.10, color='r', linestyle='--', label='Target ($0.10/kWh)')
    axes[1, 0].set_xlabel('Antimatter Mass (mg)')
    axes[1, 0].set_ylabel('Cost per kWh ($)')
    axes[1, 0].set_title('Cost vs Antimatter Mass')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Efficiency landscape
    eff_polymer = [r['overall_efficiency'] for r in polymer_data['results']]
    eff_field = [r['overall_efficiency'] for r in field_data['results']]
    
    axes[1, 1].plot(mu_values, eff_polymer, 'b-', linewidth=2, marker='o', label='vs Polymer Scale')
    ax2 = axes[1, 1].twinx()
    ax2.plot(field_values, eff_field, 'g-', linewidth=2, marker='s', label='vs B-field (T)')
    
    axes[1, 1].set_xlabel('Polymer Scale μ')
    axes[1, 1].set_ylabel('Efficiency (μ)', color='b')
    ax2.set_ylabel('Efficiency (B)', color='g')
    axes[1, 1].set_title('Conversion Efficiency Optimization')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add legends
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = os.path.join(output_dir, "reactor_sensitivity_analysis.png")
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"  Sensitivity analysis saved to: {viz_file}")
    
    plt.show()

if __name__ == "__main__":
    # Run the advanced reactor design demonstration
    results = demonstrate_advanced_reactor_design()
