"""
Plan A: Direct Mass-Energy Conversion Analysis
===========================================

This module implements the direct mass-energy conversion pathway using 
Einstein's E=mc² for polymer-induced energy extraction with economic analysis.

Based on WEST tokamak baseline (February 12, 2025):
- Confinement Time: 1,337 s (22 min 17 s) 
- Plasma Temperature: 50×10⁶ °C
- Heating Power: 2 MW RF injection
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WESTBaseline:
    """WEST tokamak world record parameters (February 12, 2025)"""
    confinement_time: float = 1337.0  # seconds
    plasma_temperature: float = 50e6  # Celsius
    heating_power: float = 2e6  # Watts
    date: str = "2025-02-12"
    location: str = "Cadarache, France"
    
    @property
    def plasma_temp_kelvin(self) -> float:
        """Convert plasma temperature to Kelvin"""
        return self.plasma_temperature + 273.15
    
    @property
    def confinement_time_hours(self) -> float:
        """Confinement time in hours"""
        return self.confinement_time / 3600.0

@dataclass
class DirectMassEnergyConverter:
    """Direct mass-energy conversion calculator using E=mc²"""
    
    # Physical constants
    c = constants.c  # Speed of light (m/s)
    
    def __init__(self, polymer_scale_mu: float = 1.0):
        """
        Initialize converter with polymer enhancement scale
        
        Args:
            polymer_scale_mu: Polymer scale parameter (dimensionless)
        """
        self.polymer_scale_mu = polymer_scale_mu
        
    def mass_energy_yield(self, mass_kg: float) -> Dict[str, float]:
        """
        Calculate energy yield from direct mass conversion
        
        Args:
            mass_kg: Mass to convert (kg)
            
        Returns:
            Dictionary with energy in various units
        """
        # Basic E = mc² calculation
        energy_joules = mass_kg * self.c**2
        
        # Convert to various useful units
        energy_kwh = energy_joules / 3.6e6  # Convert J to kWh
        energy_mwh = energy_kwh / 1000.0    # Convert kWh to MWh
        energy_twh = energy_mwh / 1e6       # Convert MWh to TWh
        
        # Apply polymer enhancement factor
        polymer_enhancement = self._polymer_enhancement_factor()
        
        return {
            'mass_kg': mass_kg,
            'energy_joules': energy_joules * polymer_enhancement,
            'energy_kwh': energy_kwh * polymer_enhancement,
            'energy_mwh': energy_mwh * polymer_enhancement,
            'energy_twh': energy_twh * polymer_enhancement,
            'polymer_enhancement_factor': polymer_enhancement,
            'polymer_scale_mu': self.polymer_scale_mu
        }
    
    def _polymer_enhancement_factor(self) -> float:
        """
        Calculate polymer-induced enhancement factor
        
        This is a phenomenological model that needs experimental calibration.
        Current form is based on theoretical polymer cross-section scaling.
        """
        # Polynomial enhancement model (to be refined with experimental data)
        base_enhancement = 1.0
        polymer_correction = 0.1 * np.log(1 + self.polymer_scale_mu)
        quantum_correction = 0.05 * self.polymer_scale_mu**0.5
        
        return base_enhancement + polymer_correction + quantum_correction
    
    def cost_analysis(self, mass_kg: float, production_cost_per_kg: float) -> Dict[str, float]:
        """
        Economic analysis of direct mass-energy conversion
        
        Args:
            mass_kg: Mass to convert (kg)
            production_cost_per_kg: Cost to produce/acquire mass ($/kg)
            
        Returns:
            Cost analysis dictionary
        """
        energy_data = self.mass_energy_yield(mass_kg)
        
        total_production_cost = mass_kg * production_cost_per_kg
        energy_kwh = energy_data['energy_kwh']
        
        cost_per_kwh = total_production_cost / energy_kwh if energy_kwh > 0 else float('inf')
        
        return {
            'total_production_cost_usd': total_production_cost,
            'energy_yield_kwh': energy_kwh,
            'cost_per_kwh_usd': cost_per_kwh,
            'polymer_scale_mu': self.polymer_scale_mu,
            'competitive_threshold_usd_per_kwh': 0.10,  # Current grid average
            'is_economically_viable': cost_per_kwh < 0.10
        }

class PolymerMassEnergyPipeline:
    """Complete pipeline for polymer-enhanced direct mass-energy conversion"""
    
    def __init__(self, west_baseline: WESTBaseline):
        self.west_baseline = west_baseline
        self.results = {}
        
    def run_polymer_scale_sweep(self, 
                               mu_range: Tuple[float, float] = (0.1, 10.0),
                               num_points: int = 50,
                               mass_kg: float = 0.001,  # 1 gram
                               production_cost_per_kg: float = 1000.0) -> Dict:
        """
        Sweep polymer scale parameter to find economic crossover
        
        Args:
            mu_range: Range of polymer scale values to test
            num_points: Number of points in sweep
            mass_kg: Mass for conversion (default 1 gram)
            production_cost_per_kg: Production cost ($/kg)
            
        Returns:
            Sweep results dictionary
        """
        mu_values = np.linspace(mu_range[0], mu_range[1], num_points)
        
        results = {
            'mu_values': mu_values.tolist(),
            'energy_yields_kwh': [],
            'cost_per_kwh_values': [],
            'enhancement_factors': [],
            'economic_viability': []
        }
        
        for mu in mu_values:
            converter = DirectMassEnergyConverter(polymer_scale_mu=mu)
            
            # Calculate energy yield
            energy_data = converter.mass_energy_yield(mass_kg)
            cost_data = converter.cost_analysis(mass_kg, production_cost_per_kg)
            
            results['energy_yields_kwh'].append(energy_data['energy_kwh'])
            results['cost_per_kwh_values'].append(cost_data['cost_per_kwh_usd'])
            results['enhancement_factors'].append(energy_data['polymer_enhancement_factor'])
            results['economic_viability'].append(cost_data['is_economically_viable'])
        
        # Find economic crossover point
        viable_indices = [i for i, viable in enumerate(results['economic_viability']) if viable]
        if viable_indices:
            crossover_index = viable_indices[0]
            results['economic_crossover_mu'] = mu_values[crossover_index]
            results['crossover_cost_per_kwh'] = results['cost_per_kwh_values'][crossover_index]
        else:
            results['economic_crossover_mu'] = None
            results['crossover_cost_per_kwh'] = None
        
        self.results['polymer_scale_sweep'] = results
        return results
    
    def benchmark_against_west(self) -> Dict:
        """
        Benchmark polymer-enhanced conversion against WEST baseline
        
        Returns:
            Comparison metrics
        """
        # WEST energy metrics
        west_energy_per_second = self.west_baseline.heating_power  # 2 MW
        west_total_energy_kwh = (west_energy_per_second * self.west_baseline.confinement_time) / 3.6e6
        
        # Polymer conversion for equivalent mass
        # Estimate equivalent mass from WEST plasma (rough approximation)
        estimated_plasma_mass = 1e-6  # kg (very rough estimate)
        
        optimal_mu = self.results.get('polymer_scale_sweep', {}).get('economic_crossover_mu', 1.0)
        if optimal_mu is None:
            optimal_mu = 1.0
            
        converter = DirectMassEnergyConverter(polymer_scale_mu=optimal_mu)
        polymer_energy_data = converter.mass_energy_yield(estimated_plasma_mass)
        
        return {
            'west_baseline': {
                'confinement_time_s': self.west_baseline.confinement_time,
                'heating_power_mw': self.west_baseline.heating_power / 1e6,
                'total_energy_kwh': west_total_energy_kwh,
                'plasma_temp_c': self.west_baseline.plasma_temperature
            },
            'polymer_conversion': {
                'optimal_mu': optimal_mu,
                'energy_yield_kwh': polymer_energy_data['energy_kwh'],
                'enhancement_factor': polymer_energy_data['polymer_enhancement_factor'],
                'energy_ratio_vs_west': polymer_energy_data['energy_kwh'] / west_total_energy_kwh if west_total_energy_kwh > 0 else float('inf')
            }
        }
    
    def generate_visualization(self, save_path: Optional[str] = None):
        """Generate visualization of polymer scale sweep results"""
        if 'polymer_scale_sweep' not in self.results:
            logger.error("No polymer scale sweep results available. Run run_polymer_scale_sweep() first.")
            return
        
        data = self.results['polymer_scale_sweep']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Energy yield vs polymer scale
        ax1.loglog(data['mu_values'], data['energy_yields_kwh'])
        ax1.set_xlabel('Polymer Scale μ')
        ax1.set_ylabel('Energy Yield (kWh)')
        ax1.set_title('Energy Yield vs Polymer Scale')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cost per kWh vs polymer scale
        ax2.semilogy(data['mu_values'], data['cost_per_kwh_values'])
        ax2.axhline(y=0.10, color='r', linestyle='--', label='Economic Threshold ($0.10/kWh)')
        ax2.set_xlabel('Polymer Scale μ')
        ax2.set_ylabel('Cost per kWh ($)')
        ax2.set_title('Cost Analysis vs Polymer Scale')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Enhancement factor
        ax3.plot(data['mu_values'], data['enhancement_factors'])
        ax3.set_xlabel('Polymer Scale μ')
        ax3.set_ylabel('Enhancement Factor')
        ax3.set_title('Polymer Enhancement Factor')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Economic viability map
        viable_mu = [mu for mu, viable in zip(data['mu_values'], data['economic_viability']) if viable]
        non_viable_mu = [mu for mu, viable in zip(data['mu_values'], data['economic_viability']) if not viable]
        
        if viable_mu:
            ax4.scatter(viable_mu, [1]*len(viable_mu), color='green', s=50, alpha=0.7, label='Economically Viable')
        if non_viable_mu:
            ax4.scatter(non_viable_mu, [0]*len(non_viable_mu), color='red', s=50, alpha=0.7, label='Not Viable')
        
        ax4.set_xlabel('Polymer Scale μ')
        ax4.set_ylabel('Economic Viability')
        ax4.set_title('Economic Viability Map')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, filepath: str):
        """Save all results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load results from JSON file"""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        logger.info(f"Results loaded from {filepath}")

def demonstrate_plan_a():
    """Demonstration of Plan A: Direct Mass-Energy Conversion"""
    print("=" * 60)
    print("Plan A: Direct Mass-Energy Conversion Demonstration")
    print("=" * 60)
    
    # Initialize WEST baseline
    west = WESTBaseline()
    print(f"WEST Baseline ({west.date}):")
    print(f"  Confinement Time: {west.confinement_time:.0f} s ({west.confinement_time_hours:.2f} h)")
    print(f"  Plasma Temperature: {west.plasma_temperature/1e6:.0f}×10⁶ °C")
    print(f"  Heating Power: {west.heating_power/1e6:.1f} MW")
    print()
    
    # Example calculation for 1 gram
    print("Direct Mass-Energy Conversion (1 gram):")
    converter = DirectMassEnergyConverter(polymer_scale_mu=1.0)
    energy_data = converter.mass_energy_yield(0.001)  # 1 gram
    
    print(f"  Basic E=mc²: {energy_data['energy_kwh']/energy_data['polymer_enhancement_factor']:.2e} kWh")
    print(f"  With polymer enhancement (μ=1.0): {energy_data['energy_kwh']:.2e} kWh")
    print(f"  Enhancement factor: {energy_data['polymer_enhancement_factor']:.3f}")
    print()
    
    # Economic analysis
    cost_data = converter.cost_analysis(0.001, 1000.0)  # $1000/kg production cost
    print("Economic Analysis:")
    print(f"  Cost per kWh: ${cost_data['cost_per_kwh_usd']:.2e}")
    print(f"  Economically viable: {cost_data['is_economically_viable']}")
    print()
    
    # Run pipeline
    pipeline = PolymerMassEnergyPipeline(west)
    sweep_results = pipeline.run_polymer_scale_sweep()
    
    if sweep_results['economic_crossover_mu'] is not None:
        print(f"Economic crossover at μ = {sweep_results['economic_crossover_mu']:.3f}")
        print(f"Crossover cost: ${sweep_results['crossover_cost_per_kwh']:.3f}/kWh")
    else:
        print("No economic crossover found in tested range")
    
    # Benchmark against WEST
    benchmark = pipeline.benchmark_against_west()
    print("\nBenchmark vs WEST:")
    print(f"  WEST total energy: {benchmark['west_baseline']['total_energy_kwh']:.3f} kWh")
    print(f"  Polymer conversion energy: {benchmark['polymer_conversion']['energy_yield_kwh']:.2e} kWh")
    print(f"  Energy ratio: {benchmark['polymer_conversion']['energy_ratio_vs_west']:.2e}×")

if __name__ == "__main__":
    demonstrate_plan_a()
