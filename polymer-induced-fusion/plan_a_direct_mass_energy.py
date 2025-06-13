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
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert the results
        converted_results = convert_numpy_types(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2)
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load results from JSON file"""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        logger.info(f"Results loaded from {filepath}")

def demonstrate_plan_a():
    """Demonstration of Plan A: Direct Mass-Energy Conversion"""
    print("=" * 80)
    print("Plan A: Direct Mass-Energy Conversion Demonstration")
    print("Includes Antimatter Production Cost Analysis (NASA Data)")
    print("=" * 80)
    
    # Initialize WEST baseline
    west = WESTBaseline()
    print(f"WEST Baseline ({west.date}):")
    print(f"  Confinement Time: {west.confinement_time:.0f} s ({west.confinement_time_hours:.2f} h)")
    print(f"  Plasma Temperature: {west.plasma_temperature/1e6:.0f}×10⁶ °C")
    print(f"  Heating Power: {west.heating_power/1e6:.1f} MW")
    print()
    
    # Example calculation for 1 gram (regular matter)
    print("Direct Mass-Energy Conversion (1 gram regular matter):")
    converter = DirectMassEnergyConverter(polymer_scale_mu=1.0)
    energy_data = converter.mass_energy_yield(0.001)  # 1 gram
    
    print(f"  Basic E=mc²: {energy_data['energy_kwh']/energy_data['polymer_enhancement_factor']:.2e} kWh")
    print(f"  With polymer enhancement (μ=1.0): {energy_data['energy_kwh']:.2e} kWh")
    print(f"  Enhancement factor: {energy_data['polymer_enhancement_factor']:.3f}")
    print()
    
    # Economic analysis for regular matter
    cost_data = converter.cost_analysis(0.001, 1000.0)  # $1000/kg production cost
    print("Economic Analysis (Regular Matter):")
    print(f"  Cost per kWh: ${cost_data['cost_per_kwh_usd']:.2e}")
    print(f"  Economically viable: {cost_data['is_economically_viable']}")
    print()
    
    # NEW: Antimatter Analysis (Step 2)
    print("=" * 60)
    print("ANTIMATTER PRODUCTION COST ANALYSIS (NASA DATA)")
    print("=" * 60)
    
    # Initialize antimatter converter
    antimatter_converter = AntimatterEnergyConverter(polymer_scale_mu=1.0)
    
    # Analysis for 1 nanogram of antimatter (realistic current scale)
    antimatter_mass = 1e-12  # 1 picogram
    print(f"Antimatter Analysis ({antimatter_mass*1e15:.1f} femtograms):")
    
    antimatter_cost_data = antimatter_converter.antimatter_production_cost_analysis(antimatter_mass)
    print(f"  Current production cost: ${antimatter_cost_data['current_production_cost_usd']:.2e}")
    print(f"  Polymer-enhanced cost: ${antimatter_cost_data['polymer_enhanced_cost_usd']:.2e}")
    print(f"  Cost reduction factor: {antimatter_cost_data['cost_reduction_factor']:.1f}×")
    print(f"  Energy yield: {antimatter_cost_data['energy_yield_kwh']:.2e} kWh")
    print(f"  Current cost/kWh: ${antimatter_cost_data['current_cost_per_kwh_usd']:.2e}")
    print(f"  Polymer cost/kWh: ${antimatter_cost_data['polymer_cost_per_kwh_usd']:.2e}")
    print()
    
    # Production efficiency requirements
    efficiency_data = antimatter_converter.production_efficiency_analysis()
    print("Production Efficiency Requirements:")
    print(f"  Current efficiency: {efficiency_data['current_efficiency']:.2e}")
    print(f"  Required for grid competitive: {efficiency_data['required_efficiency_for_grid_competitive']:.2e}")
    print(f"  Required improvement: {efficiency_data['required_improvement_factor']:.2e}×")
    print(f"  Feasible with current physics: {efficiency_data['feasible_with_current_physics']}")
    print()
    
    # NASA baseline comparison
    print("NASA Baseline (NTRS 19990080056):")
    print(f"  Antiproton cost: ${antimatter_converter.antimatter_data.antiproton_cost_per_gram:.2e}/gram")
    print(f"  Facility type: {antimatter_converter.antimatter_data.facility_type}")
    print()
    
    # Run comprehensive antimatter analysis
    antimatter_pipeline = AntimatterProductionPipeline(west)
    benchmark_data = antimatter_pipeline.benchmark_against_contemporary_facilities()
    
    print("Contemporary Facility Benchmark:")
    nasa_data = benchmark_data['nasa_baseline']
    test_data = benchmark_data['test_case']
    west_comp = benchmark_data['west_comparison']
    
    print(f"  NASA source: {nasa_data['source']}")
    print(f"  Test mass: {test_data['antimatter_mass_ng']:.3f} ng")
    print(f"  vs WEST energy ratio: {west_comp['energy_ratio']:.2e}×")
    print()
    
    # Regular matter pipeline (existing code)
    print("=" * 60)
    print("REGULAR MATTER ANALYSIS")
    print("=" * 60)
    
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
    
    print("\n" + "=" * 80)
    print("SUMMARY: Antimatter vs Regular Matter")
    print("=" * 80)
    print(f"Regular matter (1g) energy: {energy_data['energy_kwh']:.2e} kWh")
    print(f"Antimatter (1fg) energy: {antimatter_cost_data['energy_yield_kwh']:.2e} kWh")
    print(f"NASA antimatter cost: ${nasa_data['cost_per_gram_usd']:.2e}/gram")
    print("Recommendation: Focus on polymer-enhanced production efficiency")

if __name__ == "__main__":
    demonstrate_plan_a()

@dataclass
class AntimatterProductionData:
    """Antimatter production cost data from NASA/contemporary facilities"""
    
    # NASA data: antiprotons at roughly $62.5 trillion per gram
    antiproton_cost_per_gram: float = 62.5e12  # USD per gram
    source: str = "NASA NTRS 19990080056"
    facility_type: str = "Contemporary particle accelerators"
    
    # Theoretical improvements with polymer enhancement
    polymer_cost_reduction_factor: float = 0.1  # 10× cost reduction potential
    
    # Production efficiency factors
    current_efficiency: float = 1e-9  # Very low current efficiency
    theoretical_max_efficiency: float = 0.1  # 10% theoretical maximum
    
    @property
    def polymer_enhanced_cost_per_gram(self) -> float:
        """Polymer-enhanced antimatter production cost"""
        return self.antiproton_cost_per_gram * self.polymer_cost_reduction_factor
    
    @property
    def efficiency_improvement_potential(self) -> float:
        """Potential efficiency improvement factor"""
        return self.theoretical_max_efficiency / self.current_efficiency

@dataclass
class AntimatterEnergyConverter:
    """Antimatter-based energy conversion with polymer enhancement"""
    
    def __init__(self, polymer_scale_mu: float = 1.0):
        self.polymer_scale_mu = polymer_scale_mu
        self.antimatter_data = AntimatterProductionData()
        self.c = constants.c  # Speed of light
        
    def antimatter_annihilation_yield(self, antimatter_mass_kg: float) -> Dict[str, float]:
        """
        Calculate energy yield from antimatter-matter annihilation
        
        For antimatter annihilation: E = 2mc² (both matter and antimatter convert)
        
        Args:
            antimatter_mass_kg: Mass of antimatter (kg)
            
        Returns:
            Energy yield data
        """
        # Complete annihilation: antimatter + equal mass of matter → pure energy
        total_annihilating_mass = 2 * antimatter_mass_kg  # antimatter + matter
        
        # Basic annihilation energy
        energy_joules = total_annihilating_mass * self.c**2
        energy_kwh = energy_joules / 3.6e6
        
        # Apply polymer enhancement
        polymer_enhancement = self._polymer_annihilation_enhancement()
        
        return {
            'antimatter_mass_kg': antimatter_mass_kg,
            'total_annihilating_mass_kg': total_annihilating_mass,
            'energy_joules': energy_joules * polymer_enhancement,
            'energy_kwh': energy_kwh * polymer_enhancement,
            'polymer_enhancement_factor': polymer_enhancement,
            'theoretical_efficiency': 1.0,  # 100% mass-energy conversion
            'polymer_scale_mu': self.polymer_scale_mu
        }
    
    def _polymer_annihilation_enhancement(self) -> float:
        """
        Polymer enhancement factor for antimatter annihilation
        
        Polymers could potentially:
        1. Improve confinement efficiency
        2. Enhance energy capture
        3. Reduce losses during annihilation
        """
        base_efficiency = 1.0
        
        # Confinement enhancement (reduces antimatter losses)
        confinement_factor = 0.1 * np.log(1 + self.polymer_scale_mu)
        
        # Energy capture enhancement (better conversion to usable energy)
        capture_factor = 0.05 * self.polymer_scale_mu**0.5
        
        # Quantum efficiency improvements
        quantum_factor = 0.02 * self.polymer_scale_mu**0.3
        
        return base_efficiency + confinement_factor + capture_factor + quantum_factor
    
    def antimatter_production_cost_analysis(self, antimatter_mass_kg: float) -> Dict[str, float]:
        """
        Comprehensive cost analysis for antimatter production and energy conversion
        
        Args:
            antimatter_mass_kg: Required antimatter mass (kg)
            
        Returns:
            Cost analysis with polymer-enhanced scenarios
        """
        mass_grams = antimatter_mass_kg * 1000
        
        # Current production costs (NASA data)
        current_production_cost = mass_grams * self.antimatter_data.antiproton_cost_per_gram
        
        # Polymer-enhanced production costs
        polymer_enhanced_cost = mass_grams * self.antimatter_data.polymer_enhanced_cost_per_gram
        
        # Energy yield analysis
        energy_data = self.antimatter_annihilation_yield(antimatter_mass_kg)
        energy_kwh = energy_data['energy_kwh']
        
        # Cost per kWh calculations
        current_cost_per_kwh = current_production_cost / energy_kwh if energy_kwh > 0 else float('inf')
        polymer_cost_per_kwh = polymer_enhanced_cost / energy_kwh if energy_kwh > 0 else float('inf')
        
        # Economic viability thresholds
        grid_threshold = 0.10  # $0.10/kWh
        premium_threshold = 1.00  # $1.00/kWh for premium applications
        
        return {
            'antimatter_mass_grams': mass_grams,
            'current_production_cost_usd': current_production_cost,
            'polymer_enhanced_cost_usd': polymer_enhanced_cost,
            'cost_reduction_factor': current_production_cost / polymer_enhanced_cost if polymer_enhanced_cost > 0 else float('inf'),
            'energy_yield_kwh': energy_kwh,
            'current_cost_per_kwh_usd': current_cost_per_kwh,
            'polymer_cost_per_kwh_usd': polymer_cost_per_kwh,
            'grid_competitive_current': current_cost_per_kwh < grid_threshold,
            'grid_competitive_polymer': polymer_cost_per_kwh < grid_threshold,
            'premium_viable_current': current_cost_per_kwh < premium_threshold,
            'premium_viable_polymer': polymer_cost_per_kwh < premium_threshold,
            'polymer_scale_mu': self.polymer_scale_mu,
            'nasa_source': self.antimatter_data.source
        }
    
    def production_efficiency_analysis(self) -> Dict[str, float]:
        """
        Analyze production efficiency improvements needed for economic viability
        
        Returns:
            Efficiency analysis data
        """
        current_eff = self.antimatter_data.current_efficiency
        max_eff = self.antimatter_data.theoretical_max_efficiency
        improvement_potential = self.antimatter_data.efficiency_improvement_potential
        
        # Calculate required efficiency for grid competitiveness
        # Assuming base production cost scales inversely with efficiency
        base_cost_per_gram = self.antimatter_data.antiproton_cost_per_gram
        
        # For $0.10/kWh competitiveness with 1 gram antimatter
        target_cost_per_kwh = 0.10
        energy_per_gram_kwh = self.antimatter_annihilation_yield(0.001)['energy_kwh']
        required_cost_per_gram = target_cost_per_kwh * energy_per_gram_kwh
        
        required_efficiency_improvement = base_cost_per_gram / required_cost_per_gram
        required_efficiency = current_eff * required_efficiency_improvement
        
        return {
            'current_efficiency': current_eff,
            'theoretical_max_efficiency': max_eff,
            'improvement_potential_factor': improvement_potential,
            'required_efficiency_for_grid_competitive': required_efficiency,
            'required_improvement_factor': required_efficiency_improvement,
            'feasible_with_current_physics': required_efficiency <= max_eff,
            'polymer_enhancement_needed': required_efficiency_improvement / improvement_potential
        }

class AntimatterProductionPipeline:
    """Pipeline for antimatter production cost and viability analysis"""
    
    def __init__(self, west_baseline: WESTBaseline):
        self.west_baseline = west_baseline
        self.results = {}
        
    def run_antimatter_cost_sweep(self,
                                 mass_range: Tuple[float, float] = (1e-6, 1e-3),  # 1 microgram to 1 milligram
                                 mu_range: Tuple[float, float] = (0.1, 10.0),
                                 num_mass_points: int = 20,
                                 num_mu_points: int = 30) -> Dict:
        """
        Comprehensive sweep of antimatter mass and polymer scale parameters
        
        Args:
            mass_range: Range of antimatter masses in kg
            mu_range: Range of polymer scale parameters
            num_mass_points: Number of mass points to analyze
            num_mu_points: Number of polymer scale points
            
        Returns:
            Comprehensive sweep results
        """
        mass_values = np.logspace(np.log10(mass_range[0]), np.log10(mass_range[1]), num_mass_points)
        mu_values = np.linspace(mu_range[0], mu_range[1], num_mu_points)
        
        results = {
            'mass_values_kg': mass_values.tolist(),
            'mu_values': mu_values.tolist(),
            'cost_matrix_current': [],
            'cost_matrix_polymer': [],
            'energy_matrix_kwh': [],
            'viable_combinations': []
        }
        
        for mass_kg in mass_values:
            mass_costs_current = []
            mass_costs_polymer = []
            mass_energies = []
            
            for mu in mu_values:
                converter = AntimatterEnergyConverter(polymer_scale_mu=mu)
                cost_data = converter.antimatter_production_cost_analysis(mass_kg)
                
                mass_costs_current.append(cost_data['current_cost_per_kwh_usd'])
                mass_costs_polymer.append(cost_data['polymer_cost_per_kwh_usd'])
                mass_energies.append(cost_data['energy_yield_kwh'])
                
                # Track viable combinations
                if cost_data['grid_competitive_polymer']:
                    results['viable_combinations'].append({
                        'mass_kg': mass_kg,
                        'mu': mu,
                        'cost_per_kwh': cost_data['polymer_cost_per_kwh_usd'],
                        'energy_kwh': cost_data['energy_yield_kwh']
                    })
            
            results['cost_matrix_current'].append(mass_costs_current)
            results['cost_matrix_polymer'].append(mass_costs_polymer)
            results['energy_matrix_kwh'].append(mass_energies)
        
        self.results['antimatter_sweep'] = results
        return results
    
    def benchmark_against_contemporary_facilities(self) -> Dict:
        """
        Benchmark against contemporary antimatter production facilities
        
        Returns:
            Benchmark comparison data
        """
        # Example: 1 microgram of antimatter
        test_mass = 1e-9  # 1 nanogram (more realistic for current facilities)
        
        converter = AntimatterEnergyConverter(polymer_scale_mu=1.0)
        cost_data = converter.antimatter_production_cost_analysis(test_mass)
        efficiency_data = converter.production_efficiency_analysis()
        
        # Compare with WEST energy output
        west_energy_kwh = (self.west_baseline.heating_power * self.west_baseline.confinement_time) / 3.6e6
        antimatter_energy_kwh = cost_data['energy_yield_kwh']
        
        return {
            'nasa_baseline': {
                'cost_per_gram_usd': 62.5e12,
                'source': 'NASA NTRS 19990080056',
                'facility_type': 'Contemporary particle accelerators'
            },
            'test_case': {
                'antimatter_mass_ng': test_mass * 1e12,  # nanograms
                'current_cost_usd': cost_data['current_production_cost_usd'],
                'polymer_enhanced_cost_usd': cost_data['polymer_enhanced_cost_usd'],
                'energy_yield_kwh': antimatter_energy_kwh,
                'current_cost_per_kwh': cost_data['current_cost_per_kwh_usd'],
                'polymer_cost_per_kwh': cost_data['polymer_cost_per_kwh_usd']
            },
            'west_comparison': {
                'west_energy_kwh': west_energy_kwh,
                'antimatter_energy_kwh': antimatter_energy_kwh,
                'energy_ratio': antimatter_energy_kwh / west_energy_kwh if west_energy_kwh > 0 else float('inf')
            },
            'efficiency_requirements': efficiency_data
        }
    
    def generate_antimatter_visualization(self, save_path: Optional[str] = None):
        """Generate comprehensive antimatter analysis visualization"""
        if 'antimatter_sweep' not in self.results:
            logger.error("No antimatter sweep results available.")
            return
        
        data = self.results['antimatter_sweep']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Convert to numpy arrays for plotting
        mass_values = np.array(data['mass_values_kg'])
        mu_values = np.array(data['mu_values'])
        cost_matrix_polymer = np.array(data['cost_matrix_polymer'])
        energy_matrix = np.array(data['energy_matrix_kwh'])
        
        # Plot 1: Cost heatmap (polymer-enhanced)
        # Clip extreme values for visualization
        cost_plot = np.clip(cost_matrix_polymer, 0.01, 100)
        im1 = ax1.imshow(np.log10(cost_plot), aspect='auto', origin='lower', cmap='viridis')
        ax1.set_xlabel('Polymer Scale μ Index')
        ax1.set_ylabel('Antimatter Mass Index')
        ax1.set_title('Polymer-Enhanced Cost per kWh (log₁₀ scale)')
        plt.colorbar(im1, ax=ax1, label='log₁₀(Cost $/kWh)')
        
        # Plot 2: Energy yield heatmap
        im2 = ax2.imshow(np.log10(energy_matrix), aspect='auto', origin='lower', cmap='plasma')
        ax2.set_xlabel('Polymer Scale μ Index')
        ax2.set_ylabel('Antimatter Mass Index')
        ax2.set_title('Energy Yield (log₁₀ kWh)')
        plt.colorbar(im2, ax=ax2, label='log₁₀(Energy kWh)')
        
        # Plot 3: Viable combinations
        if data['viable_combinations']:
            viable_masses = [v['mass_kg'] for v in data['viable_combinations']]
            viable_mus = [v['mu'] for v in data['viable_combinations']]
            viable_costs = [v['cost_per_kwh'] for v in data['viable_combinations']]
            
            scatter = ax3.scatter(viable_mus, viable_masses, c=viable_costs, 
                                cmap='cool', s=50, alpha=0.7)
            ax3.set_xlabel('Polymer Scale μ')
            ax3.set_ylabel('Antimatter Mass (kg)')
            ax3.set_yscale('log')
            ax3.set_title('Economically Viable Combinations')
            plt.colorbar(scatter, ax=ax3, label='Cost $/kWh')
        else:
            ax3.text(0.5, 0.5, 'No Economically\nViable Combinations\nFound', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Economic Viability')
        
        # Plot 4: Production cost breakdown
        example_masses = mass_values[::5]  # Sample every 5th mass
        current_costs = []
        polymer_costs = []
        
        for mass in example_masses:
            converter = AntimatterEnergyConverter(polymer_scale_mu=1.0)
            cost_data = converter.antimatter_production_cost_analysis(mass)
            current_costs.append(cost_data['current_cost_per_kwh_usd'])
            polymer_costs.append(cost_data['polymer_cost_per_kwh_usd'])
        
        ax4.loglog(example_masses * 1e9, current_costs, 'r-', linewidth=2, label='Current Technology')
        ax4.loglog(example_masses * 1e9, polymer_costs, 'b-', linewidth=2, label='Polymer Enhanced')
        ax4.axhline(y=0.10, color='g', linestyle='--', alpha=0.7, label='Grid Competitive ($0.10/kWh)')
        ax4.axhline(y=1.00, color='orange', linestyle='--', alpha=0.7, label='Premium Market ($1.00/kWh)')
        ax4.set_xlabel('Antimatter Mass (nanograms)')
        ax4.set_ylabel('Cost per kWh ($)')
        ax4.set_title('Production Cost vs Mass')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Antimatter Production Cost Analysis\n(Based on NASA $62.5T/gram baseline)', 
                     fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Antimatter visualization saved to {save_path}")
        else:
            plt.show()
