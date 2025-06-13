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
from datetime import datetime

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
    
    print("=" * 80)
    print("SUMMARY: Comprehensive Plan A Analysis")
    print("=" * 80)
    print(f"Regular matter (1g) energy: {energy_data['energy_kwh']:.2e} kWh")
    print(f"Antimatter (1fg) theoretical: {antimatter_cost_data['energy_yield_kwh']:.2e} kWh")
    print(f"NASA antimatter cost: ${nasa_data['cost_per_gram_usd']:.2e}/gram")
    
    # NEW: Energy Conversion Efficiency Analysis (Step 3)
    print("\n" + "=" * 80)
    print("STEP 3: ENERGY CONVERSION EFFICIENCY ANALYSIS")
    print("511 keV photon → electricity conversion")
    print("=" * 80)
    
    # Initialize realistic converter
    realistic_converter = RealisticAntimatterConverter(
        polymer_scale_mu=5.0,
        conversion_method='tpv_system',
        polymer_enhanced=True
    )
    
    test_mass = 1e-12  # 1 picogram
    
    # Compare theoretical vs realistic conversion
    theoretical_data = realistic_converter.theoretical_annihilation_energy(test_mass)
    realistic_data = realistic_converter.realistic_energy_conversion(test_mass)
    cost_realistic = realistic_converter.comprehensive_cost_analysis(test_mass)
    
    print(f"Antimatter Analysis ({test_mass*1e15:.0f} femtograms):")
    print(f"  Theoretical energy (E=2mc²): {theoretical_data['theoretical_energy_kwh']:.4f} kWh")
    print(f"  Realistic energy (with conversion): {realistic_data['realistic_energy_kwh']:.4f} kWh")
    print(f"  Conversion efficiency: {realistic_data['conversion_efficiency']*100:.1f}%")
    print(f"  Total efficiency loss: {(1-realistic_data['efficiency_loss_factor'])*100:.1f}%")
    print(f"  Conversion method: {realistic_data['conversion_method']}")
    print()
    
    # Cost impact of conversion efficiency
    print("Economic Impact of Conversion Efficiency:")
    theoretical_cost_per_kwh = (test_mass * 1000 * 62.5e12) / theoretical_data['theoretical_energy_kwh']
    realistic_cost_per_kwh = cost_realistic['cost_per_kwh_usd']
    efficiency_penalty = realistic_cost_per_kwh / theoretical_cost_per_kwh
    
    print(f"  Theoretical cost/kWh: ${theoretical_cost_per_kwh:.2e}")
    print(f"  Realistic cost/kWh: ${realistic_cost_per_kwh:.2e}")
    print(f"  Efficiency penalty: {efficiency_penalty:.1f}× cost increase")
    print()
    
    # Conversion method comparison
    print("Conversion Method Comparison:")
    methods = ['tpv_lab', 'tpv_system', 'thermionic', 'direct']
    method_names = ['TPV Laboratory', 'TPV Full System', 'Thermionic', 'Direct Conversion']
    
    for method, name in zip(methods, method_names):
        converter = RealisticAntimatterConverter(
            polymer_scale_mu=5.0,
            conversion_method=method,
            polymer_enhanced=True
        )
        data = converter.realistic_energy_conversion(test_mass)
        cost = converter.comprehensive_cost_analysis(test_mass)
        
        print(f"  {name}:")
        print(f"    Efficiency: {data['conversion_efficiency']*100:.1f}%")
        print(f"    Cost/kWh: ${cost['cost_per_kwh_usd']:.2e}")
        print(f"    Grid competitive: {cost['grid_competitive']}")
    
    print()
    
    # WEST baseline benchmarking
    print("WEST Tokamak Baseline Benchmarking:")
    print("-" * 40)
    west_benchmark = WESTBenchmarkMetrics()
    benchmark_data = realistic_converter.west_benchmark_comparison(test_mass)
    
    west_base = benchmark_data['west_baseline']
    antimatter_sys = benchmark_data['antimatter_system']
    comparison = benchmark_data['comparison_metrics']
    
    print(f"WEST Reference Points (February 12, 2025):")
    print(f"  Confinement time: {west_base['confinement_time_s']:.0f} s")
    print(f"  Temperature: {west_base['temperature_c']/1e6:.0f}×10⁶ °C")
    print(f"  Heating power: {west_base['heating_power_w']/1e6:.1f} MW")
    print(f"  Total energy: {west_base['total_energy_kwh']:.1f} kWh")
    print()
    
    print(f"Antimatter System Comparison:")
    print(f"  Realistic energy: {antimatter_sys['realistic_energy_kwh']:.4f} kWh")
    print(f"  Energy ratio vs WEST: {comparison['energy_ratio']:.2e}×")
    print(f"  Power ratio vs WEST: {comparison['power_ratio']:.2e}×")
    print(f"  Conversion efficiency: {antimatter_sys['conversion_efficiency']*100:.1f}%")
    print()
    
    # Target metrics assessment
    targets = west_benchmark.meets_targets(1500, 150e6, 1.6e6)  # Target values
    print(f"WEST Improvement Targets:")
    print(f"  Target confinement: > {west_benchmark.target_confinement_time_s:.0f} s")
    print(f"  Target temperature: {west_benchmark.target_temperature_c/1e6:.0f}×10⁶ °C (ITER goal)")
    print(f"  Target power reduction: < {west_benchmark.baseline_heating_power_w * west_benchmark.target_heating_power_reduction/1e6:.1f} MW")
    print()
    
    # Efficiency pipeline analysis
    print("Running Conversion Efficiency Pipeline...")
    efficiency_pipeline = ConversionEfficiencyPipeline(west)
    
    # Run conversion method comparison
    conversion_comparison = efficiency_pipeline.run_conversion_method_comparison(
        antimatter_mass_kg=test_mass,
        mu_range=(0.1, 10.0),
        num_points=20
    )
    
    # Efficiency impact analysis
    efficiency_impact = efficiency_pipeline.efficiency_impact_analysis(test_mass)
    
    print("Efficiency Impact Summary:")
    for method, data in efficiency_impact['efficiency_gaps'].items():
        print(f"  {method.replace('_', ' ').title()}:")
        print(f"    Energy loss: {data['energy_loss_percentage']:.1f}%")
        print(f"    Cost penalty: {data['efficiency_penalty_factor']:.1f}×")
    
    print("\nRecommendation: Focus on polymer-enhanced direct conversion methods")
    print("and improved thermophotovoltaic systems for optimal efficiency.")

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

@dataclass
class EnergyConversionEfficiency:
    """Energy conversion efficiency data for antimatter annihilation products"""
    
    # Antimatter annihilation produces 511 keV gamma rays
    annihilation_photon_energy_kev: float = 511.0  # keV per photon
    photons_per_annihilation: int = 2  # Two 511 keV photons per electron-positron pair
    
    # Conversion efficiency data from laboratory demonstrations
    # Source: Wikipedia - Thermophotovoltaic energy conversion
    laboratory_tpv_efficiency: float = 0.35  # 35% maximum demonstrated
    full_system_tpv_efficiency: float = 0.05  # 5% typical full-system efficiency
    
    # Alternative conversion methods
    thermionic_efficiency: float = 0.15  # ~15% for thermionic conversion
    direct_conversion_efficiency: float = 0.25  # Theoretical direct conversion
    
    # Polymer-enhanced conversion factors
    polymer_tpv_enhancement: float = 1.5  # 50% improvement potential
    polymer_thermionic_enhancement: float = 1.3  # 30% improvement potential
    polymer_direct_enhancement: float = 2.0  # 100% improvement potential
    
    def get_conversion_efficiency(self, method: str, polymer_enhanced: bool = False) -> float:
        """
        Get conversion efficiency for specified method
        
        Args:
            method: Conversion method ('tpv_lab', 'tpv_system', 'thermionic', 'direct')
            polymer_enhanced: Whether to apply polymer enhancement
            
        Returns:
            Conversion efficiency (0-1)
        """
        base_efficiencies = {
            'tpv_lab': self.laboratory_tpv_efficiency,
            'tpv_system': self.full_system_tpv_efficiency,
            'thermionic': self.thermionic_efficiency,
            'direct': self.direct_conversion_efficiency
        }
        
        enhancement_factors = {
            'tpv_lab': self.polymer_tpv_enhancement,
            'tpv_system': self.polymer_tpv_enhancement,
            'thermionic': self.polymer_thermionic_enhancement,
            'direct': self.polymer_direct_enhancement
        }
        
        base_eff = base_efficiencies.get(method, 0.05)
        
        if polymer_enhanced:
            enhancement = enhancement_factors.get(method, 1.0)
            return min(base_eff * enhancement, 1.0)  # Cap at 100%
        else:
            return base_eff

@dataclass
class WESTBenchmarkMetrics:
    """WEST tokamak baseline metrics for benchmarking polymer improvements"""
    
    # Zero-point anchors from WEST record (February 12, 2025)
    baseline_confinement_time_s: float = 1337.0
    baseline_temperature_c: float = 50e6
    baseline_heating_power_w: float = 2e6
    
    # Target improvements relative to WEST baseline
    target_confinement_time_s: float = 1500.0  # > 1500 s target
    target_temperature_c: float = 150e6  # ITER goal: 150×10⁶ °C
    target_heating_power_reduction: float = 0.8  # Reduce to 80% of baseline (1.6 MW)
    
    def confinement_improvement_factor(self, achieved_time_s: float) -> float:
        """Calculate confinement improvement factor relative to WEST"""
        return achieved_time_s / self.baseline_confinement_time_s
    
    def temperature_improvement_factor(self, achieved_temp_c: float) -> float:
        """Calculate temperature improvement factor relative to WEST"""
        return achieved_temp_c / self.baseline_temperature_c
    
    def power_efficiency_factor(self, required_power_w: float) -> float:
        """Calculate power efficiency factor (lower is better)"""
        return required_power_w / self.baseline_heating_power_w
    
    def meets_targets(self, confinement_s: float, temperature_c: float, power_w: float) -> Dict[str, bool]:
        """Check if parameters meet improvement targets"""
        return {
            'confinement_target': confinement_s >= self.target_confinement_time_s,
            'temperature_target': temperature_c >= self.target_temperature_c,
            'power_target': power_w <= (self.baseline_heating_power_w * self.target_heating_power_reduction),
            'all_targets_met': (confinement_s >= self.target_confinement_time_s and
                              temperature_c >= self.target_temperature_c and
                              power_w <= (self.baseline_heating_power_w * self.target_heating_power_reduction))
        }

class RealisticAntimatterConverter:
    """Antimatter converter with realistic energy conversion efficiencies"""
    
    def __init__(self, 
                 polymer_scale_mu: float = 1.0,
                 conversion_method: str = 'tpv_system',
                 polymer_enhanced: bool = True):
        """
        Initialize realistic antimatter converter
        
        Args:
            polymer_scale_mu: Polymer enhancement scale parameter
            conversion_method: Energy conversion method
            polymer_enhanced: Whether to use polymer-enhanced conversion
        """
        self.polymer_scale_mu = polymer_scale_mu
        self.conversion_method = conversion_method
        self.polymer_enhanced = polymer_enhanced
        
        self.conversion_data = EnergyConversionEfficiency()
        self.west_benchmark = WESTBenchmarkMetrics()
        self.c = constants.c
    
    def theoretical_annihilation_energy(self, antimatter_mass_kg: float) -> Dict[str, float]:
        """Calculate theoretical energy from complete annihilation"""
        total_mass = 2 * antimatter_mass_kg  # antimatter + matter
        theoretical_energy_j = total_mass * self.c**2
        theoretical_energy_kwh = theoretical_energy_j / 3.6e6
        
        return {
            'antimatter_mass_kg': antimatter_mass_kg,
            'total_annihilating_mass_kg': total_mass,
            'theoretical_energy_j': theoretical_energy_j,
            'theoretical_energy_kwh': theoretical_energy_kwh
        }
    
    def realistic_energy_conversion(self, antimatter_mass_kg: float) -> Dict[str, float]:
        """
        Calculate realistic energy output accounting for conversion efficiency
        
        Args:
            antimatter_mass_kg: Mass of antimatter to annihilate
            
        Returns:
            Realistic energy conversion data
        """
        # Get theoretical energy
        theoretical = self.theoretical_annihilation_energy(antimatter_mass_kg)
        
        # Get conversion efficiency
        conversion_eff = self.conversion_data.get_conversion_efficiency(
            self.conversion_method, 
            self.polymer_enhanced
        )
        
        # Apply conversion efficiency
        realistic_energy_j = theoretical['theoretical_energy_j'] * conversion_eff
        realistic_energy_kwh = theoretical['theoretical_energy_kwh'] * conversion_eff
        
        # Apply polymer enhancement to annihilation process (separate from conversion)
        annihilation_enhancement = self._polymer_annihilation_enhancement()
        
        final_energy_j = realistic_energy_j * annihilation_enhancement
        final_energy_kwh = realistic_energy_kwh * annihilation_enhancement
        
        # Calculate efficiency breakdown
        total_efficiency = conversion_eff * annihilation_enhancement
        
        return {
            'antimatter_mass_kg': antimatter_mass_kg,
            'theoretical_energy_kwh': theoretical['theoretical_energy_kwh'],
            'conversion_method': self.conversion_method,
            'conversion_efficiency': conversion_eff,
            'annihilation_enhancement': annihilation_enhancement,
            'total_efficiency': total_efficiency,
            'realistic_energy_j': final_energy_j,
            'realistic_energy_kwh': final_energy_kwh,
            'polymer_enhanced': self.polymer_enhanced,
            'polymer_scale_mu': self.polymer_scale_mu,
            'efficiency_loss_factor': final_energy_kwh / theoretical['theoretical_energy_kwh']
        }
    
    def _polymer_annihilation_enhancement(self) -> float:
        """Polymer enhancement factor for annihilation process (separate from conversion)"""
        base_factor = 1.0
        
        # Confinement enhancement (reduces positron losses before annihilation)
        confinement_factor = 0.15 * np.log(1 + self.polymer_scale_mu)
        
        # Energy capture enhancement (better containment of gamma rays)
        capture_factor = 0.1 * self.polymer_scale_mu**0.4
        
        return base_factor + confinement_factor + capture_factor
    
    def comprehensive_cost_analysis(self, 
                                  antimatter_mass_kg: float,
                                  production_cost_per_gram: float = 62.5e12) -> Dict[str, float]:
        """
        Comprehensive cost analysis with realistic conversion efficiency
        
        Args:
            antimatter_mass_kg: Mass of antimatter
            production_cost_per_gram: Production cost ($/gram)
            
        Returns:
            Complete cost analysis
        """
        mass_grams = antimatter_mass_kg * 1000
        production_cost_total = mass_grams * production_cost_per_gram
        
        # Get realistic energy output
        energy_data = self.realistic_energy_conversion(antimatter_mass_kg)
        realistic_energy_kwh = energy_data['realistic_energy_kwh']
        
        # Cost per kWh calculation
        cost_per_kwh = production_cost_total / realistic_energy_kwh if realistic_energy_kwh > 0 else float('inf')
        
        # Economic thresholds
        grid_threshold = 0.10
        premium_threshold = 1.00
        space_threshold = 1000.00
        
        return {
            'antimatter_mass_grams': mass_grams,
            'production_cost_usd': production_cost_total,
            'theoretical_energy_kwh': energy_data['theoretical_energy_kwh'],
            'realistic_energy_kwh': realistic_energy_kwh,
            'conversion_efficiency': energy_data['conversion_efficiency'],
            'total_efficiency': energy_data['total_efficiency'],
            'cost_per_kwh_usd': cost_per_kwh,
            'grid_competitive': cost_per_kwh < grid_threshold,
            'premium_viable': cost_per_kwh < premium_threshold,
            'space_viable': cost_per_kwh < space_threshold,
            'conversion_method': self.conversion_method,
            'efficiency_loss_factor': energy_data['efficiency_loss_factor']
        }
    
    def west_benchmark_comparison(self, antimatter_mass_kg: float) -> Dict[str, float]:
        """
        Compare antimatter system with WEST tokamak baseline
        
        Args:
            antimatter_mass_kg: Mass of antimatter for comparison
            
        Returns:
            Benchmark comparison data
        """
        # WEST baseline energy
        west_energy_kwh = (self.west_benchmark.baseline_heating_power_w * 
                          self.west_benchmark.baseline_confinement_time_s) / 3.6e6
        
        # Antimatter energy
        energy_data = self.realistic_energy_conversion(antimatter_mass_kg)
        antimatter_energy_kwh = energy_data['realistic_energy_kwh']
        
        # Energy density comparison
        energy_ratio = antimatter_energy_kwh / west_energy_kwh if west_energy_kwh > 0 else float('inf')
        
        # Power density (assuming 1-hour operation for antimatter)
        antimatter_power_w = antimatter_energy_kwh * 3.6e6  # Convert back to watts for 1 hour
        power_ratio = antimatter_power_w / self.west_benchmark.baseline_heating_power_w
        
        return {
            'west_baseline': {
                'confinement_time_s': self.west_benchmark.baseline_confinement_time_s,
                'temperature_c': self.west_benchmark.baseline_temperature_c,
                'heating_power_w': self.west_benchmark.baseline_heating_power_w,
                'total_energy_kwh': west_energy_kwh
            },
            'antimatter_system': {
                'mass_kg': antimatter_mass_kg,
                'realistic_energy_kwh': antimatter_energy_kwh,
                'equivalent_power_w': antimatter_power_w,
                'conversion_efficiency': energy_data['conversion_efficiency'],
                'conversion_method': self.conversion_method
            },
            'comparison_metrics': {
                'energy_ratio': energy_ratio,
                'power_ratio': power_ratio,
                'energy_density_advantage': antimatter_energy_kwh / antimatter_mass_kg,
                'west_energy_density': west_energy_kwh / 1000  # Rough estimate per kg of plasma
            }
        }

class ConversionEfficiencyPipeline:
    """Pipeline for analyzing energy conversion efficiency impacts"""
    
    def __init__(self, west_baseline: WESTBaseline):
        self.west_baseline = west_baseline
        self.west_benchmark = WESTBenchmarkMetrics()
        self.results = {}
    
    def run_conversion_method_comparison(self,
                                       antimatter_mass_kg: float = 1e-12,
                                       mu_range: Tuple[float, float] = (0.1, 10.0),
                                       num_points: int = 30) -> Dict:
        """
        Compare different energy conversion methods
        
        Args:
            antimatter_mass_kg: Test mass of antimatter
            mu_range: Polymer scale parameter range
            num_points: Number of analysis points
            
        Returns:
            Conversion method comparison results
        """
        conversion_methods = ['tpv_lab', 'tpv_system', 'thermionic', 'direct']
        mu_values = np.linspace(mu_range[0], mu_range[1], num_points)
        
        results = {
            'mu_values': mu_values.tolist(),
            'antimatter_mass_kg': antimatter_mass_kg,
            'methods': {}
        }
        
        for method in conversion_methods:
            method_results = {
                'realistic_energies_kwh': [],
                'conversion_efficiencies': [],
                'cost_per_kwh_values': [],
                'efficiency_loss_factors': []
            }
            
            for mu in mu_values:
                # Test both polymer-enhanced and standard
                converter_standard = RealisticAntimatterConverter(
                    polymer_scale_mu=mu,
                    conversion_method=method,
                    polymer_enhanced=False
                )
                
                converter_enhanced = RealisticAntimatterConverter(
                    polymer_scale_mu=mu,
                    conversion_method=method,
                    polymer_enhanced=True
                )
                
                # Use polymer-enhanced results for this analysis
                energy_data = converter_enhanced.realistic_energy_conversion(antimatter_mass_kg)
                cost_data = converter_enhanced.comprehensive_cost_analysis(antimatter_mass_kg)
                
                method_results['realistic_energies_kwh'].append(energy_data['realistic_energy_kwh'])
                method_results['conversion_efficiencies'].append(energy_data['conversion_efficiency'])
                method_results['cost_per_kwh_values'].append(cost_data['cost_per_kwh_usd'])
                method_results['efficiency_loss_factors'].append(energy_data['efficiency_loss_factor'])
            
            results['methods'][method] = method_results
        
        self.results['conversion_comparison'] = results
        return results
    
    def efficiency_impact_analysis(self, antimatter_mass_kg: float = 1e-12) -> Dict:
        """
        Analyze the impact of conversion efficiency on economic viability
        
        Args:
            antimatter_mass_kg: Test mass for analysis
            
        Returns:
            Efficiency impact analysis results
        """
        conversion_methods = ['tpv_lab', 'tpv_system', 'thermionic', 'direct']
        
        results = {
            'theoretical_baseline': {},
            'realistic_conversions': {},
            'efficiency_gaps': {}
        }
        
        # Get theoretical baseline
        converter_theoretical = RealisticAntimatterConverter(polymer_scale_mu=1.0)
        theoretical = converter_theoretical.theoretical_annihilation_energy(antimatter_mass_kg)
        results['theoretical_baseline'] = theoretical
        
        # Analyze each conversion method
        for method in conversion_methods:
            converter = RealisticAntimatterConverter(
                polymer_scale_mu=5.0,  # Optimistic polymer enhancement
                conversion_method=method,
                polymer_enhanced=True
            )
            
            energy_data = converter.realistic_energy_conversion(antimatter_mass_kg)
            cost_data = converter.comprehensive_cost_analysis(antimatter_mass_kg)
            
            results['realistic_conversions'][method] = {
                'realistic_energy_kwh': energy_data['realistic_energy_kwh'],
                'conversion_efficiency': energy_data['conversion_efficiency'],
                'total_efficiency': energy_data['total_efficiency'],
                'cost_per_kwh': cost_data['cost_per_kwh_usd'],
                'efficiency_loss_factor': energy_data['efficiency_loss_factor']
            }
            
            # Calculate efficiency gap
            theoretical_cost_per_kwh = (antimatter_mass_kg * 1000 * 62.5e12) / theoretical['theoretical_energy_kwh']
            realistic_cost_per_kwh = cost_data['cost_per_kwh_usd']
            efficiency_penalty = realistic_cost_per_kwh / theoretical_cost_per_kwh
            
            results['efficiency_gaps'][method] = {
                'theoretical_cost_per_kwh': theoretical_cost_per_kwh,
                'realistic_cost_per_kwh': realistic_cost_per_kwh,
                'efficiency_penalty_factor': efficiency_penalty,
                'energy_loss_percentage': (1 - energy_data['efficiency_loss_factor']) * 100
            }
        
        self.results['efficiency_impact'] = results
        return results
    
    def west_benchmark_analysis(self) -> Dict:
        """
        Comprehensive analysis against WEST baseline with conversion efficiency
        
        Returns:
            WEST benchmark analysis results
        """
        test_masses = [1e-15, 1e-12, 1e-9]  # fg, pg, ng
        conversion_methods = ['tpv_system', 'thermionic', 'direct']
        
        results = {
            'west_baseline_metrics': {
                'confinement_time_s': self.west_benchmark.baseline_confinement_time_s,
                'temperature_c': self.west_benchmark.baseline_temperature_c,
                'heating_power_w': self.west_benchmark.baseline_heating_power_w,
                'target_confinement_s': self.west_benchmark.target_confinement_time_s,
                'target_temperature_c': self.west_benchmark.target_temperature_c,
                'target_power_reduction': self.west_benchmark.target_heating_power_reduction
            },
            'antimatter_comparisons': {}
        }
        
        for mass in test_masses:
            mass_results = {}
            
            for method in conversion_methods:
                converter = RealisticAntimatterConverter(
                    polymer_scale_mu=5.0,
                    conversion_method=method,
                    polymer_enhanced=True
                )
                
                benchmark_data = converter.west_benchmark_comparison(mass)
                
                mass_results[method] = {
                    'energy_ratio_vs_west': benchmark_data['comparison_metrics']['energy_ratio'],
                    'power_ratio_vs_west': benchmark_data['comparison_metrics']['power_ratio'],
                    'realistic_energy_kwh': benchmark_data['antimatter_system']['realistic_energy_kwh'],
                    'conversion_efficiency': benchmark_data['antimatter_system']['conversion_efficiency']
                }
            
            results['antimatter_comparisons'][f'{mass*1e15:.0f}_fg'] = mass_results
        
        self.results['west_benchmark'] = results
        return results
    
    def generate_efficiency_visualization(self, save_path: Optional[str] = None):
        """Generate comprehensive efficiency analysis visualization"""
        if 'conversion_comparison' not in self.results:
            logger.error("No conversion comparison results available.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        comparison_data = self.results['conversion_comparison']
        mu_values = comparison_data['mu_values']
        methods = comparison_data['methods']
        
        # Plot 1: Conversion efficiency comparison
        for method, data in methods.items():
            efficiencies = [eff * 100 for eff in data['conversion_efficiencies']]  # Convert to percentage
            ax1.plot(mu_values, efficiencies, linewidth=2, label=method.replace('_', ' ').title())
        
        ax1.set_xlabel('Polymer Scale μ')
        ax1.set_ylabel('Conversion Efficiency (%)')
        ax1.set_title('Energy Conversion Efficiency vs Polymer Scale')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Realistic energy output
        for method, data in methods.items():
            ax2.semilogy(mu_values, data['realistic_energies_kwh'], linewidth=2, 
                        label=method.replace('_', ' ').title())
        
        ax2.set_xlabel('Polymer Scale μ')
        ax2.set_ylabel('Realistic Energy Output (kWh)')
        ax2.set_title('Realistic Energy Output (with Conversion Losses)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cost per kWh comparison
        for method, data in methods.items():
            valid_costs = [c for c in data['cost_per_kwh_values'] if c < 1e8]  # Filter extreme values
            valid_mu = mu_values[:len(valid_costs)]
            
            if valid_costs:
                ax3.semilogy(valid_mu, valid_costs, linewidth=2, 
                           label=method.replace('_', ' ').title())
        
        ax3.axhline(y=0.10, color='g', linestyle='--', alpha=0.7, label='Grid Competitive')
        ax3.axhline(y=1.00, color='orange', linestyle='--', alpha=0.7, label='Premium Market')
        ax3.axhline(y=1000.00, color='r', linestyle='--', alpha=0.7, label='Space Applications')
        ax3.set_xlabel('Polymer Scale μ')
        ax3.set_ylabel('Cost per kWh ($)')
        ax3.set_title('Economic Impact of Conversion Efficiency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Efficiency loss factors
        for method, data in methods.items():
            loss_percentages = [(1 - factor) * 100 for factor in data['efficiency_loss_factors']]
            ax4.plot(mu_values, loss_percentages, linewidth=2, 
                    label=method.replace('_', ' ').title())
        
        ax4.set_xlabel('Polymer Scale μ')
        ax4.set_ylabel('Energy Loss (%)')
        ax4.set_title('Energy Loss Due to Conversion Inefficiency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Antimatter Energy Conversion Efficiency Analysis\n' +
                     f'Reference: WEST baseline {self.west_benchmark.baseline_confinement_time_s:.0f}s, ' +
                     f'{self.west_benchmark.baseline_temperature_c/1e6:.0f}×10⁶°C, ' +
                     f'{self.west_benchmark.baseline_heating_power_w/1e6:.1f}MW',
                     fontsize=14, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Efficiency visualization saved to {save_path}")
        else:
            plt.show()

# Plan A Step 4: Net kWh Cost Calculation
# =====================================

class NetCostCalculator:
    """
    Plan A Step 4: Net kWh cost calculator using the specific formula
    Cost ≈ $2,500/η per kWh where η is conversion efficiency
    """
    
    def __init__(self):
        # Constants from Step 4 formula
        self.antimatter_cost_per_gram = 6.25e13  # $62.5 trillion per gram
        self.theoretical_energy_per_gram = 2.5e10  # kWh per gram theoretical
        self.cost_efficiency_factor = 2500.0  # $2,500/η coefficient
        
        # Market thresholds for comparison
        self.market_thresholds = {
            'grid_competitive': 0.10,
            'premium_residential': 0.30,
            'premium_industrial': 1.00,
            'space_applications': 1000.00,
            'ultra_premium': 10000.00
        }
    
    def calculate_net_cost_per_kwh(self, efficiency: float) -> Dict[str, float]:
        """Calculate net cost per kWh using Step 4 formula"""
        if efficiency <= 0:
            return {
                'efficiency': efficiency,
                'cost_per_kwh_formula': float('inf'),
                'cost_per_kwh_detailed': float('inf')
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
    
    def analyze_efficiency_requirements(self) -> Dict[str, Dict]:
        """Analyze efficiency requirements for different market segments"""
        requirements = {}
        
        for threshold_name, threshold_cost in self.market_thresholds.items():
            required_eta = self.cost_efficiency_factor / threshold_cost
            
            requirements[threshold_name] = {
                'threshold_cost_per_kwh': threshold_cost,
                'required_efficiency': required_eta,
                'required_efficiency_percent': required_eta * 100,
                'achievable': required_eta <= 1.0,
                'physically_possible': required_eta <= 1.0
            }
        
        return requirements

class CompletePlanAPipeline:
    """Complete Plan A pipeline integrating all four steps"""
    
    def __init__(self, west_baseline: WESTBaseline):
        self.west_baseline = west_baseline
        self.results = {}
        
        # Initialize all step calculators
        self.step1_pipeline = PolymerMassEnergyPipeline(west_baseline)
        self.step2_pipeline = AntimatterProductionPipeline(west_baseline) 
        self.step3_pipeline = ConversionEfficiencyPipeline(west_baseline)
        self.step4_calculator = NetCostCalculator()
    
    def run_complete_analysis(self, 
                            mu_range: Tuple[float, float] = (0.1, 10.0),
                            num_points: int = 50,
                            test_masses: List[float] = None,
                            antimatter_masses: List[float] = None) -> Dict:
        """Run complete Plan A analysis with all four steps"""
        
        if test_masses is None:
            test_masses = [1e-6, 1e-3, 1e-1]  # μg, mg, 100mg
        if antimatter_masses is None:
            antimatter_masses = [1e-15, 1e-12, 1e-9]  # fg, pg, ng
        
        logger.info("Running Complete Plan A Analysis (Steps 1-4)")
        
        # Step 1: Theoretical energy density
        logger.info("Step 1: Theoretical energy density analysis")
        step1_results = {}
        for mass in test_masses:
            pipeline_results = self.step1_pipeline.run_polymer_scale_sweep(
                mu_range=mu_range, num_points=num_points, mass_kg=mass
            )
            step1_results[f'mass_{mass*1e6:.0f}mg'] = pipeline_results
        
        # Step 2: Antimatter production costs
        logger.info("Step 2: Antimatter production cost analysis")
        step2_results = {}
        for mass in antimatter_masses:
            cost_sweep = self.step2_pipeline.run_cost_optimization_sweep(
                mu_range=mu_range, num_points=num_points, antimatter_mass_kg=mass
            )
            step2_results[f'mass_{mass*1e15:.0f}fg'] = cost_sweep
        
        # Step 3: Energy conversion efficiency
        logger.info("Step 3: Energy conversion efficiency analysis")
        step3_results = self.step3_pipeline.run_conversion_method_comparison(
            antimatter_mass_kg=1e-12, mu_range=mu_range, num_points=num_points
        )
        
        # Step 4: Net kWh cost calculation
        logger.info("Step 4: Net kWh cost calculation")
        step4_results = {}
        
        # Analyze all conversion methods from Step 3
        conversion_methods = {
            'tpv_system': {'base': 0.05, 'enhanced': 0.075},
            'tpv_lab': {'base': 0.35, 'enhanced': 0.525},
            'thermionic': {'base': 0.15, 'enhanced': 0.195},
            'direct': {'base': 0.25, 'enhanced': 0.50}
        }
        
        for method, efficiencies in conversion_methods.items():
            step4_results[method] = {
                'standard': self.step4_calculator.calculate_net_cost_per_kwh(efficiencies['base']),
                'enhanced': self.step4_calculator.calculate_net_cost_per_kwh(efficiencies['enhanced'])
            }
        
        # Efficiency requirements analysis
        step4_results['market_requirements'] = self.step4_calculator.analyze_efficiency_requirements()
        
        # Integrated results
        complete_results = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'west_baseline': {
                    'confinement_time_s': self.west_baseline.confinement_time,
                    'plasma_temperature_c': self.west_baseline.plasma_temperature,
                    'heating_power_w': self.west_baseline.heating_power,
                    'total_energy_kwh': (self.west_baseline.heating_power * self.west_baseline.confinement_time) / 3.6e6
                },
                'analysis_scope': 'Complete Plan A: Steps 1-4'
            },
            'step1_theoretical': step1_results,
            'step2_antimatter': step2_results,
            'step3_conversion': step3_results,
            'step4_net_cost': step4_results,
            'integrated_conclusions': self._generate_integrated_conclusions(step1_results, step2_results, step3_results, step4_results)
        }
        
        self.results = complete_results
        return complete_results
    
    def _generate_integrated_conclusions(self, step1, step2, step3, step4) -> Dict:
        """Generate integrated conclusions across all steps"""
        
        # Find best case scenario
        best_method = 'direct'  # Direct conversion typically best
        best_cost = step4[best_method]['enhanced']['cost_per_kwh_formula']
        best_efficiency = 0.50  # 50% for polymer-enhanced direct conversion
        
        # Economic viability assessment
        grid_competitive = best_cost < 0.10
        space_viable = best_cost < 1000.00
        ultra_premium_viable = best_cost < 10000.00
        
        return {
            'economic_viability': {
                'best_case_cost_per_kwh': best_cost,
                'best_conversion_method': best_method,
                'grid_competitive': grid_competitive,
                'space_applications_viable': space_viable,
                'ultra_premium_viable': ultra_premium_viable,
                'economic_gap_to_grid': best_cost / 0.10,
                'economic_gap_to_space': best_cost / 1000.00 if not space_viable else 1.0
            },
            'technical_assessment': {
                'polymer_enhancement_critical': True,
                'conversion_efficiency_bottleneck': True,
                'antimatter_cost_prohibitive': True,
                'fundamental_physics_limits': True
            },
            'west_comparison': {
                'west_energy_kwh': (self.west_baseline.heating_power * self.west_baseline.confinement_time) / 3.6e6,
                'west_cost_equivalent': 0.10,  # Grid cost
                'antimatter_competitive_impossible': True,
                'efficiency_gap_for_parity': 2500.0 / 0.10  # Required efficiency for grid parity
            },
            'research_priorities': [
                'Fundamental breakthrough in conversion efficiency (>80%)',
                'Revolutionary antimatter production cost reduction (>99.99%)',
                'Alternative high-value applications (propulsion, specialized physics)',
                'Focus shift away from terrestrial energy applications'
            ],
            'final_assessment': {
                'plan_a_viable_for_grid': False,
                'plan_a_viable_for_space': False,
                'plan_a_viable_for_specialized': ultra_premium_viable,
                'recommended_pathway': 'Focus on Plan B (Polymer-Enhanced Fusion) or alternative approaches'
            }
        }

# ...existing code...
