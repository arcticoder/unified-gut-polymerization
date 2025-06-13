"""
Plan B: Polymer-Enhanced Fusion Analysis
=======================================

This module implements polymer-enhanced fusion pathways using the WEST tokamak
baseline as calibration reference for comparative analysis.

Based on WEST tokamak world record (February 12, 2025):
- Confinement Time: 1,337 s (22 min 17 s) - 25% increase over EAST
- Plasma Temperature: 50×10⁶ °C (≈ 3× hotter than Sun's core) 
- Heating Power: 2 MW RF injection
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
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
    
    # Additional fusion parameters
    plasma_density: float = 1e20  # particles/m³ (typical tokamak)
    magnetic_field: float = 3.7  # Tesla (WEST specification)
    major_radius: float = 2.5  # meters
    minor_radius: float = 0.5  # meters
    
    @property
    def plasma_temp_kelvin(self) -> float:
        """Convert plasma temperature to Kelvin"""
        return self.plasma_temperature + 273.15
    
    @property
    def plasma_volume(self) -> float:
        """Estimate plasma volume (torus)"""
        return 2 * np.pi**2 * self.major_radius * self.minor_radius**2

@dataclass
class FusionReaction:
    """Fusion reaction parameters and cross-sections"""
    
    def __init__(self, reaction_type: str = "D-T"):
        """
        Initialize fusion reaction
        
        Args:
            reaction_type: Type of fusion reaction ("D-T", "D-D", "D-He3")
        """
        self.reaction_type = reaction_type
        self.q_value = self._get_q_value()  # Energy release per reaction (MeV)
        
    def _get_q_value(self) -> float:
        """Get Q-value (energy release) for reaction type"""
        q_values = {
            "D-T": 17.6,    # D + T → α + n + 17.6 MeV
            "D-D": 3.27,    # D + D → T + p + 4.0 MeV (branch 1)
            "D-He3": 18.3   # D + ³He → α + p + 18.3 MeV
        }
        return q_values.get(self.reaction_type, 17.6)
    
    def cross_section(self, energy_kev: float) -> float:
        """
        Calculate fusion cross-section using parameterized fits
        
        Args:
            energy_kev: Center-of-mass energy in keV
            
        Returns:
            Cross-section in barns (10^-24 cm²)
        """
        if self.reaction_type == "D-T":
            # Parameterized D-T cross-section (Bosch-Hale formula)
            if energy_kev < 0.5:
                return 0.0
            
            A1, A2, A3, A4, A5 = 45.95, 50200, 1.368e-2, 1.076, 409.2
            sigma = (A1 / (energy_kev * (A2 + energy_kev * (A3 + energy_kev * A4)))) * \
                    np.exp(-A5 / np.sqrt(energy_kev))
            return sigma
        
        elif self.reaction_type == "D-D":
            # Simplified D-D cross-section
            if energy_kev < 1.0:
                return 0.0
            return 0.5 * np.exp(-31.4 / np.sqrt(energy_kev))
        
        else:
            # Default approximation
            return 0.1 * np.exp(-50.0 / np.sqrt(energy_kev))

class PolymerEnhancedFusion:
    """Polymer-enhanced fusion reactor model"""
    
    def __init__(self, 
                 west_baseline: WESTBaseline,
                 polymer_scale_mu: float = 1.0,
                 fusion_reaction: FusionReaction = None):
        """
        Initialize polymer-enhanced fusion model
        
        Args:
            west_baseline: WEST tokamak baseline parameters
            polymer_scale_mu: Polymer enhancement scale parameter
            fusion_reaction: Fusion reaction object
        """
        self.west = west_baseline
        self.polymer_scale_mu = polymer_scale_mu
        self.fusion_reaction = fusion_reaction or FusionReaction("D-T")
        
        # Physical constants
        self.k_b = constants.k  # Boltzmann constant
        self.m_p = constants.m_p  # Proton mass
        self.e = constants.e  # Elementary charge
        
    def polymer_enhancement_factor(self, parameter_type: str) -> float:
        """
        Calculate polymer enhancement for different physical parameters
        
        Args:
            parameter_type: Type of parameter ("confinement", "cross_section", "temperature")
            
        Returns:
            Enhancement factor
        """
        base_enhancement = 1.0
        
        if parameter_type == "confinement":
            # Polymer-enhanced magnetic confinement
            # Based on modified field line topology
            enhancement = base_enhancement + 0.2 * np.log(1 + self.polymer_scale_mu)
            
        elif parameter_type == "cross_section":
            # Polymer-induced cross-section enhancement
            # Quantum tunneling enhancement through polymer effects
            enhancement = base_enhancement + 0.15 * self.polymer_scale_mu**0.5
            
        elif parameter_type == "temperature":
            # Temperature enhancement through polymer heat trapping
            enhancement = base_enhancement + 0.1 * self.polymer_scale_mu**0.3
            
        else:
            enhancement = base_enhancement
            
        return enhancement
    
    def enhanced_confinement_time(self) -> float:
        """Calculate polymer-enhanced confinement time"""
        base_time = self.west.confinement_time
        enhancement = self.polymer_enhancement_factor("confinement")
        return base_time * enhancement
    
    def enhanced_fusion_rate(self, density: float, temperature_kev: float) -> float:
        """
        Calculate polymer-enhanced fusion reaction rate
        
        Args:
            density: Plasma density (particles/m³)
            temperature_kev: Temperature in keV
            
        Returns:
            Reaction rate (reactions/m³/s)
        """
        # Enhanced cross-section
        base_sigma = self.fusion_reaction.cross_section(temperature_kev)
        enhanced_sigma = base_sigma * self.polymer_enhancement_factor("cross_section")
        
        # Enhanced temperature
        enhanced_temp = temperature_kev * self.polymer_enhancement_factor("temperature")
        
        # Thermal velocity
        v_thermal = np.sqrt(8 * enhanced_temp * 1000 * self.e / (np.pi * self.m_p))
        
        # Reaction rate (assuming equal densities of reactants)
        n1 = n2 = density / 2  # Split density between two species
        rate = n1 * n2 * enhanced_sigma * 1e-24 * v_thermal  # Convert barns to m²
        
        return rate
    
    def power_output_analysis(self) -> Dict[str, float]:
        """
        Analyze power output for polymer-enhanced fusion
        
        Returns:
            Power analysis dictionary
        """
        # Enhanced parameters
        enhanced_confinement = self.enhanced_confinement_time()
        
        # Convert WEST temperature to keV
        temp_kev = (self.west.plasma_temp_kelvin * self.k_b) / (self.e * 1000)
        enhanced_temp_kev = temp_kev * self.polymer_enhancement_factor("temperature")
        
        # Fusion rate calculation
        fusion_rate = self.enhanced_fusion_rate(self.west.plasma_density, enhanced_temp_kev)
        
        # Power calculations
        total_reactions_per_second = fusion_rate * self.west.plasma_volume
        q_value_joules = self.fusion_reaction.q_value * 1.602e-13  # Convert MeV to J
        fusion_power = total_reactions_per_second * q_value_joules
        
        # Net power (fusion - heating)
        net_power = fusion_power - self.west.heating_power
        q_factor = fusion_power / self.west.heating_power if self.west.heating_power > 0 else 0
        
        return {
            'enhanced_confinement_time_s': enhanced_confinement,
            'enhanced_temperature_kev': enhanced_temp_kev,
            'fusion_rate_per_m3_per_s': fusion_rate,
            'total_fusion_power_w': fusion_power,
            'heating_power_w': self.west.heating_power,
            'net_power_w': net_power,
            'q_factor': q_factor,
            'polymer_scale_mu': self.polymer_scale_mu,
            'baseline_confinement_s': self.west.confinement_time,
            'confinement_improvement_factor': enhanced_confinement / self.west.confinement_time
        }
    
    def economic_analysis(self, 
                         reactor_cost_usd: float = 20e9,  # $20B reactor cost
                         operating_years: float = 30) -> Dict[str, float]:
        """
        Economic analysis of polymer-enhanced fusion
        
        Args:
            reactor_cost_usd: Total reactor construction cost
            operating_years: Operating lifetime in years
            
        Returns:
            Economic analysis dictionary
        """
        power_data = self.power_output_analysis()
        
        # Annual energy production
        net_power_w = max(0, power_data['net_power_w'])  # Can't be negative
        annual_energy_kwh = (net_power_w * 8760) / 1000  # kWh/year
        total_lifetime_energy_kwh = annual_energy_kwh * operating_years
        
        # Cost analysis
        cost_per_kwh = reactor_cost_usd / total_lifetime_energy_kwh if total_lifetime_energy_kwh > 0 else float('inf')
        
        return {
            'reactor_cost_usd': reactor_cost_usd,
            'operating_years': operating_years,
            'net_power_mw': net_power_w / 1e6,
            'annual_energy_gwh': annual_energy_kwh / 1e6,
            'lifetime_energy_twh': total_lifetime_energy_kwh / 1e9,
            'cost_per_kwh_usd': cost_per_kwh,
            'competitive_threshold_usd_per_kwh': 0.10,
            'is_economically_viable': cost_per_kwh < 0.10,
            'q_factor': power_data['q_factor'],
            'breakeven_q_factor': 1.0
        }

class PolymerFusionPipeline:
    """Complete pipeline for polymer-enhanced fusion analysis"""
    
    def __init__(self, west_baseline: WESTBaseline):
        self.west_baseline = west_baseline
        self.results = {}
        
    def run_polymer_scale_sweep(self,
                               mu_range: Tuple[float, float] = (0.1, 10.0),
                               num_points: int = 50,
                               reactor_cost_usd: float = 20e9) -> Dict:
        """
        Sweep polymer scale parameter for fusion optimization
        
        Args:
            mu_range: Range of polymer scale values
            num_points: Number of points in sweep
            reactor_cost_usd: Reactor cost for economic analysis
            
        Returns:
            Sweep results dictionary
        """
        mu_values = np.linspace(mu_range[0], mu_range[1], num_points)
        
        results = {
            'mu_values': mu_values.tolist(),
            'q_factors': [],
            'net_powers_mw': [],
            'cost_per_kwh_values': [],
            'confinement_improvements': [],
            'economic_viability': []
        }
        
        for mu in mu_values:
            fusion_model = PolymerEnhancedFusion(
                west_baseline=self.west_baseline,
                polymer_scale_mu=mu
            )
            
            power_data = fusion_model.power_output_analysis()
            economic_data = fusion_model.economic_analysis(reactor_cost_usd)
            
            results['q_factors'].append(power_data['q_factor'])
            results['net_powers_mw'].append(power_data['net_power_w'] / 1e6)
            results['cost_per_kwh_values'].append(economic_data['cost_per_kwh_usd'])
            results['confinement_improvements'].append(power_data['confinement_improvement_factor'])
            results['economic_viability'].append(economic_data['is_economically_viable'])
        
        # Find breakeven and economic crossover points
        breakeven_indices = [i for i, q in enumerate(results['q_factors']) if q >= 1.0]
        viable_indices = [i for i, viable in enumerate(results['economic_viability']) if viable]
        
        results['breakeven_mu'] = mu_values[breakeven_indices[0]] if breakeven_indices else None
        results['economic_crossover_mu'] = mu_values[viable_indices[0]] if viable_indices else None
        
        self.results['polymer_fusion_sweep'] = results
        return results
    
    def compare_with_west_baseline(self) -> Dict:
        """
        Compare polymer-enhanced fusion with WEST baseline
        
        Returns:
            Comparison metrics
        """
        # WEST baseline metrics
        west_energy_total_kwh = (self.west_baseline.heating_power * self.west_baseline.confinement_time) / 3.6e6
        
        # Find optimal polymer scale
        if 'polymer_fusion_sweep' in self.results:
            sweep_data = self.results['polymer_fusion_sweep']
            optimal_idx = np.argmax(sweep_data['q_factors'])
            optimal_mu = sweep_data['mu_values'][optimal_idx]
        else:
            optimal_mu = 1.0
        
        # Polymer-enhanced metrics
        optimal_fusion = PolymerEnhancedFusion(
            west_baseline=self.west_baseline,
            polymer_scale_mu=optimal_mu
        )
        
        optimal_power = optimal_fusion.power_output_analysis()
        optimal_economics = optimal_fusion.economic_analysis()
        
        return {
            'west_baseline': {
                'confinement_time_s': self.west_baseline.confinement_time,
                'heating_power_mw': self.west_baseline.heating_power / 1e6,
                'total_energy_input_kwh': west_energy_total_kwh,
                'q_factor': 0.0  # WEST is not ignition
            },
            'polymer_enhanced': {
                'optimal_mu': optimal_mu,
                'confinement_time_s': optimal_power['enhanced_confinement_time_s'],
                'q_factor': optimal_power['q_factor'],
                'net_power_mw': optimal_power['net_power_w'] / 1e6,
                'cost_per_kwh_usd': optimal_economics['cost_per_kwh_usd'],
                'confinement_improvement': optimal_power['confinement_improvement_factor']
            }
        }
    
    def generate_fusion_visualization(self, save_path: Optional[str] = None):
        """Generate visualization of polymer fusion analysis"""
        if 'polymer_fusion_sweep' not in self.results:
            logger.error("No polymer fusion sweep results available.")
            return
        
        data = self.results['polymer_fusion_sweep']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Q-factor vs polymer scale
        ax1.plot(data['mu_values'], data['q_factors'], 'b-', linewidth=2)
        ax1.axhline(y=1.0, color='r', linestyle='--', label='Breakeven (Q=1)')
        ax1.axhline(y=10.0, color='g', linestyle='--', label='Ignition (Q=10)')
        ax1.set_xlabel('Polymer Scale μ')
        ax1.set_ylabel('Q-Factor')
        ax1.set_title('Fusion Q-Factor vs Polymer Scale')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Net power vs polymer scale
        ax2.plot(data['mu_values'], data['net_powers_mw'], 'g-', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', label='Zero Net Power')
        ax2.set_xlabel('Polymer Scale μ')
        ax2.set_ylabel('Net Power (MW)')
        ax2.set_title('Net Power Output vs Polymer Scale')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cost analysis
        valid_costs = [c for c in data['cost_per_kwh_values'] if c < 1.0]  # Filter extreme values
        valid_mu = [mu for mu, c in zip(data['mu_values'], data['cost_per_kwh_values']) if c < 1.0]
        
        if valid_costs:
            ax3.semilogy(valid_mu, valid_costs, 'purple', linewidth=2)
        ax3.axhline(y=0.10, color='r', linestyle='--', label='Economic Threshold ($0.10/kWh)')
        ax3.set_xlabel('Polymer Scale μ')
        ax3.set_ylabel('Cost per kWh ($)')
        ax3.set_title('Economic Analysis vs Polymer Scale')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Confinement improvement
        ax4.plot(data['mu_values'], data['confinement_improvements'], 'orange', linewidth=2)
        ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='WEST Baseline')
        ax4.set_xlabel('Polymer Scale μ')
        ax4.set_ylabel('Confinement Improvement Factor')
        ax4.set_title('Confinement Enhancement vs Polymer Scale')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fusion visualization saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, filepath: str):
        """Save all results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filepath}")

def demonstrate_plan_b():
    """Demonstration of Plan B: Polymer-Enhanced Fusion"""
    print("=" * 60)
    print("Plan B: Polymer-Enhanced Fusion Demonstration")
    print("=" * 60)
    
    # Initialize WEST baseline
    west = WESTBaseline()
    print(f"WEST Baseline ({west.date}):")
    print(f"  Confinement Time: {west.confinement_time:.0f} s")
    print(f"  Plasma Temperature: {west.plasma_temperature/1e6:.0f}×10⁶ °C")
    print(f"  Heating Power: {west.heating_power/1e6:.1f} MW")
    print(f"  Magnetic Field: {west.magnetic_field:.1f} T")
    print()
    
    # Example polymer-enhanced fusion
    fusion_model = PolymerEnhancedFusion(west, polymer_scale_mu=2.0)
    power_data = fusion_model.power_output_analysis()
    
    print("Polymer-Enhanced Fusion (μ=2.0):")
    print(f"  Enhanced Confinement: {power_data['enhanced_confinement_time_s']:.0f} s")
    print(f"  Confinement Improvement: {power_data['confinement_improvement_factor']:.2f}×")
    print(f"  Q-Factor: {power_data['q_factor']:.3f}")
    print(f"  Net Power: {power_data['net_power_w']/1e6:.2f} MW")
    print()
    
    # Economic analysis
    economic_data = fusion_model.economic_analysis()
    print("Economic Analysis:")
    print(f"  Cost per kWh: ${economic_data['cost_per_kwh_usd']:.3f}")
    print(f"  Economically viable: {economic_data['is_economically_viable']}")
    print(f"  Annual energy: {economic_data['annual_energy_gwh']:.1f} GWh")
    print()
    
    # Run pipeline
    pipeline = PolymerFusionPipeline(west)
    sweep_results = pipeline.run_polymer_scale_sweep()
    
    if sweep_results['breakeven_mu'] is not None:
        print(f"Fusion breakeven (Q≥1) at μ = {sweep_results['breakeven_mu']:.3f}")
    else:
        print("No fusion breakeven found in tested range")
        
    if sweep_results['economic_crossover_mu'] is not None:
        print(f"Economic viability at μ = {sweep_results['economic_crossover_mu']:.3f}")
    else:
        print("No economic viability found in tested range")
    
    # Comparison
    comparison = pipeline.compare_with_west_baseline()
    print("\nComparison with WEST:")
    print(f"  WEST Q-factor: {comparison['west_baseline']['q_factor']:.1f}")
    print(f"  Optimal polymer Q-factor: {comparison['polymer_enhanced']['q_factor']:.3f}")
    print(f"  Optimal polymer net power: {comparison['polymer_enhanced']['net_power_mw']:.2f} MW")

if __name__ == "__main__":
    demonstrate_plan_b()
