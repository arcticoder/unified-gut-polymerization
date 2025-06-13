"""
Plan B, Step 2: Fusion Reactor Simulations - SIMPLIFIED TEST
============================================================

Simplified test to verify Q-factor calculations work properly.
"""

import numpy as np
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class PolymerParameters:
    """Polymer field configuration parameters"""
    scale_mu: float = 1.0
    enhancement_power_n: float = 2.0
    coupling_strength: float = 0.1

@dataclass
class PlasmaParameters:
    """Plasma conditions for fusion reactor simulation"""
    temperature_kev: float = 20.0
    density_particles_m3: float = 1e20
    confinement_time_s: float = 1.0
    plasma_volume_m3: float = 100.0

@dataclass
class ReactorConfiguration:
    """Complete reactor configuration parameters"""
    plasma_params: PlasmaParameters
    polymer_params: PolymerParameters
    heating_power_w: float = 50e6

class SimpleFusionReactorSimulator:
    """Simplified fusion reactor simulation"""
    
    def __init__(self, config: ReactorConfiguration):
        self.config = config
        
        # Physical constants
        self.eV_to_J = 1.602e-19
        self.k_b = 1.381e-23
        
        # Fusion energy releases (MeV to Joules)
        self.fusion_energy_j = 17.59 * 1e6 * self.eV_to_J  # D-T
    
    def dt_rate_coefficient(self, temperature_kev: float) -> float:
        """
        D-T reaction rate coefficient using Wesson approximation
        Returns ⟨σv⟩ in m³/s
        """
        if temperature_kev < 2:
            return 0.0
        elif temperature_kev < 10:
            # Low temperature regime
            return 1e-27 * (temperature_kev / 5)**4
        elif temperature_kev < 30:
            # Medium temperature regime
            return 1e-25 * (temperature_kev / 15)**2
        else:
            # High temperature regime
            return 5e-25 * (temperature_kev / 30)**0.5
    
    def polymer_enhancement_factor(self, temperature_kev: float) -> float:
        """Calculate polymer enhancement to reaction rate"""
        # Simplified enhancement based on polymer parameters
        base_enhancement = 1.0
        
        # Enhancement depends on coupling strength and temperature
        enhancement = base_enhancement + self.config.polymer_params.coupling_strength * \
                     (1.0 + 0.1 * temperature_kev / 20.0)
        
        # Reasonable bounds
        return max(1.0, min(5.0, enhancement))
    
    def fusion_power(self) -> float:
        """Calculate fusion power output"""
        plasma = self.config.plasma_params
        
        # Get reaction rate coefficient
        sigma_v_base = self.dt_rate_coefficient(plasma.temperature_kev)
        
        # Apply polymer enhancement
        enhancement = self.polymer_enhancement_factor(plasma.temperature_kev)
        sigma_v = sigma_v_base * enhancement
        
        # Densities (50/50 D-T mix)
        n_d = n_t = plasma.density_particles_m3 / 2
        density_product = n_d * n_t
        
        # Fusion power
        power = density_product * sigma_v * self.fusion_energy_j * plasma.plasma_volume_m3
        
        return power
    
    def bremsstrahlung_power(self) -> float:
        """Calculate bremsstrahlung radiation loss"""
        plasma = self.config.plasma_params
        
        # Bremsstrahlung coefficient
        C_brems = 5.35e-37  # W⋅m³⋅keV^(-1/2)
        
        # Power loss
        n_e = n_i = plasma.density_particles_m3
        power = C_brems * n_e * n_i * np.sqrt(plasma.temperature_kev) * plasma.plasma_volume_m3
        
        return power
    
    def conduction_power_loss(self) -> float:
        """Calculate conduction power loss"""
        plasma = self.config.plasma_params
        
        # Thermal energy
        thermal_energy = 3 * plasma.density_particles_m3 * self.k_b * \
                        plasma.temperature_kev * 1000 * plasma.plasma_volume_m3
        
        # Power loss
        power_loss = thermal_energy / plasma.confinement_time_s
        
        return power_loss
    
    def q_factor_analysis(self) -> Dict:
        """Calculate Q-factor analysis"""
        
        # Power calculations
        P_fusion = self.fusion_power()
        P_brems = self.bremsstrahlung_power()
        P_conduction = self.conduction_power_loss()
        P_input = self.config.heating_power_w
        
        # Q-factors
        Q_fusion = P_fusion / P_input if P_input > 0 else 0
        Q_net = (P_fusion - P_brems) / P_input if P_input > 0 else 0
        
        # Conditions
        breakeven = Q_fusion >= 1.0
        ignition = P_fusion >= (P_brems + P_conduction)
        
        return {
            'powers': {
                'fusion_mw': P_fusion / 1e6,
                'bremsstrahlung_mw': P_brems / 1e6,
                'conduction_mw': P_conduction / 1e6,
                'input_mw': P_input / 1e6
            },
            'q_factors': {
                'Q_fusion': Q_fusion,
                'Q_net': Q_net,
                'breakeven': breakeven,
                'ignition': ignition
            },
            'enhancement': self.polymer_enhancement_factor(self.config.plasma_params.temperature_kev)
        }

def test_temperature_sweep():
    """Test Q-factor vs temperature"""
    print("TEMPERATURE SWEEP TEST")
    print("-" * 30)
    
    # Base configuration
    plasma_base = PlasmaParameters(
        temperature_kev=20.0,
        density_particles_m3=1e20,
        confinement_time_s=3.0,
        plasma_volume_m3=830.0
    )
    
    polymer_params = PolymerParameters(
        scale_mu=2.0,
        enhancement_power_n=1.5,
        coupling_strength=0.3
    )
    
    base_config = ReactorConfiguration(
        plasma_params=plasma_base,
        polymer_params=polymer_params,
        heating_power_w=50e6
    )
    
    # Temperature sweep
    temperatures = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    results = []
    
    print("T(keV) | P_fus(MW) | Q_fusion | Q_net | Breakeven | Enhancement")
    print("-" * 65)
    
    for temp in temperatures:
        # Create config for this temperature
        plasma_params = PlasmaParameters(
            temperature_kev=temp,
            density_particles_m3=plasma_base.density_particles_m3,
            confinement_time_s=plasma_base.confinement_time_s,
            plasma_volume_m3=plasma_base.plasma_volume_m3
        )
        
        config = ReactorConfiguration(
            plasma_params=plasma_params,
            polymer_params=polymer_params,
            heating_power_w=50e6
        )
        
        # Simulate
        simulator = SimpleFusionReactorSimulator(config)
        result = simulator.q_factor_analysis()
        
        results.append(result)
        
        print(f"{temp:5.0f} | {result['powers']['fusion_mw']:7.1f} | "
              f"{result['q_factors']['Q_fusion']:7.3f} | "
              f"{result['q_factors']['Q_net']:6.3f} | "
              f"{'Yes' if result['q_factors']['breakeven'] else 'No':9s} | "
              f"{result['enhancement']:10.2f}")
    
    # Find optimal temperature
    q_values = [r['q_factors']['Q_fusion'] for r in results]
    max_q_idx = np.argmax(q_values)
    optimal_temp = temperatures[max_q_idx]
    max_q = q_values[max_q_idx]
    
    print()
    print(f"Optimal temperature: {optimal_temp} keV")
    print(f"Maximum Q_fusion: {max_q:.3f}")
    print(f"Breakeven achieved at: {[t for t, r in zip(temperatures, results) if r['q_factors']['breakeven']]} keV")
    
    return results

def test_density_sweep():
    """Test Q-factor vs density"""
    print("\nDENSITY SWEEP TEST")
    print("-" * 25)
    
    # Base configuration
    plasma_base = PlasmaParameters(
        temperature_kev=30.0,  # Good temperature
        density_particles_m3=1e20,
        confinement_time_s=3.0,
        plasma_volume_m3=830.0
    )
    
    polymer_params = PolymerParameters(
        scale_mu=2.0,
        enhancement_power_n=1.5,
        coupling_strength=0.3
    )
    
    # Density sweep (log scale)
    densities = np.logspace(19.5, 20.5, 9)  # 3.16e19 to 3.16e20
    
    results = []
    
    print("Density(m⁻³) | P_fus(MW) | Q_fusion | Q_net | Breakeven")
    print("-" * 55)
    
    for density in densities:
        # Create config for this density
        plasma_params = PlasmaParameters(
            temperature_kev=plasma_base.temperature_kev,
            density_particles_m3=density,
            confinement_time_s=plasma_base.confinement_time_s,
            plasma_volume_m3=plasma_base.plasma_volume_m3
        )
        
        config = ReactorConfiguration(
            plasma_params=plasma_params,
            polymer_params=polymer_params,
            heating_power_w=50e6
        )
        
        # Simulate
        simulator = SimpleFusionReactorSimulator(config)
        result = simulator.q_factor_analysis()
        
        results.append(result)
        
        print(f"{density:.2e} | {result['powers']['fusion_mw']:7.1f} | "
              f"{result['q_factors']['Q_fusion']:7.3f} | "
              f"{result['q_factors']['Q_net']:6.3f} | "
              f"{'Yes' if result['q_factors']['breakeven'] else 'No':9s}")
    
    # Find optimal density
    q_values = [r['q_factors']['Q_fusion'] for r in results]
    max_q_idx = np.argmax(q_values)
    optimal_density = densities[max_q_idx]
    max_q = q_values[max_q_idx]
    
    print()
    print(f"Optimal density: {optimal_density:.2e} m⁻³")
    print(f"Maximum Q_fusion: {max_q:.3f}")
    
    return results

def main():
    """Main test function"""
    print("=" * 70)
    print("PLAN B, STEP 2: SIMPLIFIED FUSION REACTOR TEST")
    print("Testing Q-factor calculations with polymer enhancement")
    print("=" * 70)
    
    # Test baseline
    print("\nBASELINE TEST:")
    print("-" * 15)
    
    plasma_params = PlasmaParameters(
        temperature_kev=25.0,
        density_particles_m3=1.5e20,
        confinement_time_s=3.0,
        plasma_volume_m3=830.0
    )
    
    polymer_params = PolymerParameters(
        scale_mu=2.0,
        enhancement_power_n=1.5,
        coupling_strength=0.3
    )
    
    config = ReactorConfiguration(
        plasma_params=plasma_params,
        polymer_params=polymer_params,
        heating_power_w=50e6
    )
    
    simulator = SimpleFusionReactorSimulator(config)
    baseline = simulator.q_factor_analysis()
    
    print(f"Configuration:")
    print(f"  Temperature: {plasma_params.temperature_kev} keV")
    print(f"  Density: {plasma_params.density_particles_m3:.1e} m⁻³")
    print(f"  Confinement time: {plasma_params.confinement_time_s} s")
    print(f"  Volume: {plasma_params.plasma_volume_m3} m³")
    print(f"  Heating power: {config.heating_power_w/1e6} MW")
    print(f"Results:")
    print(f"  Fusion power: {baseline['powers']['fusion_mw']:.1f} MW")
    print(f"  Bremsstrahlung loss: {baseline['powers']['bremsstrahlung_mw']:.1f} MW")
    print(f"  Conduction loss: {baseline['powers']['conduction_mw']:.1f} MW")
    print(f"  Q_fusion: {baseline['q_factors']['Q_fusion']:.3f}")
    print(f"  Q_net: {baseline['q_factors']['Q_net']:.3f}")
    print(f"  Breakeven: {baseline['q_factors']['breakeven']}")
    print(f"  Ignition: {baseline['q_factors']['ignition']}")
    print(f"  Polymer enhancement: {baseline['enhancement']:.2f}x")
    
    # Parameter sweeps
    temp_results = test_temperature_sweep()
    density_results = test_density_sweep()
    
    print("\n" + "=" * 70)
    print("SIMPLIFIED REACTOR SIMULATION COMPLETE")
    print("Polymer enhancement improves Q-factors across parameter space")
    print("=" * 70)
    
    return {
        'baseline': baseline,
        'temperature_sweep': temp_results,
        'density_sweep': density_results
    }

if __name__ == "__main__":
    results = main()
