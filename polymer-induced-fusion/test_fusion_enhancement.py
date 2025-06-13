#!/usr/bin/env python3
"""
Test the new fusion-specific enhancement function
"""

import numpy as np
import matplotlib.pyplot as plt
from integrated_gut_polymer_optimization import (
    PolymerParameters, GUTPolymerCrossSectionEngine
)

def test_fusion_enhancement():
    """Test the fusion-specific enhancement function"""
    
    print("TESTING FUSION-SPECIFIC ENHANCEMENT")
    print("="*50)
    
    # Test different μ values
    mu_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    T_keV = 20.0  # Typical fusion temperature
    
    print(f"Testing at T = {T_keV} keV")
    print("μ       | Enhancement | σ_classical (barns) | σ_enhanced (barns)")
    print("-" * 70)
    
    for mu in mu_values:
        polymer_params = PolymerParameters(scale_mu=mu, coupling_strength=0.3)
        engine = GUTPolymerCrossSectionEngine(polymer_params)
        
        # Test fusion enhancement
        classical_sigma = engine.classical_fusion_cross_section(T_keV, "D-T")
        enhanced_sigma = engine.polymer_enhanced_cross_section(T_keV, "keV", "fusion")
        
        if classical_sigma > 0:
            enhancement = enhanced_sigma / classical_sigma
        else:
            enhancement = 1.0
        
        print(f"{mu:6.1f}  | {enhancement:10.3f}  | {classical_sigma:15.3e}     | {enhanced_sigma:15.3e}")

def test_enhancement_vs_energy():
    """Test enhancement as a function of energy"""
    
    print(f"\nTESTING ENHANCEMENT VS ENERGY")
    print("="*40)
    
    energies = np.linspace(1, 100, 20)  # keV
    mu_test = 1.0
    
    polymer_params = PolymerParameters(scale_mu=mu_test, coupling_strength=0.3)
    engine = GUTPolymerCrossSectionEngine(polymer_params)
    
    print("Energy (keV) | Enhancement Factor")
    print("-" * 35)
    
    enhancements = []
    for E in energies:
        classical_sigma = engine.classical_fusion_cross_section(E, "D-T")
        enhanced_sigma = engine.polymer_enhanced_cross_section(E, "keV", "fusion")
        
        if classical_sigma > 0:
            enhancement = enhanced_sigma / classical_sigma
        else:
            enhancement = 1.0
        
        enhancements.append(enhancement)
        if E in [5, 10, 20, 30, 50, 100]:
            print(f"{E:10.1f}   | {enhancement:12.3f}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(energies, enhancements, 'b-', linewidth=2, marker='o')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Enhancement Factor')
    plt.title(f'Fusion Enhancement vs Energy (μ = {mu_test})')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(1, max(enhancements) * 1.1)
    
    plot_file = "polymer_economic_optimization/fusion_enhancement_vs_energy.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Enhancement plot saved to: {plot_file}")

def test_complete_power_calculation():
    """Test complete power calculation with different μ values"""
    
    print(f"\nTESTING COMPLETE POWER CALCULATION")
    print("="*45)
    
    mu_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    T_keV = 20.0
    n_m3 = 1e20
    
    print("μ       | Enhancement | Power (MW) | Q-factor")
    print("-" * 50)
    
    for mu in mu_values:
        polymer_params = PolymerParameters(scale_mu=mu, coupling_strength=0.3)
        engine = GUTPolymerCrossSectionEngine(polymer_params)
        
        # Calculate enhanced cross-section
        enhanced_sigma_barns = engine.polymer_enhanced_cross_section(T_keV, "keV", "fusion")
        classical_sigma_barns = engine.classical_fusion_cross_section(T_keV, "D-T")
        
        if classical_sigma_barns > 0:
            enhancement = enhanced_sigma_barns / classical_sigma_barns
        else:
            enhancement = 1.0
        
        # Simplified power calculation
        sigma_m2 = enhanced_sigma_barns * 1e-28  # Convert to m²
        
        # Thermal-averaged velocity (approximation)
        k_B = 1.381e-23  # J/K
        T_joules = T_keV * 1000 * 1.602e-19  # Convert keV to J
        m_reduced = 1.25 * 1.67e-27  # kg (reduced mass for D-T)
        v_thermal = np.sqrt(8 * k_B * T_joules / (np.pi * m_reduced))
        
        # Rate coefficient
        rate_coeff = sigma_m2 * v_thermal  # m³/s
        
        # Power calculation
        n_d = n_t = n_m3 / 2  # 50/50 D-T mix
        reaction_rate = n_d * n_t * rate_coeff  # reactions/m³/s
        fusion_energy_j = 17.59e6 * 1.602e-19  # J per reaction
        
        power_density = reaction_rate * fusion_energy_j  # W/m³
        total_power_mw = power_density * 830.0 / 1e6  # MW (ITER volume)
        
        # Q-factor (assume 50 MW heating)
        P_heating = 50.0  # MW
        Q_factor = total_power_mw / P_heating if P_heating > 0 else 0
        
        print(f"{mu:6.1f}  | {enhancement:10.3f}  | {total_power_mw:8.3f}   | {Q_factor:7.3f}")

def test_parameter_sensitivity():
    """Test sensitivity to polymer parameters"""
    
    print(f"\nTESTING PARAMETER SENSITIVITY")
    print("="*40)
    
    # Base parameters
    mu_base = 1.0
    coupling_base = 0.3
    T_keV = 20.0
    
    print("Parameter Variation | Enhancement | Power (MW)")
    print("-" * 50)
    
    # Test different coupling strengths
    for coupling in [0.1, 0.3, 0.5, 1.0]:
        polymer_params = PolymerParameters(scale_mu=mu_base, coupling_strength=coupling)
        engine = GUTPolymerCrossSectionEngine(polymer_params)
        
        enhanced_sigma = engine.polymer_enhanced_cross_section(T_keV, "keV", "fusion")
        classical_sigma = engine.classical_fusion_cross_section(T_keV, "D-T")
        enhancement = enhanced_sigma / classical_sigma if classical_sigma > 0 else 1.0
        
        # Quick power estimate
        power_estimate = enhancement * 1.0  # Relative to base case
        
        print(f"coupling = {coupling:4.1f}      | {enhancement:10.3f}  | {power_estimate:8.3f}")

def main():
    """Main test function"""
    
    print("FUSION-SPECIFIC ENHANCEMENT TESTING")
    print("="*60)
    
    test_fusion_enhancement()
    test_enhancement_vs_energy()
    test_complete_power_calculation()
    test_parameter_sensitivity()
    
    print("\n" + "="*60)
    print("FUSION ENHANCEMENT TESTING COMPLETE")

if __name__ == "__main__":
    main()
