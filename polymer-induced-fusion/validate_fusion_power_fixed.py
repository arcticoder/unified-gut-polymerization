"""
Test Realistic Fusion Power Calculation
=======================================

The optimization is showing 1.5 million MW fusion power, which is unrealistic.
Let me debug the power calculation to find the issue.
"""

import numpy as np
from integrated_gut_polymer_optimization import *

# Test the fusion power calculation with reasonable parameters
def test_fusion_power():
    print("TESTING FUSION POWER CALCULATION")
    print("="*50)
    
    # Create test parameters
    polymer_params = PolymerParameters(
        scale_mu=1.0,
        enhancement_power_n=1.5,
        coupling_strength=0.3,
        gut_scale_gev=1e16
    )
    
    engine = GUTPolymerCrossSectionEngine(polymer_params)
    
    reactor_params = ReactorDesignParameters()
    converter_params = ConverterParameters()
    simulator = ReactorPhysicsSimulator(reactor_params, converter_params)
    
    # Test conditions (ITER-like)
    T_keV = 20.0  # keV
    n_m3 = 1e20   # m^-3
    
    # Calculate power density
    power_density = simulator.fusion_power_density(T_keV, n_m3, engine)
    
    print(f"Temperature: {T_keV} keV")
    print(f"Density: {n_m3:.1e} m^-3")
    print(f"Power density: {power_density:.3e} W/m³")
    
    # Check plasma volume and total power
    plasma_volume = 830.0  # m³ (ITER scale)
    total_power_mw = power_density * plasma_volume / 1e6
    print(f"Plasma volume: {plasma_volume} m³")
    print(f"Total fusion power: {total_power_mw:.1f} MW")
    
    # Compare to ITER expectations
    iter_expected_power = 500  # MW
    print(f"ITER expected: {iter_expected_power} MW")
    print(f"Ratio: {total_power_mw / iter_expected_power:.1f}x ITER")
    
    # Test cross-section at this temperature
    classical_sigma = engine.classical_fusion_cross_section(T_keV)
    enhanced_sigma = engine.polymer_enhanced_cross_section(T_keV, "keV", "fusion")
    enhancement = enhanced_sigma / classical_sigma if classical_sigma > 0 else 0
    
    print(f"\nCross-section analysis:")
    print(f"Classical σ: {classical_sigma:.3e} barns")
    print(f"Enhanced σ: {enhanced_sigma:.3e} barns")
    print(f"Enhancement: {enhancement:.2f}x")
    
    # Debug the rate coefficient
    print(f"\nRate coefficient analysis:")
    
    # Test the rate coefficient function directly
    def test_rate_coefficient(T_keV):
        if T_keV < 2:
            return 0.0
        
        if T_keV < 20:
            # Low temperature regime - corrected coefficients  
            sigma_v_base = 3.7e-22 * (T_keV / 10.0)**2 * np.exp(-19.94 / np.sqrt(T_keV))
        elif T_keV < 100:
            # Medium temperature regime
            sigma_v_base = 1.1e-22 * (T_keV / 20.0)**1.5 * np.exp(-19.94 / np.sqrt(T_keV))
        else:
            # High temperature regime
            sigma_v_base = 3e-22 * (T_keV / 100.0)**0.5
        
        print(f"Base rate coefficient: {sigma_v_base:.3e} m³/s")
        
        # Apply enhancement
        enhanced_rate = sigma_v_base * enhancement
        print(f"Enhanced rate coefficient: {enhanced_rate:.3e} m³/s")
        
        return enhanced_rate
    
    sigma_v = test_rate_coefficient(T_keV)
    
    # Check reaction rate
    n_d = n_t = n_m3 / 2
    reaction_rate = n_d * n_t * sigma_v  # reactions/m³/s
    print(f"Reaction rate: {reaction_rate:.3e} reactions/m³/s")
    
    # Check energy per reaction
    fusion_energy_j = 17.59e6 * 1.602e-19  # D-T fusion energy in J
    print(f"Energy per reaction: {fusion_energy_j:.3e} J")
    
    # Final power density check
    calculated_power_density = reaction_rate * fusion_energy_j
    print(f"Calculated power density: {calculated_power_density:.3e} W/m³")
    
    # Compare to known values
    # ITER at Q=10 should produce ~500 MW with similar conditions
    # Expected power density should be ~600 kW/m³
    expected_power_density = 6e5  # W/m³
    print(f"Expected power density: {expected_power_density:.3e} W/m³")
    print(f"Ratio: {calculated_power_density / expected_power_density:.1f}x expected")

if __name__ == "__main__":
    test_fusion_power()
