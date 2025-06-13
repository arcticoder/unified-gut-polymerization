"""
Debug Reaction Rate Calculation Issues
=====================================

This script identifies and fixes the inconsistencies between the two power
calculation methods in the optimization framework.
"""

import numpy as np
from integrated_gut_polymer_optimization import *

def test_reaction_rate_calculations():
    """Test and compare the two different power calculation methods"""
    
    print("DEBUGGING REACTION RATE CALCULATION ISSUES")
    print("="*60)
    
    # Test conditions
    T_keV = 20.0
    n_m3 = 1.25e20  # Similar to optimization results
    
    # Create test objects
    polymer_params = PolymerParameters(scale_mu=1.0, coupling_strength=0.3)
    engine = GUTPolymerCrossSectionEngine(polymer_params)
    
    reactor_params = ReactorDesignParameters()
    converter_params = ConverterParameters()
    simulator = ReactorPhysicsSimulator(reactor_params, converter_params)
    
    print(f"Test conditions: T = {T_keV} keV, n = {n_m3:.2e} m⁻³")
    print()
    
    # Method 1: fusion_power_density (correct method)
    print("METHOD 1: fusion_power_density (thermal rate coefficient)")
    print("-" * 55)
    
    power_density_1 = simulator.fusion_power_density(T_keV, n_m3, engine)
    total_power_1 = power_density_1 * 830.0 / 1e6  # MW
    
    print(f"Power density: {power_density_1:.3e} W/m³")
    print(f"Total power: {total_power_1:.1f} MW")
    
    # Method 2: optimization method (problematic)
    print("\nMETHOD 2: optimization method (direct calculation)")
    print("-" * 50)
    
    classical_sigma = engine.classical_fusion_cross_section(T_keV)
    enhanced_sigma = engine.polymer_enhanced_cross_section(T_keV, "keV", "fusion")
    
    # Current (wrong) optimization calculation
    reaction_rate_wrong = 0.25 * n_m3 * n_m3 * enhanced_sigma * 1e-28 * 3e8
    energy_per_reaction = 17.6e6 * 1.602e-19
    power_density_2_wrong = reaction_rate_wrong * energy_per_reaction
    total_power_2_wrong = power_density_2_wrong * 830.0 / 1e6
    
    print(f"Enhanced σ: {enhanced_sigma:.3f} barns")
    print(f"Wrong reaction rate: {reaction_rate_wrong:.3e} reactions/m³/s")
    print(f"Wrong power density: {power_density_2_wrong:.3e} W/m³")
    print(f"Wrong total power: {total_power_2_wrong:.1f} MW")
    
    print(f"\nPower ratio (Method 2 / Method 1): {total_power_2_wrong / total_power_1:.1e}x")
    
    # Method 3: corrected optimization method
    print("\nMETHOD 3: corrected optimization method")
    print("-" * 42)
    
    # Correct reaction rate calculation
    # Rate = n_D * n_T * <σv> where <σv> is thermal rate coefficient
    
    # First, get the thermal rate coefficient properly
    def corrected_rate_coefficient(T_keV, enhancement_factor):
        """Calculate corrected thermal rate coefficient"""
        if T_keV < 2:
            return 0.0
        
        # Base thermal rate coefficient (realistic values)
        if T_keV < 20:
            sigma_v_base = 3.7e-22 * (T_keV / 10.0)**2 * np.exp(-19.94 / np.sqrt(T_keV))
        elif T_keV < 100:
            sigma_v_base = 1.1e-22 * (T_keV / 20.0)**1.5 * np.exp(-19.94 / np.sqrt(T_keV))
        else:
            sigma_v_base = 3e-22 * (T_keV / 100.0)**0.5
        
        return sigma_v_base * enhancement_factor
    
    enhancement_factor = enhanced_sigma / classical_sigma
    sigma_v_corrected = corrected_rate_coefficient(T_keV, enhancement_factor)
    
    n_d = n_t = n_m3 / 2  # 50/50 D-T mix
    reaction_rate_corrected = n_d * n_t * sigma_v_corrected  # reactions/m³/s
    power_density_3 = reaction_rate_corrected * energy_per_reaction  # W/m³
    total_power_3 = power_density_3 * 830.0 / 1e6  # MW
    
    print(f"Enhancement factor: {enhancement_factor:.3f}")
    print(f"Thermal rate coeff: {sigma_v_corrected:.3e} m³/s")
    print(f"Corrected reaction rate: {reaction_rate_corrected:.3e} reactions/m³/s")
    print(f"Corrected power density: {power_density_3:.3e} W/m³")
    print(f"Corrected total power: {total_power_3:.1f} MW")
    
    print(f"\nMethod 3 vs Method 1 ratio: {total_power_3 / total_power_1:.2f}x (should be ~1)")
    
    # Analysis
    print(f"\nANALYSIS:")
    print(f"• Method 1 (thermal): {total_power_1:.1f} MW - CORRECT")
    print(f"• Method 2 (wrong): {total_power_2_wrong:.0f} MW - TOO HIGH by {total_power_2_wrong/total_power_1:.0f}x")
    print(f"• Method 3 (corrected): {total_power_3:.1f} MW - SHOULD MATCH METHOD 1")
    
    # Identify the error sources
    print(f"\nERROR SOURCES IN METHOD 2:")
    print(f"1. Wrong units: enhanced_sigma in barns, then * 1e-28 * 3e8")
    print(f"2. Factor 3e8 (speed of light) inappropriate for reaction rates")
    print(f"3. Factor 0.25 is arbitrary and not physically motivated")
    print(f"4. Should use thermal rate coefficient, not direct cross-section")
    
    return {
        'method_1_power': total_power_1,
        'method_2_wrong_power': total_power_2_wrong,
        'method_3_corrected_power': total_power_3,
        'error_factor': total_power_2_wrong / total_power_1
    }

def compare_with_iter_expectations():
    """Compare results with ITER expectations"""
    
    print(f"\nCOMPARISON WITH ITER EXPECTATIONS")
    print("="*40)
    
    # ITER parameters (approximate)
    iter_conditions = {
        'temperature_kev': 15.0,  # Average T_i + T_e
        'density_m3': 1.0e20,
        'expected_power_mw': 500,  # Q=10 with 50 MW heating
        'plasma_volume_m3': 840
    }
    
    print(f"ITER conditions:")
    print(f"  Temperature: {iter_conditions['temperature_kev']} keV")
    print(f"  Density: {iter_conditions['density_m3']:.1e} m⁻³")
    print(f"  Expected power: {iter_conditions['expected_power_mw']} MW")
    print(f"  Plasma volume: {iter_conditions['plasma_volume_m3']} m³")
    
    # Calculate using corrected method
    polymer_params = PolymerParameters(scale_mu=1.0, coupling_strength=0.3)
    engine = GUTPolymerCrossSectionEngine(polymer_params)
    
    reactor_params = ReactorDesignParameters()
    converter_params = ConverterParameters()
    simulator = ReactorPhysicsSimulator(reactor_params, converter_params)
    
    iter_power_density = simulator.fusion_power_density(
        iter_conditions['temperature_kev'], 
        iter_conditions['density_m3'], 
        engine
    )
    iter_total_power = iter_power_density * iter_conditions['plasma_volume_m3'] / 1e6
    
    print(f"\nOur calculation:")
    print(f"  Power density: {iter_power_density:.1e} W/m³")
    print(f"  Total power: {iter_total_power:.1f} MW")
    print(f"  Ratio to ITER: {iter_total_power / iter_conditions['expected_power_mw']:.2f}x")
    
    # Expected power density for ITER
    iter_expected_density = iter_conditions['expected_power_mw'] * 1e6 / iter_conditions['plasma_volume_m3']
    print(f"  ITER expected density: {iter_expected_density:.1e} W/m³")
    
    if 0.1 <= iter_total_power / iter_conditions['expected_power_mw'] <= 10:
        print("✓ Result is within reasonable range of ITER expectations")
    else:
        print("✗ Result is outside reasonable range - needs adjustment")

if __name__ == "__main__":
    results = test_reaction_rate_calculations()
    compare_with_iter_expectations()
    
    print(f"\n" + "="*60)
    print("REACTION RATE DEBUG COMPLETE")
    print("="*60)
    print(f"Problem identified: Wrong formula in optimization method")
    print(f"Error factor: {results['error_factor']:.0f}x too high")
    print(f"Solution: Use thermal rate coefficient method consistently")
    print("="*60)
