#!/usr/bin/env python3
"""
Debug fusion power calculation issues step by step
"""

import numpy as np
import matplotlib.pyplot as plt
from integrated_gut_polymer_optimization import (
    PolymerParameters, GUTPolymerCrossSectionEngine, 
    ReactorPhysicsSimulator, ReactorDesignParameters, ConverterParameters
)

def test_cross_section_calculations():
    """Test cross-section calculations step by step"""
    
    print("DEBUGGING FUSION CROSS-SECTION CALCULATIONS")
    print("="*60)
    
    # Create polymer engine
    polymer_params = PolymerParameters(scale_mu=1.0)
    engine = GUTPolymerCrossSectionEngine(polymer_params)
    
    # Test energies (keV)
    test_energies = [1, 5, 10, 15, 20, 30, 50]
    
    print("Energy (keV) | Classical σ (barns) | Enhanced σ (barns) | Enhancement Factor")
    print("-" * 75)
    
    for E_keV in test_energies:
        try:
            # Classical cross-section
            classical_sigma = engine.classical_fusion_cross_section(E_keV, "D-T")
            
            # Enhanced cross-section
            enhanced_sigma = engine.polymer_enhanced_cross_section(E_keV, "keV", "fusion")
            
            # Enhancement factor
            if classical_sigma > 0:
                enhancement = enhanced_sigma / classical_sigma
            else:
                enhancement = 0.0
            
            print(f"{E_keV:8.1f}     | {classical_sigma:15.3e}     | {enhanced_sigma:14.3e}      | {enhancement:12.3f}")
            
        except Exception as e:
            print(f"{E_keV:8.1f}     | ERROR: {e}")
    
    print()
    
    # Test GUT polymer enhancement at different energies
    print("GUT POLYMER ENHANCEMENT FACTORS:")
    print("-" * 40)
    
    for E_keV in test_energies:
        E_GeV = E_keV * 1e-6  # Convert to GeV
        enhancement = engine.gut_polymer_sinc_enhancement(E_GeV)
        print(f"E = {E_keV:5.1f} keV ({E_GeV:.1e} GeV): enhancement = {enhancement:.6f}")
    
    return True

def test_fusion_power_density():
    """Test fusion power density calculation"""
    
    print("\nDEBUGGING FUSION POWER DENSITY CALCULATION")
    print("="*50)
    
    # Create test setup
    polymer_params = PolymerParameters(scale_mu=1.0)
    engine = GUTPolymerCrossSectionEngine(polymer_params)
    
    reactor_params = ReactorDesignParameters(plasma_volume_m3=830.0)
    converter_params = ConverterParameters(thermal_efficiency=0.45)
    reactor_sim = ReactorPhysicsSimulator(reactor_params, converter_params)
    
    # Test conditions
    T_keV = 20.0
    n_m3 = 1e20
    
    print(f"Test conditions: T = {T_keV} keV, n = {n_m3:.1e} m⁻³")
    
    try:
        # Test the fusion power density calculation
        power_density = reactor_sim.fusion_power_density(T_keV, n_m3, engine)
        print(f"Fusion power density: {power_density:.3e} W/m³")
        
        # Test individual components
        print("\nBreaking down the calculation:")
        
        # Test rate coefficient calculation (replicate internal logic)
        def debug_rate_coefficient(T_keV):
            print(f"  Rate coefficient for T = {T_keV} keV:")
            
            if T_keV < 2:
                base_rate = 0.0
                print(f"    Temperature too low: base_rate = {base_rate}")
            elif T_keV < 10:
                base_rate = 1e-27 * (T_keV / 5)**4
                print(f"    Low T regime: base_rate = {base_rate:.3e} m³/s")
            elif T_keV < 30:
                base_rate = 1e-25 * (T_keV / 15)**2
                print(f"    Medium T regime: base_rate = {base_rate:.3e} m³/s")
            else:
                base_rate = 5e-25 * (T_keV / 30)**0.5
                print(f"    High T regime: base_rate = {base_rate:.3e} m³/s")
            
            # Test enhancement calculation
            enhancement_sum = 0
            print(f"    Testing enhancement at different energies:")
            for E_test in [0.5 * T_keV, T_keV, 2 * T_keV]:
                if E_test > 0.1:
                    classical_sigma = engine.classical_fusion_cross_section(E_test)
                    enhanced_sigma = engine.polymer_enhanced_cross_section(E_test, "keV", "fusion")
                    
                    if classical_sigma > 0:
                        local_enhancement = enhanced_sigma / classical_sigma
                        enhancement_sum += local_enhancement
                        print(f"      E = {E_test:5.1f} keV: σ_classical = {classical_sigma:.3e} barns, σ_enhanced = {enhanced_sigma:.3e} barns, factor = {local_enhancement:.6f}")
                    else:
                        print(f"      E = {E_test:5.1f} keV: σ_classical = 0 (skipped)")
            
            avg_enhancement = enhancement_sum / 3 if enhancement_sum > 0 else 1.0
            enhanced_rate = base_rate * avg_enhancement
            
            print(f"    Average enhancement factor: {avg_enhancement:.6f}")
            print(f"    Enhanced rate coefficient: {enhanced_rate:.3e} m³/s")
            
            return enhanced_rate
        
        sigma_v = debug_rate_coefficient(T_keV)
        
        # Test power calculation
        n_d = n_t = n_m3 / 2  # 50/50 D-T mix
        fusion_energy_j = 17.59e6 * 1.602e-19  # D-T fusion energy in Joules
        
        print(f"\n  Power density calculation:")
        print(f"    n_D = n_T = {n_d:.1e} m⁻³")
        print(f"    <σv> = {sigma_v:.3e} m³/s")
        print(f"    Fusion energy = {fusion_energy_j:.3e} J")
        
        calculated_power_density = n_d * n_t * sigma_v * fusion_energy_j
        print(f"    Power density = n_D × n_T × <σv> × E_fusion = {calculated_power_density:.3e} W/m³")
        
        # Total power
        total_power_mw = calculated_power_density * 830.0 / 1e6
        print(f"    Total power = {total_power_mw:.3f} MW")
        
        return calculated_power_density
        
    except Exception as e:
        print(f"Error in fusion power calculation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def test_simplified_fusion_calculation():
    """Test a simplified, more direct fusion calculation"""
    
    print("\nTESTING SIMPLIFIED FUSION CALCULATION")
    print("="*45)
    
    # Test parameters
    T_keV = 20.0
    n_m3 = 1e20
    
    # Create polymer engine
    polymer_params = PolymerParameters(scale_mu=1.0)
    engine = GUTPolymerCrossSectionEngine(polymer_params)
    
    # Direct calculation approach
    print(f"Direct calculation approach:")
    print(f"  T = {T_keV} keV, n = {n_m3:.1e} m⁻³")
    
    # Get enhanced cross-section at plasma temperature
    classical_sigma_barns = engine.classical_fusion_cross_section(T_keV, "D-T")
    enhanced_sigma_barns = engine.polymer_enhanced_cross_section(T_keV, "keV", "fusion")
    
    print(f"  Classical σ(T) = {classical_sigma_barns:.3e} barns")
    print(f"  Enhanced σ(T) = {enhanced_sigma_barns:.3e} barns")
    
    if classical_sigma_barns > 0:
        enhancement_factor = enhanced_sigma_barns / classical_sigma_barns
        print(f"  Enhancement factor = {enhancement_factor:.6f}")
    else:
        enhancement_factor = 1.0
        print(f"  Enhancement factor = {enhancement_factor:.6f} (classical σ = 0)")
    
    # Convert cross-section to m²
    sigma_m2 = enhanced_sigma_barns * 1e-28  # barns to m²
    
    # Thermal-averaged reaction rate (approximation)
    v_thermal = np.sqrt(8 * 1.381e-23 * T_keV * 1000 * 1.602e-19 / (np.pi * 2.5 * 1.67e-27))  # m/s
    rate_coefficient = sigma_m2 * v_thermal  # m³/s
    
    print(f"  Cross-section = {sigma_m2:.3e} m²")
    print(f"  Thermal velocity ≈ {v_thermal:.3e} m/s")
    print(f"  Rate coefficient ≈ {rate_coefficient:.3e} m³/s")
    
    # Reaction rate and power
    n_d = n_t = n_m3 / 2  # 50/50 D-T mix
    reaction_rate = n_d * n_t * rate_coefficient  # reactions/m³/s
    fusion_energy_j = 17.59e6 * 1.602e-19  # J per reaction
    
    power_density = reaction_rate * fusion_energy_j  # W/m³
    total_power_mw = power_density * 830.0 / 1e6  # MW
    
    print(f"  Reaction rate = {reaction_rate:.3e} reactions/m³/s")
    print(f"  Power density = {power_density:.3e} W/m³")
    print(f"  Total power = {total_power_mw:.3f} MW")
    
    return total_power_mw

def compare_different_mu_values():
    """Compare calculations for different μ values"""
    
    print("\nCOMPARING DIFFERENT μ VALUES")
    print("="*35)
    
    mu_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    T_keV = 20.0
    
    print("μ       | Enhancement | Power (MW)")
    print("-" * 35)
    
    for mu in mu_values:
        polymer_params = PolymerParameters(scale_mu=mu)
        engine = GUTPolymerCrossSectionEngine(polymer_params)
        
        # Test enhancement
        enhancement = engine.gut_polymer_sinc_enhancement(T_keV * 1e-6)  # Convert keV to GeV
        
        # Test power (simplified)
        enhanced_sigma = engine.polymer_enhanced_cross_section(T_keV, "keV", "fusion")
        sigma_m2 = enhanced_sigma * 1e-28
        
        v_thermal = np.sqrt(8 * 1.381e-23 * T_keV * 1000 * 1.602e-19 / (np.pi * 2.5 * 1.67e-27))
        rate_coeff = sigma_m2 * v_thermal
        
        n_m3 = 1e20
        n_d = n_t = n_m3 / 2
        reaction_rate = n_d * n_t * rate_coeff
        fusion_energy_j = 17.59e6 * 1.602e-19
        power_density = reaction_rate * fusion_energy_j
        total_power_mw = power_density * 830.0 / 1e6
        
        print(f"{mu:6.1f}  | {enhancement:10.6f}  | {total_power_mw:8.3f}")

def main():
    """Main debugging function"""
    
    print("FUSION POWER CALCULATION DEBUG")
    print("="*70)
    
    # Test 1: Cross-section calculations
    test_cross_section_calculations()
    
    # Test 2: Power density calculation
    test_fusion_power_density()
    
    # Test 3: Simplified calculation
    test_simplified_fusion_calculation()
    
    # Test 4: Different μ values
    compare_different_mu_values()
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE")

if __name__ == "__main__":
    main()
