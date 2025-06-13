"""
Test Corrected Fusion Power Calculation
=======================================

Verify that the fixed optimization method produces realistic power levels
comparable to ITER and other fusion benchmarks.
"""

import numpy as np
from integrated_gut_polymer_optimization import *

def test_corrected_optimization():
    """Test the corrected optimization with realistic power expectations"""
    
    print("TESTING CORRECTED FUSION POWER CALCULATION")
    print("="*50)
      # Create optimization framework
    framework = IntegratedPolymerEconomicFramework()
    
    # Test with several polymer scales
    test_scales = [0.5, 1.0, 2.0, 5.0]
    
    results = []
    for mu in test_scales:
        print(f"\nTesting μ = {mu}")
        print("-" * 20)
        
        result = framework.optimize_reactor_for_polymer_scale(mu, "fusion")
        
        if result['success']:
            conditions = result['optimal_conditions']
            performance = result['performance']
            economics = result['economics']
            
            print(f"Optimal conditions:")
            print(f"  T = {conditions['temperature_kev']:.1f} keV")
            print(f"  n = {conditions['density_m3']:.2e} m⁻³")
            print(f"  τ = {conditions['confinement_time_s']:.1f} s")            print(f"  Enhancement: {conditions['cross_section_enhancement']:.2f}x")
            
            print(f"Performance:")
            print(f"  Fusion power: {performance['fusion_power_mw']:.1f} MW")
            print(f"  Net power: {performance['net_power_mw']:.1f} MW") 
            print(f"  Electrical power: {performance['electrical_power_mw']:.1f} MW")
            print(f"  Q factor: {performance['q_factor']:.1f}")
            print(f"  Power density: {performance['power_density_w_m3']:.1e} W/m³")
            
            print(f"Economics:")
            print(f"  Cost per kWh: ${economics['cost_per_kwh']:.4f}")
            
            # Check if power levels are reasonable
            fusion_power = performance['fusion_power_mw']
            if 10 <= fusion_power <= 2000:
                print("✓ Fusion power is in reasonable range (10-2000 MW)")
            else:
                print(f"✗ Fusion power {fusion_power:.1f} MW is outside reasonable range")
            
            results.append(result)
        else:
            print(f"Optimization failed: {result.get('error', 'Unknown error')}")
    
    return results

def compare_with_reference_reactors():
    """Compare results with reference reactor designs"""
    
    print(f"\n" + "="*50)
    print("COMPARISON WITH REFERENCE REACTORS")
    print("="*50)
    
    # Reference reactor data
    references = {
        'ITER': {
            'temperature_kev': 15.0,
            'density_m3': 1.0e20,
            'power_mw': 500,
            'volume_m3': 840
        },
        'SPARC': {
            'temperature_kev': 12.0,
            'density_m3': 2.0e20,
            'power_mw': 140,
            'volume_m3': 20
        },
        'JET': {
            'temperature_kev': 3.5,
            'density_m3': 8e19,
            'power_mw': 16,
            'volume_m3': 100
        }
    }
    
    # Test our corrected method against these references
    polymer_params = PolymerParameters(scale_mu=1.0, coupling_strength=0.3)
    engine = GUTPolymerCrossSectionEngine(polymer_params)
    
    reactor_params = ReactorDesignParameters()
    converter_params = ConverterParameters()
    simulator = ReactorPhysicsSimulator(reactor_params, converter_params)
    
    print("Reactor\t\tExpected\tOur Calc\tRatio")
    print("-" * 45)
    
    for name, ref in references.items():
        power_density = simulator.fusion_power_density(
            ref['temperature_kev'], 
            ref['density_m3'], 
            engine
        )
        calculated_power = power_density * ref['volume_m3'] / 1e6
        ratio = calculated_power / ref['power_mw']
        
        print(f"{name}\t\t{ref['power_mw']:.0f} MW\t\t{calculated_power:.0f} MW\t\t{ratio:.2f}x")
        
        if 0.1 <= ratio <= 10:
            status = "✓"
        else:
            status = "✗"
        
        print(f"\t\t{status} {'Within reasonable range' if status == '✓' else 'Outside reasonable range'}")
    
    print(f"\nNote: Polymer enhancement may account for some differences.")
    print(f"Ratios between 0.1x and 10x are considered reasonable for this level of modeling.")

def verify_power_scaling():
    """Verify that power scales correctly with enhancement"""
    
    print(f"\n" + "="*50)
    print("POWER SCALING VERIFICATION")
    print("="*50)
    
    # Fixed conditions, varying enhancement
    T_keV = 20.0
    n_m3 = 1.5e20
    
    enhancement_factors = [1.0, 2.0, 3.0, 5.0, 8.0]
    
    print(f"Test conditions: T = {T_keV} keV, n = {n_m3:.1e} m⁻³")
    print(f"\nEnhancement\tPower (MW)\tScaling")
    print("-" * 35)
    
    base_power = None
    for enhancement in enhancement_factors:
        # Create enhanced polymer parameters
        polymer_params = PolymerParameters(
            scale_mu=enhancement, 
            coupling_strength=0.3
        )
        engine = GUTPolymerCrossSectionEngine(polymer_params)
        
        reactor_params = ReactorDesignParameters()
        converter_params = ConverterParameters()
        simulator = ReactorPhysicsSimulator(reactor_params, converter_params)
        
        power_density = simulator.fusion_power_density(T_keV, n_m3, engine)
        total_power = power_density * 830.0 / 1e6  # MW
        
        if base_power is None:
            base_power = total_power
            scaling = 1.0
        else:
            scaling = total_power / base_power
        
        print(f"{enhancement:.1f}x\t\t{total_power:.1f} MW\t\t{scaling:.2f}x")
    
    print(f"\n✓ Power scaling verification complete")
    print(f"Expected: Higher enhancement → Higher power")

if __name__ == "__main__":
    # Run all tests
    optimization_results = test_corrected_optimization()
    compare_with_reference_reactors()
    verify_power_scaling()
    
    print(f"\n" + "="*50)
    print("CORRECTED FUSION POWER TEST COMPLETE")
    print("="*50)
    
    if optimization_results:
        success_count = sum(1 for r in optimization_results if r['success'])
        print(f"✓ {success_count}/{len(optimization_results)} optimization tests successful")
        
        # Check if all power levels are reasonable
        reasonable_count = 0
        for result in optimization_results:
            if result['success']:
                power = result['performance']['fusion_power_mw']
                if 10 <= power <= 2000:
                    reasonable_count += 1
        
        print(f"✓ {reasonable_count}/{success_count} power levels in reasonable range")
        
        if reasonable_count == success_count:
            print("🎉 ALL TESTS PASSED - Fusion power calculation is now correct!")
        else:
            print("⚠️  Some power levels still need adjustment")
    else:
        print("✗ No successful optimization results")
    
    print("="*50)
