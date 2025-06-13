#!/usr/bin/env python3
"""
Debug the fusion optimization issues
"""

import numpy as np
import matplotlib.pyplot as plt
from integrated_gut_polymer_optimization import (
    PolymerParameters, ReactorDesignParameters, ConverterParameters, 
    EconomicParameters, GUTPolymerCrossSectionEngine, ReactorPhysicsSimulator,
    EconomicOptimizer
)

def test_fusion_calculations():
    """Test basic fusion power calculations"""
    
    print("Testing basic fusion calculations...")
    
    # Create test polymer parameters
    polymer_params = PolymerParameters(scale_mu=1.0)
    engine = GUTPolymerCrossSectionEngine(polymer_params)
    
    # Create reactor simulator
    reactor_params = ReactorDesignParameters(
        reactor_type="tokamak",
        plasma_volume_m3=830.0,
        availability_factor=0.85
    )
    
    converter_params = ConverterParameters(
        converter_type="thermal",
        thermal_efficiency=0.45,
        parasitic_losses=0.15
    )
    
    reactor_sim = ReactorPhysicsSimulator(reactor_params, converter_params)
    
    # Test conditions
    T_keV = 20.0  # Temperature
    n_m3 = 1e20   # Density
    
    try:
        # Test cross-section calculation
        sigma_enhanced = engine.enhanced_fusion_cross_section(T_keV)
        print(f"Enhanced D-T cross-section at {T_keV} keV: {sigma_enhanced:.2e} barns")
        
        # Test power density calculation
        power_density = reactor_sim.fusion_power_density(T_keV, n_m3, engine)
        print(f"Fusion power density: {power_density:.2e} W/m³")
        
        # Test total power
        total_power_mw = power_density * reactor_params.plasma_volume_m3 / 1e6
        print(f"Total fusion power: {total_power_mw:.2f} MW")
        
        # Test losses
        P_brems = 5.35e-37 * n_m3 * n_m3 * np.sqrt(T_keV) * reactor_params.plasma_volume_m3 / 1e6
        print(f"Bremsstrahlung losses: {P_brems:.2f} MW")
        
        tau_E = 2.0  # Confinement time
        P_conduction = 3 * n_m3 * 1.381e-23 * T_keV * 1000 * 1.602e-19 * reactor_params.plasma_volume_m3 / (tau_E * 1e6)
        print(f"Conduction losses: {P_conduction:.2f} MW")
        
        net_power_mw = total_power_mw - P_brems - P_conduction
        print(f"Net power: {net_power_mw:.2f} MW")
        
        # Test efficiency
        efficiency = reactor_sim.power_conversion_efficiency("fusion")
        print(f"Conversion efficiency: {efficiency:.3f}")
        
        electrical_power_mw = net_power_mw * efficiency
        print(f"Electrical power: {electrical_power_mw:.2f} MW")
        
        # Test LCOE calculation
        economic_params = EconomicParameters(
            capital_cost_b_usd=25.0,
            capacity_factor=0.85
        )
        
        optimizer = EconomicOptimizer(economic_params)
        
        if electrical_power_mw > 0:
            lcoe = optimizer.levelized_cost_of_energy(reactor_sim, electrical_power_mw)
            print(f"LCOE: ${lcoe:.2f}/MWh")
            print(f"Cost per kWh: ${lcoe/1000:.3f}/kWh")
        else:
            print("Negative electrical power - cannot calculate LCOE")
        
        return True
        
    except Exception as e:
        print(f"Error in fusion calculations: {e}")
        import traceback
        traceback.print_exc()
        return False

def scan_parameter_space():
    """Scan parameter space to find viable regions"""
    
    print("\nScanning parameter space...")
    
    # Create test setup
    polymer_params = PolymerParameters(scale_mu=1.0)
    engine = GUTPolymerCrossSectionEngine(polymer_params)
    
    reactor_params = ReactorDesignParameters(plasma_volume_m3=830.0)
    converter_params = ConverterParameters(thermal_efficiency=0.45, parasitic_losses=0.15)
    reactor_sim = ReactorPhysicsSimulator(reactor_params, converter_params)
    
    # Parameter ranges
    T_range = np.linspace(10, 50, 10)  # keV
    n_range = np.logspace(19.5, 20.5, 10)  # m^-3
    
    viable_points = []
    
    for T in T_range:
        for n in n_range:
            try:
                # Calculate power
                power_density = reactor_sim.fusion_power_density(T, n, engine)
                total_power_mw = power_density * 830.0 / 1e6
                
                # Simple losses
                P_brems = 5.35e-37 * n * n * np.sqrt(T) * 830.0 / 1e6
                tau_E = 2.0  # Fixed confinement time
                P_conduction = 3 * n * 1.381e-23 * T * 1000 * 1.602e-19 * 830.0 / (tau_E * 1e6)
                
                net_power_mw = total_power_mw - P_brems - P_conduction
                
                if net_power_mw > 0:
                    Q_factor = total_power_mw / 50.0  # Assume 50 MW heating
                    viable_points.append((T, n, net_power_mw, Q_factor))
                    
            except Exception as e:
                continue
    
    print(f"Found {len(viable_points)} viable operating points")
    
    if viable_points:
        print("Best operating points:")
        # Sort by Q-factor
        viable_points.sort(key=lambda x: x[3], reverse=True)
        for i, (T, n, P_net, Q) in enumerate(viable_points[:5]):
            print(f"  {i+1}: T={T:.1f} keV, n={n:.1e} m⁻³, P_net={P_net:.1f} MW, Q={Q:.2f}")
    
    return viable_points

def create_simplified_fusion_optimizer():
    """Create a simplified fusion optimization based on viable parameter space"""
    
    print("\nCreating simplified fusion optimization...")
    
    def simplified_fusion_optimization(mu):
        """Simplified fusion optimization for a given μ"""
        
        # Create polymer-enhanced engine
        polymer_params = PolymerParameters(scale_mu=mu)
        engine = GUTPolymerCrossSectionEngine(polymer_params)
        
        # Get enhancement factor
        enhancement = engine.gut_polymer_sinc_enhancement(1.0)  # At ~GeV energy
        
        # Base fusion parameters (ITER-like)
        T_base = 20.0  # keV
        n_base = 1e20   # m^-3
        tau_base = 3.0  # s
        
        # Enhanced parameters (simple scaling)
        T_opt = T_base * (1 + 0.1 * (enhancement - 1))  # Modest temperature increase
        n_opt = n_base * enhancement**0.5  # Density scales with sqrt(enhancement)
        tau_opt = tau_base * enhancement**0.3  # Confinement improves modestly
        
        # Calculate power with enhancement
        sigma_base = 4.6e-28  # D-T cross-section in m² at 20 keV
        sigma_enhanced = sigma_base * enhancement
        
        # Fusion power (simplified)
        reaction_rate = 0.25 * n_opt * n_opt * sigma_enhanced * 2.99e8  # reactions/m³/s
        energy_per_reaction = 17.6e6 * 1.602e-19  # J per D-T reaction
        power_density = reaction_rate * energy_per_reaction  # W/m³
        
        total_power_mw = power_density * 830.0 / 1e6  # MW
        
        # Losses (simplified)
        P_brems = 5.35e-37 * n_opt * n_opt * np.sqrt(T_opt) * 830.0 / 1e6
        P_conduction = 3 * n_opt * 1.381e-23 * T_opt * 1000 * 1.602e-19 * 830.0 / (tau_opt * 1e6)
        
        net_power_mw = total_power_mw - P_brems - P_conduction
        
        if net_power_mw > 0:
            # Q-factor
            P_heating = 50.0  # MW
            Q_factor = total_power_mw / P_heating
            
            # Electrical power
            efficiency = 0.45 * (1 - 0.15)  # Thermal efficiency minus parasitic losses
            electrical_power_mw = net_power_mw * efficiency
            
            # Simple cost model
            capital_cost = 25e9  # $25B
            annual_energy = electrical_power_mw * 8760 * 0.85  # MWh/year
            
            if annual_energy > 0:
                lcoe = (capital_cost * 0.1) / (annual_energy * 1000)  # $/kWh (simplified)
                
                return {
                    'success': True,
                    'polymer_scale_mu': mu,
                    'optimal_conditions': {
                        'temperature_kev': T_opt,
                        'density_m3': n_opt,
                        'confinement_time_s': tau_opt,
                        'cross_section_enhancement': enhancement
                    },
                    'performance': {
                        'fusion_power_mw': total_power_mw,
                        'net_power_mw': net_power_mw,
                        'electrical_power_mw': electrical_power_mw,
                        'q_factor': Q_factor
                    },
                    'economics': {
                        'cost_per_kwh': lcoe
                    }
                }
        
        # Failed optimization
        return {
            'success': False,
            'polymer_scale_mu': mu,
            'error': 'No net power gain',
            'economics': {'cost_per_kwh': 10.0}
        }
    
    # Test the simplified optimizer
    mu_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("Testing simplified optimizer:")
    for mu in mu_values:
        result = simplified_fusion_optimization(mu)
        if result['success']:
            print(f"μ={mu:.1f}: Q={result['performance']['q_factor']:.2f}, "
                  f"cost=${result['economics']['cost_per_kwh']:.3f}/kWh")
        else:
            print(f"μ={mu:.1f}: {result['error']}")
    
    return simplified_fusion_optimization

if __name__ == "__main__":
    print("DEBUGGING FUSION OPTIMIZATION")
    print("="*50)
    
    # Test basic calculations
    if test_fusion_calculations():
        print("✓ Basic calculations working")
    else:
        print("✗ Basic calculations failed")
    
    # Scan parameter space
    viable_points = scan_parameter_space()
    
    # Create simplified optimizer
    simplified_optimizer = create_simplified_fusion_optimizer()
    
    print("\nDebugging complete.")
