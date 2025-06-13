"""
Debug cross-section calculations for Plan B Step 2
"""

import numpy as np
from plan_b_step1_corrected import PolymerParameters, PolymerCorrectedCrossSection, FUSION_REACTIONS

def debug_cross_sections():
    print("DEBUGGING CROSS-SECTION CALCULATIONS")
    print("=" * 50)
    
    # Create polymer parameters
    polymer_params = PolymerParameters(
        scale_mu=5.0,
        enhancement_power_n=2.0,
        coupling_strength=0.15
    )
    
    polymer_cross_section = PolymerCorrectedCrossSection(polymer_params)
    reaction = FUSION_REACTIONS['D-T']
    
    print(f"Reaction: {reaction.reaction_name}")
    print(f"Q-value: {reaction.q_value_mev} MeV")
    print(f"Coulomb barrier: {reaction.coulomb_barrier_kev} keV")
    print()
    
    # Test at various energies
    energies = [5, 10, 15, 20, 30, 50, 70, 100]  # keV
    
    print("Energy (keV) | Classical σ (barns) | Polymer σ (barns) | Enhancement | Sinc Factor")
    print("-" * 85)
    
    for E in energies:
        sigma_classical = polymer_cross_section.classical_cross_section(E, reaction)
        sigma_polymer = polymer_cross_section.polymer_corrected_cross_section(E, reaction)
        enhancement = sigma_polymer / sigma_classical if sigma_classical > 0 else 0
        sinc_factor = polymer_cross_section.sinc_enhancement_factor(E)
        
        print(f"{E:10.1f} | {sigma_classical:15.3e} | {sigma_polymer:13.3e} | {enhancement:10.2f} | {sinc_factor:10.3f}")
    
    print()
    
    # Test Maxwell-Boltzmann integration parameters
    print("MAXWELL-BOLTZMANN INTEGRATION TEST")
    print("-" * 40)
    
    temperature_kev = 20.0
    E_thermal = temperature_kev
    E_max = 10 * E_thermal
    E_min = 0.1
    
    print(f"Temperature: {temperature_kev} keV")
    print(f"Integration range: {E_min} to {E_max} keV")
    print(f"Thermal energy: {E_thermal} keV")
    
    # Test a few integration points
    test_energies = [0.1, 1, 5, 10, 20, 50, 100, 200]
    print()
    print("Energy (keV) | σ(E) (barns) | v_rel (m/s) | MB factor | Integrand")
    print("-" * 70)
    
    from scipy import constants
    
    for E in test_energies:
        if E <= E_max:
            sigma = polymer_cross_section.polymer_corrected_cross_section(E, reaction) * 1e-28  # barns to m²
            
            # Calculate relative velocity
            mass_1_kg = reaction.mass_1_amu * constants.atomic_mass
            mass_2_kg = reaction.mass_2_amu * constants.atomic_mass
            reduced_mass = (mass_1_kg * mass_2_kg) / (mass_1_kg + mass_2_kg)
            
            energy_j = E * 1000 * constants.eV
            v_rel = np.sqrt(2 * energy_j / reduced_mass)
            
            # Maxwell-Boltzmann factors
            mb_factor = np.sqrt(2 / np.pi) * np.sqrt(energy_j / (constants.k * temperature_kev * 1000))
            mb_exp = np.exp(-energy_j / (constants.k * temperature_kev * 1000))
            
            integrand = sigma * v_rel * mb_factor * mb_exp
            
            print(f"{E:10.1f} | {sigma/1e-28:11.3e} | {v_rel:9.2e} | {mb_factor*mb_exp:8.3e} | {integrand:8.3e}")

if __name__ == "__main__":
    debug_cross_sections()
