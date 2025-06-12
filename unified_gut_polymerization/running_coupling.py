"""
Running coupling and instanton effects in polymerized GUTs.

This module implements the formulas for calculating running couplings and
instanton rates in polymerized Grand Unified Theories (GUTs).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class RunningCouplingInstanton:
    """
    Class for calculating running coupling and instanton effects in
    polymerized Grand Unified Theories.
    """
    
    # GUT-specific constants
    GUT_CONSTANTS = {
        'SU5': {
            'dimension': 24,
            'rank': 4,
            'casimir': 5,
            'beta_coef': 16.17,
            'gut_scale': 1e16,  # GeV
            'lambda_scale': 1e14,  # GeV
            'alpha_gut': 1/25,
        },
        'SO10': {
            'dimension': 45,
            'rank': 5,
            'casimir': 8,
            'beta_coef': 27.17,
            'gut_scale': 2e16,  # GeV
            'lambda_scale': 5e13,  # GeV
            'alpha_gut': 1/24,
        },
        'E6': {
            'dimension': 78,
            'rank': 6,
            'casimir': 12,
            'beta_coef': 41.83,
            'gut_scale': 3e16,  # GeV
            'lambda_scale': 2e13,  # GeV
            'alpha_gut': 1/23,
        }
    }
    
    def __init__(self, group: str = 'SU5', n_f: int = 3, n_s: int = 1):
        """
        Initialize the calculator with a specific gauge group and matter content.
        
        Args:
            group: The gauge group ('SU5', 'SO10', or 'E6')
            n_f: Number of fermion generations
            n_s: Number of scalar fields
        """
        self.group = group
        self.n_f = n_f
        self.n_s = n_s
        
        if group not in self.GUT_CONSTANTS:
            raise ValueError(f"Unsupported gauge group: {group}")
            
        self.constants = self.GUT_CONSTANTS[group]
        
        # Recalculate beta coefficient if custom n_f or n_s is provided
        if n_f != 3 or n_s != 1:
            self.constants['beta_coef'] = self.calculate_beta_coefficient()
    
    def calculate_beta_coefficient(self) -> float:
        """
        Calculate the one-loop β-function coefficient based on the gauge group
        and matter content.
        
        Returns:
            The β-function coefficient b_G
        """
        c2 = self.constants['casimir']
        b_g = (11/3) * c2 - (2/3) * self.n_f - (1/6) * self.n_s
        return b_g
    
    def running_coupling(self, energy: float, reference_energy: Optional[float] = None,
                        reference_coupling: Optional[float] = None) -> float:
        """
        Calculate the running coupling at a given energy scale.
        
        Args:
            energy: The energy scale in GeV
            reference_energy: Reference energy scale in GeV (defaults to GUT scale)
            reference_coupling: Coupling at reference energy (defaults to GUT coupling)
            
        Returns:
            The coupling constant at the specified energy
        """
        if reference_energy is None:
            reference_energy = self.constants['gut_scale']
            
        if reference_coupling is None:
            reference_coupling = self.constants['alpha_gut']
            
        b_g = self.constants['beta_coef']
        denominator = 1 - (b_g / (2 * np.pi)) * reference_coupling * np.log(energy / reference_energy)
        
        return reference_coupling / denominator
    
    def instanton_action_classical(self, coupling: float) -> float:
        """
        Calculate the classical instanton action.
        
        Args:
            coupling: The gauge coupling constant
            
        Returns:
            The classical instanton action
        """
        return 8 * np.pi**2 / coupling
    
    def instanton_action_polymer(self, coupling: float, mu: float, phi_inst: float = 1.0) -> float:
        """
        Calculate the polymerized instanton action.
        
        Args:
            coupling: The gauge coupling constant
            mu: The polymer scale parameter
            phi_inst: The instanton topological charge (default = 1.0)
            
        Returns:
            The polymerized instanton action
        """
        if abs(mu * phi_inst) < 1e-10:
            # Use series expansion for small mu to avoid numerical issues
            polymer_factor = 1 - (mu * phi_inst)**2 / 6
        else:
            polymer_factor = np.sin(mu * phi_inst) / (mu * phi_inst)
            
        return self.instanton_action_classical(coupling) * polymer_factor
    
    def instanton_rate(self, coupling: float, mu: float, phi_inst: float = 1.0) -> float:
        """
        Calculate the polymerized instanton transition rate.
        
        Args:
            coupling: The gauge coupling constant
            mu: The polymer scale parameter
            phi_inst: The instanton topological charge (default = 1.0)
            
        Returns:
            The instanton transition rate
        """
        lambda_g = self.constants['lambda_scale']
        action = self.instanton_action_polymer(coupling, mu, phi_inst)
        
        return lambda_g**4 * np.exp(-action)
    
    def running_coupling_vs_energy(self, energies: List[float]) -> np.ndarray:
        """
        Calculate the running coupling across a range of energies.
        
        Args:
            energies: List of energy values in GeV
            
        Returns:
            Array of corresponding coupling values
        """
        return np.array([self.running_coupling(E) for E in energies])
    
    def instanton_rate_vs_polymer(self, mu_values: List[float], energy: float) -> np.ndarray:
        """
        Calculate instanton rates for different polymer scale values.
        
        Args:
            mu_values: List of polymer scale parameter values
            energy: The energy scale in GeV
            
        Returns:
            Array of instanton rates
        """
        coupling = self.running_coupling(energy)
        return np.array([self.instanton_rate(coupling, mu) for mu in mu_values])


def demonstrate_running_coupling():
    """Demonstrate the running coupling calculations for different GUT groups."""
    groups = ['SU5', 'SO10', 'E6']
    
    # Energy range from electroweak to GUT scale (GeV)
    energies = np.logspace(3, 16, 100)
    
    print("Running Coupling Demonstration:")
    print("==============================")
    
    for group in groups:
        calculator = RunningCouplingInstanton(group)
        couplings = calculator.running_coupling_vs_energy(energies)
        
        print(f"\n{group} Group:")
        print(f"  β-coefficient: {calculator.constants['beta_coef']:.2f}")
        print(f"  α(10^3 GeV): {calculator.running_coupling(1e3):.6f}")
        print(f"  α(10^6 GeV): {calculator.running_coupling(1e6):.6f}")
        print(f"  α(10^9 GeV): {calculator.running_coupling(1e9):.6f}")
        print(f"  α(10^12 GeV): {calculator.running_coupling(1e12):.6f}")
        print(f"  α(10^16 GeV): {calculator.running_coupling(1e16):.6f}")


def demonstrate_instanton_effects():
    """Demonstrate instanton rate calculations with polymer modifications."""
    # Use SU(5) as an example
    calculator = RunningCouplingInstanton('SU5')
    
    # Polymer scale values
    mu_values = np.linspace(0, 5, 20)
    
    # Energy at GUT scale
    energy = calculator.constants['gut_scale']
    coupling = calculator.running_coupling(energy)
    
    print("\nInstanton Effects Demonstration:")
    print("==============================")
    print(f"SU(5) Group at E = {energy:.1e} GeV:")
    print(f"  Coupling: α = {coupling:.6f}")
    
    # Classical instanton action
    s_classical = calculator.instanton_action_classical(coupling)
    print(f"  Classical instanton action: S = {s_classical:.2f}")
    
    # Polymer modified actions
    print("\n  Polymer modifications:")
    for mu in [0.1, 0.5, 1.0, 2.0]:
        s_poly = calculator.instanton_action_polymer(coupling, mu)
        enhancement = np.exp(s_classical - s_poly)
        print(f"  μ = {mu:.1f}: S_poly = {s_poly:.2f}, enhancement factor = {enhancement:.2e}")
    
    # Instanton rates
    rates = calculator.instanton_rate_vs_polymer(mu_values, energy)
    
    # Calculate relative rates normalized to classical case
    relative_rates = rates / calculator.instanton_rate(coupling, 1e-10)
    
    print("\n  Relative instanton rates:")
    for i, mu in enumerate(mu_values[1:10:2]):
        print(f"  μ = {mu:.2f}: Rate/Rate_classical = {relative_rates[i+1]:.2e}")


if __name__ == "__main__":
    demonstrate_running_coupling()
    demonstrate_instanton_effects()
