"""
Running coupling and instanton effects demonstration for GUT polymerization.

This script visualizes the running coupling and instanton rates for different
GUT groups and polymer scales.
"""

import numpy as np
import matplotlib.pyplot as plt
from unified_gut_polymerization.running_coupling import RunningCouplingInstanton

def plot_running_coupling():
    """Plot running coupling for different GUT groups."""
    plt.figure(figsize=(10, 6))
    
    # Energy range from electroweak to GUT scale (GeV)
    energies = np.logspace(3, 16, 100)
    
    # Plot for each GUT group
    groups = ['SU5', 'SO10', 'E6']
    colors = ['blue', 'red', 'green']
    
    for group, color in zip(groups, colors):
        calculator = RunningCouplingInstanton(group)
        couplings = calculator.running_coupling_vs_energy(energies)
        
        plt.loglog(energies, couplings, label=f"{group}", color=color, linewidth=2)
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Energy (GeV)', fontsize=14)
    plt.ylabel('Effective Coupling $\\alpha_{\\rm eff}$', fontsize=14)
    plt.title('Running Coupling Comparison for GUT Groups', fontsize=16)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('running_coupling_comparison.png', dpi=300)
    plt.close()
    
def plot_instanton_effects():
    """Plot instanton rate vs. polymer scale for different GUT groups."""
    plt.figure(figsize=(10, 6))
    
    # Polymer scale values
    mu_values = np.linspace(0.01, 5, 100)
    
    # Plot for each GUT group
    groups = ['SU5', 'SO10', 'E6']
    colors = ['blue', 'red', 'green']
    
    for group, color in zip(groups, colors):
        calculator = RunningCouplingInstanton(group)
        energy = calculator.constants['gut_scale']
        
        # Compute rates relative to classical case
        rates = calculator.instanton_rate_vs_polymer(mu_values, energy)
        classical_rate = calculator.instanton_rate(calculator.running_coupling(energy), 0.01)
        relative_rates = rates / classical_rate
        
        plt.semilogy(mu_values, relative_rates, label=f"{group}", color=color, linewidth=2)
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Polymer Scale $\\mu$', fontsize=14)
    plt.ylabel('Relative Instanton Rate $\\Gamma/\\Gamma_{\\rm class}$', fontsize=14)
    plt.title('Polymer-Modified Instanton Rates', fontsize=16)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('instanton_rate_vs_polymer.png', dpi=300)
    plt.close()
    
def plot_polymer_modified_action():
    """Plot polymer-modified instanton action vs. polymer scale."""
    plt.figure(figsize=(10, 6))
    
    # Polymer scale values
    mu_values = np.linspace(0.01, 10, 100)
    
    # Plot sinc function modification factor
    modification = np.sin(mu_values) / mu_values
    plt.plot(mu_values, modification, 'k-', label='$\\sin(\\mu)/\\mu$', linewidth=2)
    
    # For each GUT group, plot the modified action ratio
    groups = ['SU5', 'SO10', 'E6']
    colors = ['blue', 'red', 'green']
    
    for group, color in zip(groups, colors):
        calculator = RunningCouplingInstanton(group)
        energy = calculator.constants['gut_scale']
        coupling = calculator.running_coupling(energy)
        
        # Compute modified action relative to classical action
        classical_action = calculator.instanton_action_classical(coupling)
        modified_actions = [calculator.instanton_action_polymer(coupling, mu) / classical_action 
                           for mu in mu_values]
        
        plt.plot(mu_values, modified_actions, color=color, label=f"{group}", linewidth=2, linestyle='--')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Polymer Scale $\\mu$', fontsize=14)
    plt.ylabel('Modified Action Ratio $S_{\\rm poly}/S_{\\rm class}$', fontsize=14)
    plt.title('Polymer Modification to Instanton Action', fontsize=16)
    plt.legend(fontsize=12)
    
    # Highlight transitions
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=np.pi, color='gray', linestyle='--', alpha=0.5, 
                label='$\\mu = \\pi$ (sign flip)')
    
    plt.tight_layout()
    plt.savefig('polymer_action_modification.png', dpi=300)
    plt.close()
    
def plot_beta_function_comparison():
    """Plot beta function coefficients for different GUT groups."""
    plt.figure(figsize=(8, 6))
    
    groups = ['SU5', 'SO10', 'E6']
    beta_coefs = []
    
    for group in groups:
        calculator = RunningCouplingInstanton(group)
        beta_coefs.append(calculator.constants['beta_coef'])
    
    plt.bar(groups, beta_coefs, color=['blue', 'red', 'green'])
    plt.grid(True, axis='y', alpha=0.3)
    plt.xlabel('GUT Group', fontsize=14)
    plt.ylabel('One-Loop $\\beta$-Function Coefficient $b_G$', fontsize=14) 
    plt.title('Comparison of $\\beta$-Function Coefficients', fontsize=16)
    
    # Add exact values as text on bars
    for i, v in enumerate(beta_coefs):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('beta_coefficient_comparison.png', dpi=300)
    plt.close()
    
if __name__ == "__main__":
    print("Generating plots for running coupling and instanton effects...")
    
    plot_running_coupling()
    print("- Running coupling plot created.")
    
    plot_instanton_effects()
    print("- Instanton rate plot created.")
    
    plot_polymer_modified_action()
    print("- Polymer action modification plot created.")
    
    plot_beta_function_comparison()
    print("- Beta function comparison plot created.")
    
    print("\nAll plots have been generated successfully.")
