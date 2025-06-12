"""
Demonstration of polymerized Feynman rules for GUT theories.

This script shows how to numerically compute the polymerized propagator
and vertex form factors for different GUT groups.
"""

import numpy as np
import matplotlib.pyplot as plt
from unified_gut_polymerization.core import UnifiedGaugePolymerization, GUTConfig

def compute_classical_propagator(k_squared, mass_squared=0):
    """Compute the classical propagator (scalar part)."""
    return 1 / (k_squared + mass_squared)

def compute_polymerized_propagator(k_squared, mu, mass_squared=0):
    """Compute the polymerized propagator (scalar part)."""
    argument = mu * np.sqrt(k_squared + mass_squared)
    prefactor = 1 / mu**2
    propagator = prefactor * (np.sin(argument)**2 / (k_squared + mass_squared))
    return propagator

def compute_vertex_form_factor(momenta, mu):
    """Compute the polymerized vertex form factor."""
    form_factor = 1.0
    for p in momenta:
        p_mag = np.sqrt(np.sum(p**2))
        if p_mag > 1e-10:  # Avoid division by zero
            form_factor *= np.sin(mu * p_mag) / (mu * p_mag)
    return form_factor

def plot_propagator_comparison():
    """Plot comparison of classical and polymerized propagators."""
    k_values = np.linspace(0.01, 10, 1000)
    mu = 1.0  # Polymer scale
    
    classical_prop = [compute_classical_propagator(k**2) for k in k_values]
    poly_prop = [compute_polymerized_propagator(k**2, mu) for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, classical_prop, 'b-', label='Classical')
    plt.plot(k_values, poly_prop, 'r-', label='Polymerized')
    plt.xlabel('Momentum |k|')
    plt.ylabel('Propagator (scalar part)')
    plt.title('Comparison of Classical and Polymerized Propagators')
    plt.legend()
    plt.grid(True)
    plt.savefig('propagator_comparison.png', dpi=300)
    plt.close()
    
    # Log scale view
    plt.figure(figsize=(10, 6))
    plt.loglog(k_values, classical_prop, 'b-', label='Classical')
    plt.loglog(k_values, poly_prop, 'r-', label='Polymerized')
    plt.xlabel('Momentum |k| (log scale)')
    plt.ylabel('Propagator (log scale)')
    plt.title('Comparison of Classical and Polymerized Propagators (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('propagator_comparison_log.png', dpi=300)
    plt.close()
    
def plot_vertex_form_factors():
    """Plot vertex form factors for different momenta."""
    p_values = np.linspace(0.01, 10, 100)
    mu_values = [0.5, 1.0, 2.0]
    
    plt.figure(figsize=(10, 6))
    
    for mu in mu_values:
        form_factors = []
        for p in p_values:
            # Create a simple 3-point vertex with equal momenta
            momenta = [np.array([p, 0, 0, 0]), np.array([0, p, 0, 0]), np.array([0, 0, p, 0])]
            form_factors.append(compute_vertex_form_factor(momenta, mu))
        
        plt.plot(p_values, form_factors, label=f'μ = {mu}')
    
    plt.xlabel('Momentum Magnitude |p|')
    plt.ylabel('Vertex Form Factor')
    plt.title('Polymerized Vertex Form Factors for Different μ Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('vertex_form_factors.png', dpi=300)
    plt.close()
    
def demonstrate_gut_specific_features():
    """Demonstrate GUT-specific features of the polymerization."""
    gut_groups = ['SU5', 'SO10', 'E6']
    ranks = {'SU5': 4, 'SO10': 5, 'E6': 6}
    dimensions = {'SU5': 24, 'SO10': 45, 'E6': 78}
    
    print("\nGUT Group Specific Polymerization Features:")
    print("-----------------------------------------")
    for group in gut_groups:
        print(f"\nGroup: {group}")
        print(f"Rank: {ranks[group]}")
        print(f"Dimension (number of gauge bosons): {dimensions[group]}")
        
        # Create a simple example of polymerized form factor
        mu = 1.0
        p = 2.0
        momenta = [np.array([p, 0, 0, 0]), np.array([0, p, 0, 0]), np.array([0, 0, p, 0])]
        form_factor = compute_vertex_form_factor(momenta, mu)
        
        print(f"3-point vertex form factor (p={p}, μ={mu}): {form_factor:.6f}")
        
    # Check classical limit
    print("\nClassical Limit Verification:")
    print("--------------------------")
    p = 2.0
    momenta = [np.array([p, 0, 0, 0]), np.array([0, p, 0, 0]), np.array([0, 0, p, 0])]
    
    mu_values = [1.0, 0.1, 0.01, 0.001]
    print(f"\nVertex form factor as μ → 0 (p={p}):")
    for mu in mu_values:
        form_factor = compute_vertex_form_factor(momenta, mu)
        print(f"μ = {mu:.3f}: {form_factor:.6f}")
        
    print("\nPropagator as μ → 0 (k={p}):")
    for mu in mu_values:
        prop_poly = compute_polymerized_propagator(p**2, mu)
        prop_classical = compute_classical_propagator(p**2)
        print(f"μ = {mu:.3f}: {prop_poly:.6f} (classical: {prop_classical:.6f}, ratio: {prop_poly/prop_classical:.6f})")
    
if __name__ == "__main__":
    print("Unified Polymerized Feynman Rules - Numerical Demonstration")
    print("=======================================================")
    
    plot_propagator_comparison()
    print("\nGenerated propagator comparison plots.")
    
    plot_vertex_form_factors()
    print("Generated vertex form factor plots.")
    
    demonstrate_gut_specific_features()
    print("\nDemonstration complete.")
