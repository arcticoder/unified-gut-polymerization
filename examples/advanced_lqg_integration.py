#!/usr/bin/env python3
"""
Advanced example demonstrating integration with existing LQG frameworks.

This script shows how to use the GUT unified polymerization framework 
in conjunction with the unified-lqg package for more complex scenarios.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging

# Add the parent directory to the path to import the package
sys.path.append(str(Path(__file__).parent.parent))

# Import our GUT polymerization package
from unified_gut_polymerization import GUTConfig, UnifiedGaugePolymerization

# Try to import from unified-lqg package (if available)
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / 'unified-lqg'))
    from unified_lqg import LQGConfig, LQGFramework
    HAS_UNIFIED_LQG = True
except ImportError:
    logging.warning("Could not import unified-lqg package. Running in standalone mode.")
    HAS_UNIFIED_LQG = False
    
    # Create stub classes for compatibility
    class LQGConfig:
        def __init__(self, **kwargs):
            self.immirzi = kwargs.get('immirzi', 0.2375)
            self.discretization = kwargs.get('discretization', 'improved')
            self.spin_foam_model = kwargs.get('spin_foam_model', 'EPRL')
    
    class LQGFramework:
        def __init__(self, config):
            self.config = config
            
        def compute_spectrum(self, operator, cutoff):
            # Dummy implementation
            return np.linspace(0, 10, cutoff) ** 2


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced GUT Polymerization Example")
    
    parser.add_argument("--group", type=str, default="SU5", choices=["SU5", "SO10", "E6"],
                        help="GUT gauge group to use")
    
    parser.add_argument("--polymer-scale", type=float, default=1.0e19,
                        help="Polymer energy scale in GeV")
    
    parser.add_argument("--polymer-length", type=float, default=1.0,
                        help="Dimensionless polymer length parameter")
    
    parser.add_argument("--immirzi", type=float, default=0.2375,
                        help="Immirzi parameter for LQG integration")
                        
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for plot (without extension)")
                        
    return parser.parse_args()


def setup_integrated_framework(args):
    """Set up the integrated GUT-LQG framework."""
    # Create GUT configuration
    gut_config = GUTConfig(
        group=args.group,
        polymer_scale=args.polymer_scale,
        polymer_length=args.polymer_length,
    )
    
    # Create LQG configuration
    lqg_config = LQGConfig(
        immirzi=args.immirzi,
        discretization='improved',
        spin_foam_model='EPRL'
    )
    
    # Initialize both frameworks
    gut_model = UnifiedGaugePolymerization(gut_config)
    lqg_model = LQGFramework(lqg_config)
    
    return gut_model, lqg_model


def compute_combined_predictions(gut_model, lqg_model, energies):
    """Compute predictions from combined frameworks."""
    # Get GUT predictions
    gut_results = gut_model.compute_phenomenology(energies)
    
    # Create combined results dictionary
    results = {
        'energies': energies,
        'gut_coupling': gut_results['coupling'],
        'gut_cross_section': gut_results['cross_section_factor'],
        'gut_propagator': gut_results['propagator_modifications'],
    }
    
    # Add LQG-related predictions if available
    if HAS_UNIFIED_LQG:
        # Calculate area spectrum (up to some maximum eigenvalue)
        area_spectrum = lqg_model.compute_spectrum('area', cutoff=20)
        
        # Calculate the minimal area (first non-zero eigenvalue)
        min_area = min([a for a in area_spectrum if a > 0])
        
        # Relate minimal area to polymer length
        # This connects the LQG area gap to the polymer discreteness parameter
        polymer_length_from_lqg = np.sqrt(min_area) / gut_model.config.polymer_scale
        
        # Calculate modified predictions with LQG-derived polymer length
        lqg_gut_config = GUTConfig(
            group=gut_model.config.group,
            polymer_scale=gut_model.config.polymer_scale,
            polymer_length=polymer_length_from_lqg
        )
        lqg_gut_model = UnifiedGaugePolymerization(lqg_gut_config)
        lqg_gut_results = lqg_gut_model.compute_phenomenology(energies)
        
        # Add LQG-integrated results
        results['lqg_polymer_length'] = polymer_length_from_lqg
        results['lqg_gut_coupling'] = lqg_gut_results['coupling']
        results['lqg_gut_cross_section'] = lqg_gut_results['cross_section_factor']
    
    return results


def plot_combined_results(results, args):
    """Plot the results from the combined framework."""
    energies = results['energies']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(f'GUT-LQG Integration: {args.group} with LQG-Derived Parameters', fontsize=16)
    
    # Plot running coupling
    axes[0].loglog(energies, results['gut_coupling'], 'b-', label=f'{args.group} (λ={args.polymer_length})')
    
    if 'lqg_gut_coupling' in results:
        axes[0].loglog(energies, results['lqg_gut_coupling'], 'r--', 
                      label=f'{args.group} (LQG λ={results["lqg_polymer_length"]:.4f})')
    
    axes[0].set_xlabel('Energy (GeV)')
    axes[0].set_ylabel('Coupling α')
    axes[0].set_title('Running Coupling with Polymer Effects')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[0].legend()
    
    # Plot cross section enhancement
    axes[1].loglog(energies, results['gut_cross_section'], 'b-', label=f'{args.group} (λ={args.polymer_length})')
    
    if 'lqg_gut_cross_section' in results:
        axes[1].loglog(energies, results['lqg_gut_cross_section'], 'r--', 
                      label=f'{args.group} (LQG λ={results["lqg_polymer_length"]:.4f})')
    
    axes[1].set_xlabel('Energy (GeV)')
    axes[1].set_ylabel('Enhancement Factor')
    axes[1].set_title('Cross Section Enhancement')
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if args.output:
        plt.savefig(f"{args.output}_lqg_integrated.png", dpi=300)
        print(f"Plot saved to {args.output}_lqg_integrated.png")
    
    return fig


def main():
    """Main function to run the demo."""
    args = parse_arguments()
    
    # Print information about the run
    print(f"GUT-LQG Integration Demo")
    print(f"=======================")
    print(f"Group: {args.group}")
    print(f"Polymer scale: {args.polymer_scale:.2e} GeV")
    print(f"Polymer length: {args.polymer_length}")
    print(f"Immirzi parameter: {args.immirzi}")
    if HAS_UNIFIED_LQG:
        print(f"LQG framework: Available")
    else:
        print(f"LQG framework: Not available (running in standalone mode)")
    print()
    
    # Setup the integrated framework
    gut_model, lqg_model = setup_integrated_framework(args)
    
    # Compute predictions
    energies = np.logspace(12, 20, 100)  # 10^12 to 10^20 GeV
    results = compute_combined_predictions(gut_model, lqg_model, energies)
    
    # Plot results
    fig = plot_combined_results(results, args)
    
    plt.show()


if __name__ == "__main__":
    main()
