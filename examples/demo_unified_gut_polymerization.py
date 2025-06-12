#!/usr/bin/env python3
"""
Demo script for the Unified GUT Polymerization framework.

This script demonstrates the capabilities of the GUT polymerization framework,
performing parameter sweeps and generating visualizations for different
symmetry groups and polymerization configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

# Add the parent directory to the path to import the package
sys.path.append(str(Path(__file__).parent.parent))

from unified_gut_polymerization import GUTConfig, UnifiedGaugePolymerization


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo for GUT Polymerization Framework")
    
    parser.add_argument("--group", type=str, default="SU5", choices=["SU5", "SO10", "E6"],
                        help="GUT gauge group to use")
    
    parser.add_argument("--polymer-scale", type=float, default=1.0e19,
                        help="Polymer energy scale in GeV")
    
    parser.add_argument("--polymer-length", type=float, default=1.0,
                        help="Dimensionless polymer length parameter")
    
    parser.add_argument("--energy-min", type=float, default=1.0e12,
                        help="Minimum energy for scans in GeV")
    
    parser.add_argument("--energy-max", type=float, default=1.0e20,
                        help="Maximum energy for scans in GeV")
    
    parser.add_argument("--scan-points", type=int, default=100,
                        help="Number of points in energy scans")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for plot (without extension)")
    
    parser.add_argument("--compare-groups", action="store_true",
                        help="Compare different gauge groups")
    
    parser.add_argument("--sweep-parameter", action="store_true",
                        help="Perform parameter sweep across polymerization parameter")
    
    return parser.parse_args()


def compare_gauge_groups(polymer_scale, polymer_length, e_min, e_max, points, output=None):
    """Compare effects across different gauge groups."""
    groups = ["SU5", "SO10", "E6"]
    colors = ["b", "r", "g"]
    
    # Create energy array
    energies = np.logspace(np.log10(e_min), np.log10(e_max), points)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Comparison of Polymer Effects Across Gauge Groups', fontsize=16)
    
    for group, color in zip(groups, colors):
        # Create configuration and model
        config = GUTConfig(
            group=group,
            polymer_scale=polymer_scale,
            polymer_length=polymer_length
        )
        model = UnifiedGaugePolymerization(config)
        
        # Compute phenomenology
        results = model.compute_phenomenology(energies)
        
        # Plot running coupling
        axes[0, 0].loglog(energies, results['coupling'], f'{color}-', label=group)
        
        # Plot cross section enhancement
        axes[0, 1].loglog(energies, results['cross_section_factor'], f'{color}-', label=group)
        
        # Plot propagator modification
        axes[1, 0].loglog(energies, results['propagator_modifications'], f'{color}-', label=group)
        
        # Plot threshold corrections
        axes[1, 1].semilogx(energies, results['threshold_corrections'], f'{color}-', label=group)
    
    # Add labels and grid
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        axes[i, j].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[i, j].legend()
        
    axes[0, 0].set_xlabel('Energy (GeV)')
    axes[0, 0].set_ylabel('Coupling α')
    axes[0, 0].set_title('Running Coupling')
    
    axes[0, 1].set_xlabel('Energy (GeV)')
    axes[0, 1].set_ylabel('Enhancement Factor')
    axes[0, 1].set_title('Cross Section Enhancement')
    
    axes[1, 0].set_xlabel('Energy (GeV)')
    axes[1, 0].set_ylabel('Modification Factor')
    axes[1, 0].set_title('Propagator Modification')
    
    axes[1, 1].set_xlabel('Energy (GeV)')
    axes[1, 1].set_ylabel('Correction')
    axes[1, 1].set_title('Threshold Corrections')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if output:
        plt.savefig(f"{output}_group_comparison.png", dpi=300)
    
    return fig


def parameter_sweep(group, polymer_scale, e_min, e_max, points, output=None):
    """Perform a parameter sweep across polymer length values."""
    polymer_lengths = [0.1, 0.5, 1.0, 2.0, 5.0]
    colors = ["b", "g", "r", "m", "c"]
    
    # Create energy array
    energies = np.logspace(np.log10(e_min), np.log10(e_max), points)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Parameter Sweep: {group} with Different Polymer Lengths', fontsize=16)
    
    for length, color in zip(polymer_lengths, colors):
        # Create configuration and model
        config = GUTConfig(
            group=group,
            polymer_scale=polymer_scale,
            polymer_length=length
        )
        model = UnifiedGaugePolymerization(config)
        
        # Compute phenomenology
        results = model.compute_phenomenology(energies)
        
        # Plot running coupling
        axes[0, 0].loglog(energies, results['coupling'], f'{color}-', label=f'λ = {length}')
        
        # Plot cross section enhancement
        axes[0, 1].loglog(energies, results['cross_section_factor'], f'{color}-', label=f'λ = {length}')
        
        # Plot propagator modification
        axes[1, 0].loglog(energies, results['propagator_modifications'], f'{color}-', label=f'λ = {length}')
        
        # Plot threshold corrections
        axes[1, 1].semilogx(energies, results['threshold_corrections'], f'{color}-', label=f'λ = {length}')
    
    # Add labels and grid
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        axes[i, j].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[i, j].legend()
        
    axes[0, 0].set_xlabel('Energy (GeV)')
    axes[0, 0].set_ylabel('Coupling α')
    axes[0, 0].set_title('Running Coupling')
    
    axes[0, 1].set_xlabel('Energy (GeV)')
    axes[0, 1].set_ylabel('Enhancement Factor')
    axes[0, 1].set_title('Cross Section Enhancement')
    
    axes[1, 0].set_xlabel('Energy (GeV)')
    axes[1, 0].set_ylabel('Modification Factor')
    axes[1, 0].set_title('Propagator Modification')
    
    axes[1, 1].set_xlabel('Energy (GeV)')
    axes[1, 1].set_ylabel('Correction')
    axes[1, 1].set_title('Threshold Corrections')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if output:
        plt.savefig(f"{output}_parameter_sweep.png", dpi=300)
    
    return fig


def main():
    """Main function to run the demo."""
    args = parse_arguments()
    
    # Print information about the run
    print(f"GUT Polymerization Demo")
    print(f"=======================")
    print(f"Group: {args.group}")
    print(f"Polymer scale: {args.polymer_scale:.2e} GeV")
    print(f"Polymer length: {args.polymer_length}")
    print(f"Energy range: {args.energy_min:.2e} - {args.energy_max:.2e} GeV")
    print()
    
    if args.compare_groups:
        print("Comparing different gauge groups...")
        fig = compare_gauge_groups(
            args.polymer_scale,
            args.polymer_length,
            args.energy_min,
            args.energy_max,
            args.scan_points,
            args.output
        )
    elif args.sweep_parameter:
        print(f"Performing parameter sweep for {args.group}...")
        fig = parameter_sweep(
            args.group,
            args.polymer_scale,
            args.energy_min,
            args.energy_max,
            args.scan_points,
            args.output
        )
    else:
        # Default: just show the effects for the specified group
        config = GUTConfig(
            group=args.group,
            polymer_scale=args.polymer_scale,
            polymer_length=args.polymer_length
        )
        model = UnifiedGaugePolymerization(config)
        
        print(f"Computing polymer effects...")
        fig = model.plot_polymer_effects(
            args.energy_min,
            args.energy_max,
            args.scan_points
        )
        
        if args.output:
            plt.savefig(f"{args.output}.png", dpi=300)
            print(f"Plot saved to {args.output}.png")
    
    plt.show()


if __name__ == "__main__":
    main()
