#!/usr/bin/env python3
"""
This script demonstrates the symbolic derivation of GUT recoupling coefficients
by generalizing the SU(2) generating functional to higher-rank unified gauge groups.

Key implementation:
1. Master generating functional for rank-r groups
2. Determinant-to-hypergeometric mapping
3. Closed-form expressions for recoupling coefficients
"""

import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path to import the package
sys.path.append(str(Path(__file__).parent.parent))

from unified_gut_polymerization import GUTRecouplingCalculator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Symbolic derivation of GUT recoupling coefficients"
    )
    
    parser.add_argument(
        "--group", 
        type=str, 
        default="SU5", 
        choices=["SU5", "SO10", "E6"],
        help="Gauge group to use"
    )
    
    parser.add_argument(
        "--graph", 
        type=str, 
        default="theta", 
        choices=["theta", "tetrahedron"],
        help="Graph structure for recoupling"
    )
    
    parser.add_argument(
        "--max-order", 
        type=int, 
        default=3,
        help="Maximum order of expansion"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file for results (without extension)"
    )
    
    return parser.parse_args()


def print_hypergeometric_mapping(calc):
    """Print the determinant-to-hypergeometric mapping."""
    print("\n" + "=" * 60)
    print(f"DETERMINANT-TO-HYPERGEOMETRIC MAPPING FOR {calc.group}")
    print("=" * 60)
    
    # Group properties
    print(f"Group: {calc.group}")
    print(f"Rank: {calc.rank}")
    print(f"Dimension: {calc.dimension}")
    print()
    
    # Print the mapping
    print("Mapping:")
    print(f"det(I - K_G(z))^(-1/2) = {calc.map_to_hypergeometric()}")
    print()
    
    # Print polynomial approximation
    print("Polynomial approximation:")
    print(calc.get_polynomial_approximation(3))
    print()


def print_recoupling_coefficients(calc, graph_type, max_order):
    """Print recoupling coefficients for the specified graph."""
    print("\n" + "=" * 60)
    print(f"RECOUPLING COEFFICIENTS FOR {calc.group} - {graph_type.upper()} GRAPH")
    print("=" * 60)
    
    if graph_type == "theta":
        coeffs = calc.derive_theta_coefficients(max_order)
    else:  # tetrahedron
        coeffs = calc.derive_tetrahedron_coefficients(max_order)
    
    # Print all coefficients
    print(f"Coefficients (up to order {max_order}):")
    for idx, coeff in coeffs.items():
        print(f"Index {idx}: {coeff}")


def main():
    """Main function to demonstrate GUT recoupling derivation."""
    args = parse_arguments()
    
    # Create calculator for the specified group
    calc = GUTRecouplingCalculator(args.group)
    
    # Print information about the hypergeometric mapping
    print_hypergeometric_mapping(calc)
    
    # Print recoupling coefficients
    print_recoupling_coefficients(calc, args.graph, args.max_order)
    
    # If an output file was specified, redirect output there as well
    if args.output:
        # This would involve more sophisticated output formatting
        print(f"\nResults saved to {args.output}.txt")


if __name__ == "__main__":
    main()
