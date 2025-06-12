"""
High-resolution parameter scan demonstration for polymerized GUTs.

This script demonstrates how to perform scans across the parameter space
of polymer-modified GUT theories to identify optimal regions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from unified_gut_polymerization.parameter_scan import ParameterScanner, generate_tikz_contour, generate_3d_slices

def create_output_directory():
    """Create output directory for figures and TikZ files."""
    output_dir = "parameter_scan_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def run_2d_parameter_scan(group='SU5'):
    """
    Run a 2D parameter scan over μ and b, fixing Φ_inst.
    
    Args:
        group: GUT group ('SU5', 'SO10', or 'E6')
    """
    print(f"Running 2D parameter scan for {group}...")
    
    # Create parameter scanner
    scanner = ParameterScanner(group=group)
    
    # Define parameter ranges
    mu_range = (0.01, 5.0)  # Polymer scale
    b_range = (0.5, 5.0)    # Field strength parameter
    resolution = 100        # Grid resolution
    phi_inst = 1.0          # Fixed instanton topological charge
    
    # Run the scan
    results = scanner.scan_2d_parameter_space(
        mu_range=mu_range,
        b_range=b_range,
        resolution=resolution,
        phi_inst=phi_inst,
        E_field=1e16,  # V/m
        energy=1e12    # GeV
    )
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Generate contour plot for rate ratio
    generate_tikz_contour(
        data=results,
        filename=f"{output_dir}/{group}_rate_ratio",
        title=f"{group} Rate Ratio",
        levels=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    )
    
    # Generate contour plot for critical field ratio
    critical_field_data = {
        'mu_grid': results['mu_grid'],
        'b_grid': results['b_grid'],
        'R_rate': results['R_E_crit']
    }
    
    generate_tikz_contour(
        data=critical_field_data,
        filename=f"{output_dir}/{group}_critical_field",
        title=f"{group} Critical Field Ratio",
        levels=[0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        colormap='plasma'
    )
    
    print(f"Saved 2D scan results to {output_dir}/")
    
    # Generate additional visualizations
    plt.figure(figsize=(10, 8))
    
    # Plot contour of regions where R_rate > 1
    plt.contour(
        results['mu_grid'], 
        results['b_grid'], 
        results['R_rate'], 
        levels=[1.0], 
        colors='red', 
        linewidths=2,
        linestyles='dashed'
    )
    
    # Plot contour of regions where E_crit^poly < 1e17 V/m
    critical_field = results['R_E_crit'] * scanner.E_crit
    plt.contour(
        results['mu_grid'], 
        results['b_grid'], 
        critical_field, 
        levels=[1e17], 
        colors='blue', 
        linewidths=2,
        linestyles='dotted'
    )
    
    # Combine these regions to highlight the "inexpensive" parameter space
    combined_mask = (results['R_rate'] > 1.0) & (critical_field < 1e17)
    plt.contourf(
        results['mu_grid'], 
        results['b_grid'], 
        combined_mask.astype(float), 
        levels=[0.5, 1.5], 
        colors=['green'], 
        alpha=0.3
    )
    
    plt.xlabel('Polymer Scale $\\mu$', fontsize=14)
    plt.ylabel('Field Strength Parameter $b$', fontsize=14)
    plt.title(f'{group} "Inexpensive" Parameter Regions', fontsize=16)
    
    # Add legend
    plt.plot([], [], 'r--', linewidth=2, label='$R_{\\rm rate} > 1$')
    plt.plot([], [], 'b:', linewidth=2, label='$E_{\\rm crit}^{\\rm poly} < 10^{17}\\,\\mathrm{V/m}$')
    plt.fill([], [], 'green', alpha=0.3, label='Optimal Region')
    
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{group}_optimal_regions.png", dpi=300)
    plt.close()
    
    return results

def run_3d_parameter_scan(group='SU5'):
    """
    Run a 3D parameter scan over μ, b, and Φ_inst.
    
    Args:
        group: GUT group ('SU5', 'SO10', or 'E6')
    """
    print(f"Running 3D parameter scan for {group}...")
    
    # Create parameter scanner
    scanner = ParameterScanner(group=group)
    
    # Define parameter ranges
    mu_range = (0.01, 5.0)     # Polymer scale
    b_range = (0.5, 5.0)       # Field strength parameter
    phi_range = (0.1, 3.0)     # Instanton topological charge
    resolution = 20            # Grid resolution (lower for 3D to manage computation)
    
    # Run the scan
    results = scanner.scan_3d_parameter_space(
        mu_range=mu_range,
        b_range=b_range,
        phi_range=phi_range,
        resolution=resolution,
        E_field=1e16,  # V/m
        energy=1e12    # GeV
    )
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Generate 3D slice visualizations
    generate_3d_slices(
        data=results,
        filename_prefix=f"{output_dir}/{group}_3d_slice"
    )
    
    print(f"Saved 3D scan results to {output_dir}/")
    
    return results

def find_optimal_parameters(group='SU5'):
    """
    Find optimal parameters for maximizing rate enhancement while
    keeping critical field below constraints.
    
    Args:
        group: GUT group ('SU5', 'SO10', or 'E6')
    """
    print(f"Finding optimal parameters for {group}...")
    
    # Create parameter scanner
    scanner = ParameterScanner(group=group)
    
    # Define parameter ranges
    mu_range = (0.01, 10.0)    # Polymer scale
    b_range = (0.5, 10.0)      # Field strength parameter
    phi_range = (0.1, 5.0)     # Instanton topological charge
    
    # Find optimal parameters
    optimal = scanner.find_optimal_parameters(
        mu_range=mu_range,
        b_range=b_range,
        phi_range=phi_range,
        target_ratio=10.0,
        E_field_constraint=1e17,  # V/m
        energy=1e12               # GeV
    )
    
    # Print results
    print("\nOptimal Parameters:")
    print(f"  μ = {optimal['mu']:.4f}")
    print(f"  b = {optimal['b']:.4f}")
    print(f"  Φ_inst = {optimal['phi_inst']:.4f}")
    print(f"  Rate Ratio = {optimal['rate_ratio']:.2f}x")
    print(f"  Critical Field = {optimal['E_crit']:.2e} V/m")
    print(f"  Optimization Status: {optimal['message']}")
    
    return optimal

def generate_combined_tikz_file(groups=None):
    """
    Generate a combined TikZ file with side-by-side comparisons
    of different GUT groups.
    
    Args:
        groups: List of GUT groups to include
    """
    if groups is None:
        groups = ['SU5', 'SO10', 'E6']
        
    output_dir = create_output_directory()
    
    # Generate combined TikZ file
    tikz_code = "%% Combined Parameter Scan Visualization\n"
    tikz_code += "\\documentclass{article}\n"
    tikz_code += "\\usepackage{pgfplots}\n"
    tikz_code += "\\usepackage{tikz}\n"
    tikz_code += "\\pgfplotsset{compat=1.16}\n"
    tikz_code += "\\usepgfplotslibrary{colormaps}\n\n"
    tikz_code += "\\begin{document}\n\n"
    
    # Add title
    tikz_code += "\\section*{High-Resolution Parameter Scans for Polymerized GUTs}\n\n"
    
    # Rate ratio comparison
    tikz_code += "\\subsection*{Rate Ratio Comparison}\n\n"
    tikz_code += "\\begin{figure}[htb]\n"
    tikz_code += "\\centering\n"
    
    for i, group in enumerate(groups):
        tikz_code += f"\\begin{{minipage}}[b]{{0.32\\textwidth}}\n"
        tikz_code += f"\\input{{{group}_rate_ratio.tex}}\n"
        tikz_code += f"\\caption*{{{group}}}\n"
        tikz_code += f"\\end{{minipage}}\n"
        if i < len(groups) - 1:
            tikz_code += "\\hfill\n"
            
    tikz_code += "\\caption{Comparison of rate ratio $R_{\\rm rate}(\\mu, b)$ across GUT groups.}\n"
    tikz_code += "\\end{figure}\n\n"
    
    # Critical field comparison
    tikz_code += "\\subsection*{Critical Field Comparison}\n\n"
    tikz_code += "\\begin{figure}[htb]\n"
    tikz_code += "\\centering\n"
    
    for i, group in enumerate(groups):
        tikz_code += f"\\begin{{minipage}}[b]{{0.32\\textwidth}}\n"
        tikz_code += f"\\input{{{group}_critical_field.tex}}\n"
        tikz_code += f"\\caption*{{{group}}}\n"
        tikz_code += f"\\end{{minipage}}\n"
        if i < len(groups) - 1:
            tikz_code += "\\hfill\n"
            
    tikz_code += "\\caption{Comparison of critical field ratio $R_{E_{\\rm crit}}(\\mu)$ across GUT groups.}\n"
    tikz_code += "\\end{figure}\n\n"
    
    # 3D parameter space
    tikz_code += "\\subsection*{3D Parameter Space Slices}\n\n"
    tikz_code += "\\begin{figure}[htb]\n"
    tikz_code += "\\centering\n"
    tikz_code += f"\\input{{SU5_3d_slice_phi_1.00.tex}}\n"
    tikz_code += "\\caption{Slice of 3D parameter space for SU(5) with fixed $\\Phi_{\\rm inst} = 1.0$.}\n"
    tikz_code += "\\end{figure}\n\n"
    
    # Close document
    tikz_code += "\\end{document}\n"
    
    # Save the combined file
    with open(f"{output_dir}/combined_parameter_scans.tex", "w") as f:
        f.write(tikz_code)
        
    print(f"Generated combined TikZ file: {output_dir}/combined_parameter_scans.tex")


if __name__ == "__main__":
    print("High-Resolution Parameter Scan Demonstration")
    print("===========================================")
    
    # Run 2D parameter scans
    for group in ['SU5', 'SO10', 'E6']:
        run_2d_parameter_scan(group)
    
    # Run 3D parameter scan (just for SU5 to save computation time)
    run_3d_parameter_scan('SU5')
    
    # Find optimal parameters for each group
    for group in ['SU5', 'SO10', 'E6']:
        find_optimal_parameters(group)
    
    # Generate combined TikZ file
    generate_combined_tikz_file()
    
    print("\nAll parameter scans completed successfully!")
