"""
Parameter scanning module for polymerized GUT theories.

This module provides tools for scanning the parameter space of polymer-modified
GUT theories to identify regions with enhanced effects at reduced energy costs.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import scipy.optimize as optimize

from unified_gut_polymerization.core import GUTConfig, UnifiedGaugePolymerization
from unified_gut_polymerization.running_coupling import RunningCouplingInstanton


class ParameterScanner:
    """
    Class for performing parameter scans over polymer parameters and
    analyzing enhanced regions for polymerized GUT theories.
    """
    
    def __init__(self, group: str = 'SU5'):
        """
        Initialize the scanner with a specific gauge group.
        
        Args:
            group: The gauge group ('SU5', 'SO10', or 'E6')
        """
        self.group = group
        self.running_coupling = RunningCouplingInstanton(group=group)
        
        # Constants for Schwinger critical field
        self.E_crit = 1.3e18  # V/m, standard Schwinger critical field
        
        # Constants for particle production rates
        self.gamma_0 = 1.0  # Normalized standard rate
        
    def _compute_schwinger_rate(self, mu: float, b: float, E_field: float) -> float:
        """
        Compute the polymer-modified Schwinger rate.
        
        Args:
            mu: Polymer scale parameter
            b: Field strength parameter
            E_field: Electric field strength (V/m)
            
        Returns:
            Modified Schwinger rate
        """
        # Standard Schwinger rate (normalized)
        gamma_std = self.gamma_0 * (E_field/self.E_crit)**2 * np.exp(-np.pi * self.E_crit/E_field)
        
        # Polymer modification factor
        if mu > 0:
            # For mu > 0, the sinc function provides the modification
            s_factor = E_field / mu  # Proportional to sqrt(s) in the formula
            mod_factor = (np.sin(s_factor) / s_factor)**b
            
            # Avoid numerical issues when s_factor is close to zero
            if np.abs(s_factor) < 1e-10:
                mod_factor = 1.0
        else:
            mod_factor = 1.0  # No modification for mu=0 (classical case)
            
        return gamma_std * mod_factor
    
    def compute_instanton_rate(self, mu: float, phi_inst: float, energy: float = 1e12) -> float:
        """
        Compute the polymer-modified instanton rate.
        
        Args:
            mu: Polymer scale parameter
            phi_inst: Instanton topological charge
            energy: Energy scale in GeV
            
        Returns:
            Modified instanton rate
        """
        # Get coupling at the energy scale
        coupling = self.running_coupling.running_coupling(energy)
        
        # Compute the instanton rate
        gamma_inst = self.running_coupling.instanton_rate(coupling, mu, phi_inst)
        
        # Normalize by the classical rate
        gamma_inst_classical = self.running_coupling.instanton_rate(coupling, 1e-10, phi_inst)
        
        # Avoid division by zero
        if gamma_inst_classical < 1e-30:
            return 1.0
            
        return gamma_inst / gamma_inst_classical
    
    def compute_critical_field_ratio(self, mu: float) -> float:
        """
        Compute the ratio of polymer-modified critical field to standard critical field.
        
        Args:
            mu: Polymer scale parameter
            
        Returns:
            Ratio E_crit^poly / E_crit
        """
        if mu <= 0:
            return 1.0  # Classical case
            
        # Simple approximation based on polymer modification
        # In a full implementation, this would be calculated by solving for the field
        # where the polymer-modified rate matches a reference value
        modifier = np.sin(np.pi/mu) / (np.pi/mu) if mu > 0 else 1.0
        return modifier
    
    def compute_total_rate_ratio(self, mu: float, b: float, phi_inst: float, 
                                E_field: float = 1e16, energy: float = 1e12) -> float:
        """
        Compute the total rate ratio combining Schwinger and instanton contributions.
        
        Args:
            mu: Polymer scale parameter
            b: Field strength parameter
            phi_inst: Instanton topological charge
            E_field: Electric field strength (V/m)
            energy: Energy scale in GeV
            
        Returns:
            Total rate ratio R_rate
        """
        # Compute Schwinger rate ratio
        R_schwinger = self._compute_schwinger_rate(mu, b, E_field) / self._compute_schwinger_rate(0, b, E_field)
        
        # Compute instanton rate ratio
        R_instanton = self.compute_instanton_rate(mu, phi_inst, energy)
        
        # Combined rate ratio (simple sum as in the formula)
        R_total = R_schwinger + R_instanton - 1.0  # -1 to avoid double-counting the baseline
        
        return max(R_total, 1e-10)  # Ensure positive value for log plotting
    
    def scan_2d_parameter_space(self, 
                              mu_range: Tuple[float, float], 
                              b_range: Tuple[float, float],
                              resolution: int = 100,
                              phi_inst: float = 1.0,
                              E_field: float = 1e16,
                              energy: float = 1e12) -> Dict[str, np.ndarray]:
        """
        Perform a 2D parameter scan over mu and b.
        
        Args:
            mu_range: (min, max) for polymer scale parameter
            b_range: (min, max) for field strength parameter
            resolution: Number of points in each dimension
            phi_inst: Fixed instanton topological charge
            E_field: Electric field strength (V/m)
            energy: Energy scale in GeV
            
        Returns:
            Dictionary with mu_grid, b_grid, R_rate, R_E_crit
        """
        # Create parameter grids
        mu_values = np.linspace(mu_range[0], mu_range[1], resolution)
        b_values = np.linspace(b_range[0], b_range[1], resolution)
        mu_grid, b_grid = np.meshgrid(mu_values, b_values)
        
        # Initialize result arrays
        R_rate = np.zeros_like(mu_grid)
        R_E_crit = np.zeros_like(mu_grid)
        
        # Perform the scan
        for i in range(resolution):
            for j in range(resolution):
                mu = mu_grid[i, j]
                b = b_grid[i, j]
                
                # Compute rate ratio
                R_rate[i, j] = self.compute_total_rate_ratio(mu, b, phi_inst, E_field, energy)
                
                # Compute critical field ratio
                R_E_crit[i, j] = self.compute_critical_field_ratio(mu)
        
        return {
            'mu_grid': mu_grid,
            'b_grid': b_grid,
            'R_rate': R_rate,
            'R_E_crit': R_E_crit
        }
    
    def scan_3d_parameter_space(self,
                              mu_range: Tuple[float, float],
                              b_range: Tuple[float, float],
                              phi_range: Tuple[float, float],
                              resolution: int = 20,
                              E_field: float = 1e16,
                              energy: float = 1e12) -> Dict[str, np.ndarray]:
        """
        Perform a 3D parameter scan over mu, b, and phi_inst.
        
        Args:
            mu_range: (min, max) for polymer scale parameter
            b_range: (min, max) for field strength parameter
            phi_range: (min, max) for instanton topological charge
            resolution: Number of points in each dimension
            E_field: Electric field strength (V/m)
            energy: Energy scale in GeV
            
        Returns:
            Dictionary with parameter grids and computed values
        """
        # Create parameter grids
        mu_values = np.linspace(mu_range[0], mu_range[1], resolution)
        b_values = np.linspace(b_range[0], b_range[1], resolution)
        phi_values = np.linspace(phi_range[0], phi_range[1], resolution)
        
        # Initialize result arrays
        R_total = np.zeros((resolution, resolution, resolution))
        
        # Perform the scan
        for i, mu in enumerate(mu_values):
            for j, b in enumerate(b_values):
                for k, phi in enumerate(phi_values):
                    # Compute total rate ratio including instanton effects
                    R_total[i, j, k] = self.compute_total_rate_ratio(mu, b, phi, E_field, energy)
        
        return {
            'mu_values': mu_values,
            'b_values': b_values,
            'phi_values': phi_values,
            'R_total': R_total
        }
    
    def find_optimal_parameters(self, 
                              mu_range: Tuple[float, float],
                              b_range: Tuple[float, float],
                              phi_range: Tuple[float, float],
                              target_ratio: float = 10.0,
                              E_field_constraint: float = 1e17,
                              energy: float = 1e12) -> Dict[str, float]:
        """
        Find optimal parameters that maximize rate enhancement while keeping
        critical field below the constraint.
        
        Args:
            mu_range: (min, max) for polymer scale parameter
            b_range: (min, max) for field strength parameter
            phi_range: (min, max) for instanton topological charge
            target_ratio: Target enhancement ratio
            E_field_constraint: Maximum allowed critical field (V/m)
            energy: Energy scale in GeV
            
        Returns:
            Dictionary with optimal parameters and achieved values
        """
        # Define objective function to minimize (negative of rate ratio)
        def objective(params):
            mu, b, phi = params
            rate_ratio = self.compute_total_rate_ratio(mu, b, phi, E_field_constraint, energy)
            return -rate_ratio
        
        # Define constraint: critical field must be below constraint
        def constraint(params):
            mu, _, _ = params
            E_crit_poly = self.compute_critical_field_ratio(mu) * self.E_crit
            return E_field_constraint - E_crit_poly
        
        # Initial guess
        x0 = [
            (mu_range[0] + mu_range[1]) / 2,
            (b_range[0] + b_range[1]) / 2,
            (phi_range[0] + phi_range[1]) / 2
        ]
        
        # Define bounds
        bounds = [mu_range, b_range, phi_range]
        
        # Constraint dictionary
        cons = {'type': 'ineq', 'fun': constraint}
        
        # Run optimization
        result = optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        # Extract optimal parameters
        opt_mu, opt_b, opt_phi = result.x
        opt_rate_ratio = -result.fun  # Negative because we minimized negative of rate
        opt_E_crit_ratio = self.compute_critical_field_ratio(opt_mu)
        opt_E_crit = opt_E_crit_ratio * self.E_crit
        
        return {
            'mu': opt_mu,
            'b': opt_b,
            'phi_inst': opt_phi,
            'rate_ratio': opt_rate_ratio,
            'E_crit_ratio': opt_E_crit_ratio,
            'E_crit': opt_E_crit,
            'success': result.success,
            'message': result.message
        }


def export_data_for_pgfplots(data: Dict[str, np.ndarray], filename: str, value_key: str = 'R_rate'):
    """
    Export parameter scan data in a format suitable for PGFPlots.
    
    Args:
        data: Dictionary with mu_grid, b_grid, and values to plot
        filename: Output filename for the data file
        value_key: Key for the values to plot (default: 'R_rate')
    """
    X = data['mu_grid']
    Y = data['b_grid']
    Z = data[value_key]
    
    # Create a table of values suitable for PGFPlots
    with open(f"{filename}.dat", "w") as f:
        f.write("# mu b value\n")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write(f"{X[i, j]} {Y[i, j]} {Z[i, j]}\n")


def generate_tikz_contour(data: Dict[str, np.ndarray], 
                         filename: str, 
                         levels: List[float] = None,
                         title: str = None,
                         colormap: str = 'viridis'):
    """
    Generate a TikZ-compatible contour plot.
    
    Args:
        data: Dictionary with mu_grid, b_grid, and values to plot
        filename: Output filename (will add .tex and .png)
        levels: List of contour levels
        title: Plot title
        colormap: Matplotlib colormap name
    """
    # Extract data
    X = data['mu_grid']
    Y = data['b_grid']
    Z = data['R_rate']  # Default to rate ratio
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Default contour levels if not provided
    if levels is None:
        levels = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    
    # Create filled contour plot with log scale
    contour = plt.contourf(X, Y, Z, levels=levels, cmap=colormap, norm=LogNorm())
    
    # Add contour lines
    cont_lines = plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.7)
    
    # Add labels
    plt.clabel(cont_lines, inline=True, fontsize=8, fmt='%1.1f')
    
    # Add colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label('Rate Ratio $R_{\\rm rate}(\\mu, b)$')
    
    # Labels
    plt.xlabel('Polymer Scale $\\mu$')
    plt.ylabel('Field Strength Parameter $b$')
    
    # Title
    if title:
        plt.title(title)
    
    # Save as PNG
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    
    # Export data for PGFPlots
    export_data_for_pgfplots(data, f"{filename}_data", 'R_rate')
    
    # Generate TikZ code for a standalone contour plot
    tikz_code = "\\begin{tikzpicture}[scale=1.0]\n"
    tikz_code += "\\begin{axis}[\n"
    tikz_code += "    title={" + (title if title else "Parameter Scan") + "},\n"
    tikz_code += "    xlabel={Polymer Scale $\\mu$},\n"
    tikz_code += "    ylabel={Field Strength Parameter $b$},\n"
    tikz_code += "    colorbar,\n"
    tikz_code += "    colorbar style={ylabel=$R_{\\rm rate}(\\mu, b)$},\n"
    tikz_code += "    colormap name=" + colormap + ",\n"
    tikz_code += "    view={0}{90},\n"
    tikz_code += "    point meta min=" + str(min(levels)) + ",\n"
    tikz_code += "    point meta max=" + str(max(levels)) + ",\n"
    tikz_code += "]\n"
    
    # Add contour plot using actual data
    tikz_code += f"\\addplot3[surf,shader=interp,mesh/rows={X.shape[0]}] table {{./{filename}_data.dat}};\n"
    
    # Add contour lines for specific levels
    for level in [1.0, 10.0]:
        tikz_code += f"% Contour line for level {level}\n"
        tikz_code += f"\\addplot3[contour gnuplot={level},thick,red,samples=50] table {{./{filename}_data.dat}};\n"
    
    tikz_code += "\\end{axis}\n"
    tikz_code += "\\end{tikzpicture}\n"
    
    # Save TikZ code
    with open(f"{filename}.tex", "w") as f:
        f.write(tikz_code)
    
    # Close figure
    plt.close()


def generate_3d_slices(data: Dict[str, np.ndarray], filename_prefix: str):
    """
    Generate slices of 3D parameter space and save as PGFPlots/TikZ compatible files.
    
    Args:
        data: Dictionary with 3D scan results
        filename_prefix: Prefix for output filenames
    """
    mu_values = data['mu_values']
    b_values = data['b_values']
    phi_values = data['phi_values']
    R_total = data['R_total']
    
    # Create slices along each axis at the middle value
    mid_mu_idx = len(mu_values) // 2
    mid_b_idx = len(b_values) // 2
    mid_phi_idx = len(phi_values) // 2
    
    # Slice 1: fixed mu (b vs phi)
    fixed_mu = mu_values[mid_mu_idx]
    slice_1 = R_total[mid_mu_idx, :, :]
    
    plt.figure(figsize=(8, 6))
    plt.contourf(phi_values, b_values, slice_1, cmap='viridis', levels=10, norm=LogNorm())
    plt.colorbar(label='$R_{\\rm total}$')
    plt.xlabel('Instanton Charge $\\Phi_{\\rm inst}$')
    plt.ylabel('Field Parameter $b$')
    plt.title(f'Fixed $\\mu = {fixed_mu:.2f}$')
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_mu_{fixed_mu:.2f}.png", dpi=300)
    plt.close()
    
    # Slice 2: fixed b (mu vs phi)
    fixed_b = b_values[mid_b_idx]
    slice_2 = R_total[:, mid_b_idx, :]
    
    plt.figure(figsize=(8, 6))
    plt.contourf(phi_values, mu_values, slice_2, cmap='viridis', levels=10, norm=LogNorm())
    plt.colorbar(label='$R_{\\rm total}$')
    plt.xlabel('Instanton Charge $\\Phi_{\\rm inst}$')
    plt.ylabel('Polymer Scale $\\mu$')
    plt.title(f'Fixed $b = {fixed_b:.2f}$')
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_b_{fixed_b:.2f}.png", dpi=300)
    plt.close()
    
    # Slice 3: fixed phi (mu vs b)
    fixed_phi = phi_values[mid_phi_idx]
    slice_3 = R_total[:, :, mid_phi_idx]
    
    plt.figure(figsize=(8, 6))
    plt.contourf(b_values, mu_values, slice_3, cmap='viridis', levels=10, norm=LogNorm())
    plt.colorbar(label='$R_{\\rm total}$')
    plt.xlabel('Field Parameter $b$')
    plt.ylabel('Polymer Scale $\\mu$')
    plt.title(f'Fixed $\\Phi_{\\rm inst} = {fixed_phi:.2f}$')
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_phi_{fixed_phi:.2f}.png", dpi=300)
    plt.close()
    
    # Generate TikZ code example for one slice
    tikz_code = "\\begin{tikzpicture}[scale=1.0]\n"
    tikz_code += "\\begin{axis}[\n"
    tikz_code += f"    title={{Fixed $\\\\Phi_{{\\\\rm inst}} = {fixed_phi:.2f}$}},\n"
    tikz_code += "    xlabel={Field Parameter $b$},\n"
    tikz_code += "    ylabel={Polymer Scale $\\\\mu$},\n"
    tikz_code += "    colorbar,\n"
    tikz_code += "    colorbar style={ylabel=$R_{\\\\rm total}$},\n"
    tikz_code += "    colormap name=viridis,\n"
    tikz_code += "]\n"
    tikz_code += "\\addplot[contour filled={number=10}] table {contour_data_slice3.dat};\n"
    
    # Add region markers
    tikz_code += "% Highlight region where R > 1\n"
    tikz_code += "\\addplot[thick, red, dashed, domain=0:5] coordinates {\n"
    
    # Find the contour line approximately where R = 1
    for i in range(len(mu_values)):
        for j in range(len(b_values) - 1):
            if slice_3[i, j] < 1.0 and slice_3[i, j + 1] >= 1.0:
                interp = b_values[j] + (b_values[j + 1] - b_values[j]) * (1.0 - slice_3[i, j]) / (slice_3[i, j + 1] - slice_3[i, j])
                tikz_code += f"    ({interp}, {mu_values[i]})\n"
    
    tikz_code += "};\n"
    tikz_code += "\\end{axis}\n"
    tikz_code += "\\end{tikzpicture}\n"
    
    with open(f"{filename_prefix}_slice3.tex", "w") as f:
        f.write(tikz_code)
