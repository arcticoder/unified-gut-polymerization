"""
Core implementation of the unified gauge polymerization framework
for Grand Unified Theories (GUTs).

This module provides the foundational classes and methods for working with
polymerized gauge theories at the GUT scale, supporting SU(5), SO(10), and E6
symmetry groups with both perturbative and non-perturbative techniques.
"""

import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional, Callable
import matplotlib.pyplot as plt
from scipy.special import hyp2f1
from scipy.integrate import quad


@dataclass
class GUTConfig:
    """Configuration for GUT polymerization settings.
    
    This class stores the parameters needed to define a specific
    polymerized GUT scenario, including symmetry group, energy scales,
    and polymerization parameters.
    """
    
    # Symmetry group: 'SU5', 'SO10', or 'E6'
    group: str = 'SU5'
    
    # Energy scales in GeV
    unification_scale: float = 2.0e16  # GUT scale
    polymer_scale: float = 1.0e19      # Typically Planck scale
    susy_breaking_scale: Optional[float] = 1.0e10
    
    # Polymerization parameters
    polymer_length: float = 1.0        # Dimensionless polymer scale parameter
    discreteness_parameter: float = 0.1  # Controls quantum geometry discreteness
    holonomy_cutoff: int = 10          # Truncation for holonomy expansions
    
    # Whether to include threshold corrections
    include_threshold_corrections: bool = True
    
    # Numerical computation settings
    integration_points: int = 100
    convergence_threshold: float = 1e-6
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_groups = ['SU5', 'SO10', 'E6']
        if self.group not in valid_groups:
            raise ValueError(f"Group must be one of {valid_groups}")
        
        if self.polymer_scale <= self.unification_scale:
            raise ValueError("Polymer scale must be higher than unification scale")
            
        if self.susy_breaking_scale is not None and self.susy_breaking_scale >= self.unification_scale:
            raise ValueError("SUSY breaking scale must be lower than unification scale")


class UnifiedGaugePolymerization:
    """Core implementation of unified gauge polymerization for GUT theories.
    
    This class implements the mathematical framework for polymer quantization
    of gauge theories at the GUT scale, providing methods for computing 
    modified propagators, vertex form factors, and physical observables.
    """
    
    def __init__(self, config: GUTConfig):
        """Initialize with the specified GUT configuration."""
        self.config = config
        self.group_data = self._initialize_group_data()
        self.propagator_cache = {}
        self.form_factor_cache = {}
        self._setup_symbolic_variables()
    
    def _setup_symbolic_variables(self):
        """Set up symbolic variables for analytical computations."""
        self.p = sp.Symbol('p', real=True)  # Momentum
        self.mu = sp.Symbol('mu', real=True, positive=True)  # Polymer scale
        self.lambda_sym = sp.Symbol('lambda', real=True)  # Polymerization parameter
        
    def _initialize_group_data(self) -> Dict:
        """Initialize data for the selected gauge group."""
        group_data = {
            'dimension': 0,
            'rank': 0,
            'casimir': 0.0,
            'generators': None,
            'structure_constants': None,
            'coupling_normalization': 1.0,
        }
        
        if self.config.group == 'SU5':
            group_data['dimension'] = 24
            group_data['rank'] = 4
            group_data['casimir'] = 5
            group_data['coupling_normalization'] = 1/30
        elif self.config.group == 'SO10':
            group_data['dimension'] = 45
            group_data['rank'] = 5
            group_data['casimir'] = 8
            group_data['coupling_normalization'] = 1/60
        elif self.config.group == 'E6':
            group_data['dimension'] = 78
            group_data['rank'] = 6
            group_data['casimir'] = 12
            group_data['coupling_normalization'] = 1/100
            
        return group_data
    
    def calculate_polymer_modified_propagator(self, momentum: float) -> complex:
        """Calculate the polymer-modified gauge boson propagator.
        
        Args:
            momentum: The momentum magnitude in GeV
            
        Returns:
            The modified propagator value
        """
        # Check cache first
        if momentum in self.propagator_cache:
            return self.propagator_cache[momentum]
        
        # Standard propagator (in Feynman gauge)
        std_propagator = 1.0 / (momentum**2 + 1e-10)
        
        # Apply polymer modification via hypergeometric function
        # This implements the determinant-to-hypergeometric mapping
        # for the gauge boson propagator
        mu = self.config.polymer_scale
        lambda_p = self.config.polymer_length
        
        if momentum/mu > 10.0:
            # Far above polymer scale - use asymptotic form
            polymer_factor = 1.0 + (self.group_data['casimir'] * lambda_p**2 * mu**2) / (momentum**2)
        else:
            # Near or below polymer scale - use full hypergeometric form
            polymer_factor = hyp2f1(
                1, 
                self.group_data['dimension'] / 4,
                1 + self.group_data['rank'] / 2,
                -(lambda_p**2 * mu**2) / (momentum**2)
            )
            
        result = std_propagator / polymer_factor
        
        # Cache the result
        self.propagator_cache[momentum] = result
        return result
    
    def vertex_form_factor(self, momenta: List[float]) -> complex:
        """Calculate the polymer-modified vertex form factor.
        
        Args:
            momenta: List of the three momenta entering the vertex
            
        Returns:
            The form factor modifying the standard vertex
        """
        # Create a hashable key for caching
        key = tuple(sorted(momenta))
        if key in self.form_factor_cache:
            return self.form_factor_cache[key]
            
        # Compute the energy scale (highest momentum)
        energy_scale = max(momenta)
        
        # Compute the form factor based on the holonomy expansion
        mu = self.config.polymer_scale
        lambda_p = self.config.polymer_length
        cutoff = self.config.holonomy_cutoff
        
        # Base form factor
        form_factor = 1.0
        
        # Add holonomy corrections
        for n in range(1, cutoff + 1):
            # This implements the vertex correction from the polymerized
            # path integral measure
            coef = self._holonomy_coefficient(n)
            form_factor += coef * (lambda_p * energy_scale / mu)**(2*n)
        
        # Apply asymptotic suppression at high energies
        if energy_scale > 0.1 * mu:
            suppression = np.exp(-(energy_scale / mu)**2 * lambda_p**2 / 4)
            form_factor *= suppression
            
        self.form_factor_cache[key] = form_factor
        return form_factor
    
    def _holonomy_coefficient(self, order: int) -> float:
        """Compute the coefficient for the holonomy expansion at given order.
        
        Args:
            order: The order in the holonomy expansion
            
        Returns:
            The coefficient value
        """
        # These coefficients depend on the gauge group and implement
        # the polymer corrections to the vertex structure
        if self.config.group == 'SU5':
            return (-1)**order / (np.math.factorial(order) * (2*order + 1))
        elif self.config.group == 'SO10':
            return (-1)**order / (np.math.factorial(order) * (2*order + 3))
        else:  # E6
            return (-1)**order / (np.math.factorial(order) * (2*order + 5))
    
    def running_coupling(self, energy: float) -> float:
        """Calculate the polymer-modified running coupling at specified energy.
        
        Args:
            energy: The energy scale in GeV
            
        Returns:
            The effective coupling constant value
        """
        # Standard running (one-loop approximation)
        # α⁻¹(E) = α⁻¹(E₀) - (b/2π) * log(E/E₀)
        alpha_gut = 1/24.0  # Typical GUT-scale coupling
        
        # Beta function coefficient depends on the group
        if self.config.group == 'SU5':
            beta_coeff = -3
        elif self.config.group == 'SO10':
            beta_coeff = -1
        else:  # E6
            beta_coeff = 3
            
        # Standard running
        inv_alpha = 1/alpha_gut - (beta_coeff / (2*np.pi)) * np.log(energy / self.config.unification_scale)
        
        # Polymer modifications only significant near polymer scale
        if energy > 0.01 * self.config.polymer_scale:
            # Implement polymer corrections to the beta function
            polymer_ratio = energy / self.config.polymer_scale
            correction = self._polymer_beta_correction(polymer_ratio)
            inv_alpha += correction
            
        return 1/inv_alpha
    
    def _polymer_beta_correction(self, energy_ratio: float) -> float:
        """Compute polymer correction to the beta function.
        
        Args:
            energy_ratio: Ratio of energy to polymer scale
            
        Returns:
            The correction term to the inverse coupling
        """
        # Polynomial approximation of the correction, based on the 
        # hypergeometric asymptotics of the polymer deformation
        lambda_p = self.config.polymer_length
        x = energy_ratio * lambda_p
        
        if x < 0.01:
            return 0.0  # Negligible correction at low energies
        
        # Group-dependent correction terms
        if self.config.group == 'SU5':
            return (self.group_data['casimir'] / np.pi) * np.arctan(x**2) * x
        elif self.config.group == 'SO10':
            return (self.group_data['casimir'] / np.pi) * np.arctan(x**2) * x**1.5
        else:  # E6
            return (self.group_data['casimir'] / np.pi) * np.arctan(x**2) * x**2
    
    def compute_cross_section_enhancement(self, process_energy: float) -> float:
        """Compute the enhancement/suppression factor for cross-sections.
        
        Args:
            process_energy: The characteristic energy of the process in GeV
            
        Returns:
            The enhancement factor relative to standard GUT predictions
        """
        # This computes how polymer effects modify cross sections
        # Factor of 1.0 means no change from standard predictions
        
        if process_energy < 0.001 * self.config.polymer_scale:
            return 1.0  # No effect at low energies
            
        # Compute combined effect from propagator and vertex modifications
        propagator_factor = abs(self.calculate_polymer_modified_propagator(process_energy)) / (1.0 / process_energy**2)
        vertex_factor = abs(self.vertex_form_factor([process_energy] * 3))
        
        # Additional phase space and measure factors
        measure_factor = 1.0
        if process_energy > 0.1 * self.config.polymer_scale:
            # Phase space gets modified at high energies due to 
            # minimal length effects in the polymerized theory
            lambda_p = self.config.polymer_length
            mu = self.config.polymer_scale
            measure_factor = 1.0 / (1.0 + (lambda_p * process_energy / mu)**2)
        
        return propagator_factor * vertex_factor**2 * measure_factor
    
    def threshold_analysis(self, energies: np.ndarray) -> np.ndarray:
        """Analyze unification threshold corrections with polymer effects.
        
        Args:
            energies: Array of energy values to analyze (in GeV)
            
        Returns:
            Array of threshold correction values
        """
        if not self.config.include_threshold_corrections:
            return np.zeros_like(energies)
            
        # Compute threshold corrections for each energy
        corrections = np.zeros_like(energies, dtype=float)
        
        for i, energy in enumerate(energies):
            # Standard logarithmic threshold correction
            if self.config.susy_breaking_scale is not None:
                standard_term = np.log(energy / self.config.susy_breaking_scale) / (2*np.pi)
            else:
                standard_term = 0
            
            # Polymer modification to threshold effects
            if energy > 0.01 * self.config.polymer_scale:
                ratio = energy / self.config.polymer_scale
                lambda_p = self.config.polymer_length
                
                # Implement the hypergeometric threshold correction
                polymer_term = (lambda_p**2 * ratio**2) / (1 + lambda_p**2 * ratio**2)
                polymer_term *= self.group_data['casimir'] / (4*np.pi)
                
                corrections[i] = standard_term + polymer_term
            else:
                corrections[i] = standard_term
                
        return corrections
    
    def compute_phenomenology(self, energies: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute phenomenological predictions across energy ranges.
        
        Args:
            energies: Array of energy values (in GeV)
            
        Returns:
            Dictionary of arrays with predictions for various observables
        """
        results = {
            'coupling': np.zeros_like(energies),
            'cross_section_factor': np.zeros_like(energies),
            'threshold_corrections': np.zeros_like(energies),
            'propagator_modifications': np.zeros_like(energies),
        }
        
        # Compute each quantity across the energy range
        for i, energy in enumerate(energies):
            results['coupling'][i] = self.running_coupling(energy)
            results['cross_section_factor'][i] = self.compute_cross_section_enhancement(energy)
            results['propagator_modifications'][i] = abs(self.calculate_polymer_modified_propagator(energy)) * energy**2
            
        # Calculate threshold corrections
        results['threshold_corrections'] = self.threshold_analysis(energies)
        
        return results
    
    def plot_polymer_effects(self, e_min: float = 1e12, e_max: float = 1e20, points: int = 100):
        """Plot the various polymer effects across energy scales.
        
        Args:
            e_min: Minimum energy in GeV
            e_max: Maximum energy in GeV
            points: Number of points to calculate
        """
        energies = np.logspace(np.log10(e_min), np.log10(e_max), points)
        results = self.compute_phenomenology(energies)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Polymer Effects in {self.config.group} Unified Theory', fontsize=16)
        
        # Plot running coupling
        axes[0, 0].loglog(energies, results['coupling'], 'b-')
        axes[0, 0].set_xlabel('Energy (GeV)')
        axes[0, 0].set_ylabel('Coupling α')
        axes[0, 0].set_title('Running Coupling')
        axes[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Plot cross section enhancement
        axes[0, 1].loglog(energies, results['cross_section_factor'], 'r-')
        axes[0, 1].set_xlabel('Energy (GeV)')
        axes[0, 1].set_ylabel('Enhancement Factor')
        axes[0, 1].set_title('Cross Section Enhancement')
        axes[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Plot propagator modification
        axes[1, 0].loglog(energies, results['propagator_modifications'], 'g-')
        axes[1, 0].set_xlabel('Energy (GeV)')
        axes[1, 0].set_ylabel('Modification Factor')
        axes[1, 0].set_title('Propagator Modification')
        axes[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Plot threshold corrections
        axes[1, 1].semilogx(energies, results['threshold_corrections'], 'm-')
        axes[1, 1].set_xlabel('Energy (GeV)')
        axes[1, 1].set_ylabel('Correction')
        axes[1, 1].set_title('Threshold Corrections')
        axes[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        return fig
