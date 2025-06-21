# Grand Unified Polymerization Framework

This repository implements a comprehensive theoretical and computational framework for applying polymer quantization to Grand Unified Theories (GUTs). By extending the closed-form SU(2) recoupling machinery to unified gauge groups (SU(5), SO(10), E6), this framework enables simultaneous enhancement of quantum inequality violations across all charge sectors.

## Recent Updates (June 2025)

**🔄 Repository Reorganization Complete**
- **Fusion Code Migration**: All polymer-induced-fusion code has been successfully migrated to the dedicated [`polymer-fusion-framework`](https://github.com/arcticoder/polymer-fusion-framework) repository
- **Focused Scope**: This repository now exclusively focuses on fundamental GUT polymerization theory and mathematics
- **Enhanced Integration**: Maintains seamless integration with [`warp-bubble-optimizer`](https://github.com/arcticoder/warp-bubble-optimizer) for phenomenological applications

**🚀 Latest Developments**
- **Platinum Road Deliverables**: Complete implementation of platinum-road parameter validation and testing framework
- **3D Integration**: Advanced 3D parameter space exploration with comprehensive validation
- **QFT-ANEC Restoration**: Restored and enhanced QFT averaged null energy condition framework
- **Instanton Validation**: Complete validation of polymer-modified instanton calculations with uncertainty quantification
- **Enhanced Documentation**: Comprehensive documentation updates and integration summaries

## Key Features

- **Unified Gauge Polymerization**: Apply polymer quantization directly to GUT gauge fields rather than individual Standard Model sectors
- **Hypergeometric Product Formulas**: Closed-form expressions for SU(N) recoupling coefficients
- **Multiplicative Enhancement**: Coherently enhance electroweak, strong, and unified interactions with a single polymer parameter
- **Running Coupling Analysis**: Calculate unified coupling running with explicit β-function coefficients for SU(5), SO(10), and E6
- **Non-perturbative Instanton Effects**: Polymer-modified instanton rates with group-specific parameters
- **Experimental Predictions**: Quantitative predictions for proton decay, neutrino masses, and collider signatures

## Mathematical Framework

The core mathematics extends our recent SU(2) closed-form generating functional work to unified groups:

```
G_G({x_e}) = ∫∏ᵥ d²ʳwᵥ/π^r e^(-∑ᵥ||wᵥ||²) ∏ₑ e^(xₑϵ_G(wᵢ,wⱼ)) = 1/√det(I - K_G({x_e}))
```

Where r is the rank of group G (r=4 for SU(5), r=5 for SO(10), r=6 for E6).

The polymerized unified propagator takes the form:

```
D̃ᵃᵇₘᵤᵥ(k) = δᵃᵇ × [ηₘᵤᵥ - kₘkᵥ/k²]/μ² × sinc²(μ√(k²+m²))
```

Where indices a,b run over the entire adjoint representation of the unified group.

The running coupling with polymer effects incorporated:

```
α_eff(E) = α₀/(1 - b_G/(2π)·α₀·ln(E/E₀))
```

Where b_G is the one-loop β-function coefficient specific to each GUT group.

The instanton rate with polymer modification:

```
Γ_inst^poly = Λ_G⁴ exp[-8π²/α_s(μ)·sin(μ·Φ_inst)/μ]
```

Where Λ_G is the characteristic scale of the gauge group and Φ_inst is the instanton topological charge.

## Repository Structure

- `unified_gut_polymerization/`: Core implementation modules
  - `core.py`: Core numerical implementation with unified gauge polymerization
  - `recoupling.py`: Symbolic derivation of GUT recoupling coefficients
  - `taylor_extraction.py`: Taylor extraction to hypergeometric product mapping
  - `running_coupling.py`: Running coupling and instanton effects
  - `parameter_scan.py`: Advanced parameter space exploration
  - `platinum_road_*.py`: Platinum road deliverable implementations
  - `restore_qft_anec_framework.py`: QFT-ANEC framework restoration
- `gut_unified_polymerization/`: Alternative core implementation
  - `core.py`: Unified gauge polymerization with GUT group support
- `docs/`: Mathematical derivations and documentation
  - `gut_polymer_core.tex`: Mathematical derivations for GUT polymerization
  - `taylor_extraction_su5.tex`: Self-contained SU(5) Taylor extraction
  - `unified_polymerized_feynman_rules.tex`: Side-by-side comparison of classical vs polymerized Feynman rules
  - `running_coupling_instantons.tex`: Derivation of running coupling and instanton formulas with GUT constants
  - `advanced_parameter_scan_visualization.tex`: Advanced visualization techniques
  - `high_resolution_parameter_scans.tex`: High-resolution scanning methodologies
- `examples/`: Usage examples and demonstration scripts
  - `demo_unified_gut_polymerization.py`: Parameter scans and plots
  - `advanced_lqg_integration.py`: Integration with LQG code
  - `symbolic_gut_recoupling.py`: Demonstrates symbolic derivation
  - `polymerized_feynman_rules_demo.py`: Numerical demonstration of polymerized propagator and vertices
  - `running_coupling_demo.py`: Visualization of running coupling and instanton rates
  - `high_resolution_parameter_scan.py`: Advanced parameter scanning demonstration
- `tests/`: Validation and testing suite
- `plan_a_step5_reactor_design/`: Advanced reactor design analysis and validation

## Connected Repositories

This repository is part of a connected ecosystem of theoretical physics research:

- **[warp-bubble-optimizer](https://github.com/arcticoder/warp-bubble-optimizer)**: Phenomenological applications and warp drive physics
- **[polymer-fusion-framework](https://github.com/arcticoder/polymer-fusion-framework)**: Dedicated fusion research (migrated from this repo)

## Installation

```bash
# Clone the repository
git clone https://github.com/arcticoder/unified-gut-polymerization.git

# Install dependencies
cd unified-gut-polymerization
pip install -e .
```

## Usage Example

```python
from unified_gut_polymerization import UnifiedGaugePolymerization, GUTConfig, RunningCouplingInstanton

# Configure a SU(5) GUT polymerization
config = GUTConfig(gut_group="SU(5)", mu_polymer=0.1)
gut_poly = UnifiedGaugePolymerization(config)

# Calculate cross-section enhancements
enhancements = gut_poly.unified_cross_section_enhancement(center_of_mass_energy=1000.0)
print(f"Total multiplicative enhancement: {enhancements['total_multiplicative']:.2e}x")

# Generate comprehensive enhancement plots
fig = gut_poly.plot_enhancement_spectra(save_path="enhancement_analysis.png")

# Running coupling and instanton calculations
rc_calculator = RunningCouplingInstanton(group="SU5")
coupling_at_1TeV = rc_calculator.running_coupling(energy=1e3)
instanton_rate = rc_calculator.instanton_rate(coupling=coupling_at_1TeV, mu=0.1)
print(f"SU(5) coupling at 1 TeV: {coupling_at_1TeV:.5f}")
print(f"Polymerized instanton rate: {instanton_rate:.2e}")

# Advanced parameter space exploration
from unified_gut_polymerization.parameter_scan import ParameterSpaceScan
scanner = ParameterSpaceScan(config)
results = scanner.platinum_road_scan()
```

## Dependencies

- NumPy, SciPy, SymPy
- Matplotlib
- Unified-LQG package (for LQG integration)

## Migration Notice

**🔄 Repository Reorganization**: All fusion-related code has been migrated to the dedicated [`polymer-fusion-framework`](https://github.com/arcticoder/polymer-fusion-framework) repository as of June 2025. This repository now focuses exclusively on fundamental GUT polymerization theory.

## References

- A. Arcticoder, "A Universal Generating Functional for SU(2) 3nj Symbols", May 24, 2025
- A. Arcticoder, "A Closed-Form Hypergeometric Product Formula for General SU(2) 3nj Recoupling Coefficients", May 25, 2025
- A. Arcticoder, "Closed-Form Matrix Elements for Arbitrary-Valence SU(2) Nodes via Generating Functionals", June 10, 2025
