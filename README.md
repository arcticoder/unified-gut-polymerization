# Grand Unified Polymerization Framework

This repository implements a comprehensive theoretical and computational framework for applying polymer quantization to Grand Unified Theories (GUTs). By extending the closed-form SU(2) recoupling machinery to unified gauge groups (SU(5), SO(10), E6), this framework enables simultaneous enhancement of quantum inequality violations across all charge sectors.

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
  - `core.py`: Core numerical implementation
  - `recoupling.py`: Symbolic derivation of GUT recoupling coefficients
  - `taylor_extraction.py`: Taylor extraction to hypergeometric product mapping
  - `running_coupling.py`: Running coupling and instanton effects
- `docs/`: Mathematical derivations and documentation
  - `gut_polymer_core.tex`: Mathematical derivations for GUT polymerization
  - `taylor_extraction_su5.tex`: Self-contained SU(5) Taylor extraction
  - `unified_polymerized_feynman_rules.tex`: Side-by-side comparison of classical vs polymerized Feynman rules
  - `running_coupling_instantons.tex`: Derivation of running coupling and instanton formulas with GUT constants
- `examples/`: Usage examples and demonstration scripts
  - `demo_unified_gut_polymerization.py`: Parameter scans and plots
  - `advanced_lqg_integration.py`: Integration with LQG code
  - `symbolic_gut_recoupling.py`: Demonstrates symbolic derivation
  - `polymerized_feynman_rules_demo.py`: Numerical demonstration of polymerized propagator and vertices
  - `running_coupling_demo.py`: Visualization of running coupling and instanton rates
- `tests/`: Validation and testing suite

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

# Running coupling and instanton calculations
rc_calculator = RunningCouplingInstanton(group="SU5")
coupling_at_1TeV = rc_calculator.running_coupling(energy=1e3)
instanton_rate = rc_calculator.instanton_rate(coupling=coupling_at_1TeV, mu=0.1)
print(f"SU(5) coupling at 1 TeV: {coupling_at_1TeV:.5f}")
print(f"Polymerized instanton rate: {instanton_rate:.2e}")
```

## Dependencies

- NumPy, SciPy, SymPy
- Matplotlib
- Unified-LQG package (for LQG integration)

## References

- A. Arcticoder, "A Universal Generating Functional for SU(2) 3nj Symbols", May 24, 2025
- A. Arcticoder, "A Closed-Form Hypergeometric Product Formula for General SU(2) 3nj Recoupling Coefficients", May 25, 2025
- A. Arcticoder, "Closed-Form Matrix Elements for Arbitrary-Valence SU(2) Nodes via Generating Functionals", June 10, 2025
