"""
Unified GUT Polymerization Framework

This package provides tools for GUT-level polymer quantization, implementing a
unified approach to gauge theory polymerization across multiple symmetry groups
including SU(5), SO(10), and E6.
"""

from .core import GUTConfig, UnifiedGaugePolymerization
from .recoupling import GUTRecouplingCalculator, derive_recoupling_coeffs
from .taylor_extraction import TaylorHypergeometricExtractor

__all__ = [
    "GUTConfig",
    "UnifiedGaugePolymerization",
    "GUTRecouplingCalculator",
    "derive_recoupling_coeffs",
    "TaylorHypergeometricExtractor"
]
