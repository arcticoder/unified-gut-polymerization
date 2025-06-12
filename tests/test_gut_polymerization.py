"""
Basic tests for the GUT unified polymerization package.
"""

import unittest
import numpy as np
from unified_gut_polymerization import GUTConfig, UnifiedGaugePolymerization


class TestGUTConfig(unittest.TestCase):
    """Test the GUTConfig class."""

    def test_valid_config(self):
        """Test that valid configuration parameters are accepted."""
        config = GUTConfig(
            group='SU5',
            polymer_scale=1.0e19,
            unification_scale=2.0e16,
            polymer_length=1.0,
        )
        self.assertEqual(config.group, 'SU5')
        self.assertEqual(config.polymer_scale, 1.0e19)
        self.assertEqual(config.polymer_length, 1.0)

    def test_invalid_group(self):
        """Test that invalid groups are rejected."""
        with self.assertRaises(ValueError):
            GUTConfig(group='SU3')

    def test_invalid_scales(self):
        """Test that invalid scale relationships are rejected."""
        with self.assertRaises(ValueError):
            # Polymer scale must be higher than unification scale
            GUTConfig(
                group='SU5',
                polymer_scale=1.0e15,
                unification_scale=2.0e16,
            )
            
        with self.assertRaises(ValueError):
            # SUSY breaking must be lower than unification
            GUTConfig(
                group='SU5',
                susy_breaking_scale=3.0e16,
                unification_scale=2.0e16,
            )


class TestUnifiedGaugePolymerization(unittest.TestCase):
    """Test the UnifiedGaugePolymerization class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_su5 = GUTConfig(group='SU5')
        self.config_so10 = GUTConfig(group='SO10')
        self.config_e6 = GUTConfig(group='E6')
        
        self.model_su5 = UnifiedGaugePolymerization(self.config_su5)
        self.model_so10 = UnifiedGaugePolymerization(self.config_so10)
        self.model_e6 = UnifiedGaugePolymerization(self.config_e6)

    def test_group_data_initialization(self):
        """Test that group data is correctly initialized."""
        self.assertEqual(self.model_su5.group_data['dimension'], 24)
        self.assertEqual(self.model_su5.group_data['rank'], 4)
        
        self.assertEqual(self.model_so10.group_data['dimension'], 45)
        self.assertEqual(self.model_so10.group_data['rank'], 5)
        
        self.assertEqual(self.model_e6.group_data['dimension'], 78)
        self.assertEqual(self.model_e6.group_data['rank'], 6)

    def test_propagator_calculation(self):
        """Test propagator calculation."""
        # Low energy should be close to standard propagator
        low_energy = 1.0e12
        high_energy = 1.0e19
        
        # At low energy, should be close to standard propagator
        propagator_low = self.model_su5.calculate_polymer_modified_propagator(low_energy)
        expected_low = 1.0 / (low_energy**2)
        self.assertAlmostEqual(
            abs(propagator_low) / abs(expected_low),
            1.0,
            delta=0.01  # Close to standard value
        )
        
        # At high energy, should show modifications
        propagator_high = self.model_su5.calculate_polymer_modified_propagator(high_energy)
        expected_high = 1.0 / (high_energy**2)
        # Should be different from standard value
        self.assertNotAlmostEqual(
            abs(propagator_high) / abs(expected_high),
            1.0,
            delta=0.1
        )

    def test_running_coupling(self):
        """Test running coupling calculation."""
        # Running coupling should increase with energy
        e1 = 1.0e14
        e2 = 1.0e16
        
        alpha1 = self.model_su5.running_coupling(e1)
        alpha2 = self.model_su5.running_coupling(e2)
        
        # Coupling should get stronger at higher energy
        self.assertGreater(alpha2, alpha1)
        
    def test_cross_section_enhancement(self):
        """Test cross-section enhancement calculation."""
        low_energy = 1.0e12
        high_energy = 1.0e19
        
        # Low energy should have minimal enhancement
        enhancement_low = self.model_su5.compute_cross_section_enhancement(low_energy)
        self.assertAlmostEqual(enhancement_low, 1.0, delta=0.01)
        
        # High energy should show significant enhancement or suppression
        enhancement_high = self.model_su5.compute_cross_section_enhancement(high_energy)
        self.assertNotEqual(round(enhancement_high, 2), 1.0)

    def test_threshold_analysis(self):
        """Test threshold correction analysis."""
        energies = np.array([1e12, 1e14, 1e16, 1e18, 1e20])
        
        corrections = self.model_su5.threshold_analysis(energies)
        
        # Should return an array of the same size
        self.assertEqual(len(corrections), len(energies))
        
        # Corrections should increase with energy
        self.assertLess(corrections[0], corrections[-1])
        
    def test_compute_phenomenology(self):
        """Test the computation of phenomenological predictions."""
        energies = np.array([1e12, 1e14, 1e16, 1e18, 1e20])
        
        results = self.model_su5.compute_phenomenology(energies)
        
        # Should return a dictionary with the expected keys
        expected_keys = ['coupling', 'cross_section_factor', 
                         'threshold_corrections', 'propagator_modifications']
        
        for key in expected_keys:
            self.assertIn(key, results)
            self.assertEqual(len(results[key]), len(energies))


if __name__ == '__main__':
    unittest.main()
