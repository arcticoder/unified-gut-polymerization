"""
Plan A: Complete Demonstration - Direct Mass-Energy Conversion with WEST Benchmarking
===================================================================================

Complete implementation of Plan A with all three steps:
1. Theoretical energy density (E=mc²) with polymer enhancement
2. Antimatter production cost assessment using NASA data
3. Energy conversion efficiency analysis (thermophotovoltaic/thermionic)

All analyses benchmarked against WEST tokamak world record (Feb 12, 2025):
- Confinement Time: 1,337 s
- Plasma Temperature: 50×10⁶ °C  
- Heating Power: 2 MW RF injection

Economic viability assessment: cost per kWh vs. polymer scale μ
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os

# Import the complete framework
try:
    from plan_a_direct_mass_energy import (
        WESTBaseline, DirectMassEnergyConverter, PolymerMassEnergyPipeline,
        AntimatterEnergyConverter, AntimatterProductionPipeline,
        RealisticAntimatterConverter, ConversionEfficiencyPipeline,
        WESTBenchmarkMetrics
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure plan_a_direct_mass_energy.py is available.")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompletePlanADemonstration:
    """Complete demonstration of Plan A with all three steps integrated"""
    
    def __init__(self, output_dir: str = "plan_a_complete_results"):
        """
        Initialize complete Plan A demonstration
        
        Args:
            output_dir: Directory for saving results and visualizations
        """
        self.output_dir = output_dir
        self.west_baseline = WESTBaseline()
        self.west_benchmark = WESTBenchmarkMetrics()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.results = {
            'step1_theoretical': {},
            'step2_antimatter': {},
            'step3_conversion': {},
            'integrated_analysis': {},
            'west_benchmarking': {},
            'economic_assessment': {}
        }
        
        logger.info(f"Plan A Complete Demonstration initialized")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"WEST baseline: {self.west_baseline.confinement_time:.0f}s, "
                   f"{self.west_baseline.plasma_temperature/1e6:.0f}×10⁶°C, "
                   f"{self.west_baseline.heating_power/1e6:.1f}MW")
    
    def run_step1_theoretical_analysis(self,
                                     mu_range: Tuple[float, float] = (0.1, 10.0),
                                     num_points: int = 50,
                                     test_masses: List[float] = None) -> Dict:
        """
        Step 1: Theoretical energy density analysis with polymer enhancement
        
        Args:
            mu_range: Range of polymer scale parameters
            num_points: Number of analysis points
            test_masses: List of masses to analyze (kg)
            
        Returns:
            Step 1 analysis results
        """
        logger.info("Running Step 1: Theoretical Energy Density Analysis")
        
        if test_masses is None:
            test_masses = [1e-6, 1e-3, 1e-1]  # μg, mg, 100mg
        
        # Initialize pipeline
        pipeline = PolymerMassEnergyPipeline(self.west_baseline)
        
        results = {
            'mu_range': mu_range,
            'test_masses_kg': test_masses,
            'mass_analyses': {},
            'polymer_sweep': {}
        }
        
        # Analyze each test mass
        for mass in test_masses:
            logger.info(f"  Analyzing mass: {mass*1e6:.1f} mg")
            
            # Run polymer scale sweep for this mass
            sweep_results = pipeline.run_polymer_scale_sweep(
                mu_range=mu_range,
                num_points=num_points,
                mass_kg=mass,
                production_cost_per_kg=1000.0  # Regular matter cost
            )
            
            # Single point analysis for detailed breakdown
            converter = DirectMassEnergyConverter(polymer_scale_mu=5.0)
            energy_data = converter.mass_energy_yield(mass)
            cost_data = converter.cost_analysis(mass, 1000.0)
            
            results['mass_analyses'][f'mass_{mass*1e6:.0f}mg'] = {
                'theoretical_energy_kwh': energy_data['energy_kwh'],
                'polymer_enhancement': energy_data['polymer_enhancement_factor'],
                'cost_per_kwh': cost_data['cost_per_kwh_usd'],
                'economically_viable': cost_data['is_economically_viable']
            }
            
            results['polymer_sweep'][f'mass_{mass*1e6:.0f}mg'] = sweep_results
        
        # WEST benchmark comparison
        west_comparison = pipeline.benchmark_against_west()
        results['west_benchmark'] = west_comparison
        
        self.results['step1_theoretical'] = results
        logger.info(f"Step 1 completed: {len(test_masses)} mass points analyzed")
        
        return results
    
    def run_step2_antimatter_analysis(self,
                                    mu_range: Tuple[float, float] = (0.1, 10.0),
                                    num_points: int = 50,
                                    antimatter_masses: List[float] = None) -> Dict:
        """
        Step 2: Antimatter production cost assessment
        
        Args:
            mu_range: Range of polymer scale parameters
            num_points: Number of analysis points
            antimatter_masses: List of antimatter masses to analyze (kg)
            
        Returns:
            Step 2 analysis results
        """
        logger.info("Running Step 2: Antimatter Production Cost Assessment")
        
        if antimatter_masses is None:
            antimatter_masses = [1e-15, 1e-12, 1e-9]  # fg, pg, ng
        
        # NASA antimatter production cost: $62.5 trillion per gram
        nasa_cost_per_gram = 62.5e12
        
        results = {
            'nasa_cost_per_gram': nasa_cost_per_gram,
            'mu_range': mu_range,
            'antimatter_masses_kg': antimatter_masses,
            'mass_analyses': {},
            'cost_scaling': {},
            'economic_thresholds': {
                'grid_competitive': 0.10,
                'premium_market': 1.00,
                'space_applications': 1000.00
            }
        }
        
        # Initialize antimatter pipeline
        pipeline = AntimatterProductionPipeline(self.west_baseline)
        
        for mass in antimatter_masses:
            mass_grams = mass * 1000
            logger.info(f"  Analyzing antimatter mass: {mass*1e15:.0f} fg")
            
            # Run cost analysis sweep
            cost_sweep = pipeline.run_cost_optimization_sweep(
                mu_range=mu_range,
                num_points=num_points,
                antimatter_mass_kg=mass
            )
            
            # Detailed analysis at optimal point
            converter = AntimatterEnergyConverter(polymer_scale_mu=5.0)
            energy_data = converter.antimatter_annihilation_energy(mass)
            cost_data = converter.comprehensive_cost_analysis(mass, nasa_cost_per_gram)
            
            results['mass_analyses'][f'mass_{mass*1e15:.0f}fg'] = {
                'theoretical_energy_kwh': energy_data['theoretical_energy_kwh'],
                'polymer_enhanced_energy_kwh': energy_data['polymer_enhanced_energy_kwh'],
                'production_cost_usd': cost_data['total_production_cost_usd'],
                'cost_per_kwh': cost_data['cost_per_kwh_usd'],
                'grid_competitive': cost_data['grid_competitive'],
                'space_viable': cost_data['space_viable'],
                'enhancement_factor': energy_data['polymer_enhancement_factor']
            }
            
            results['cost_scaling'][f'mass_{mass*1e15:.0f}fg'] = cost_sweep
        
        # Economic viability analysis
        viability_analysis = pipeline.economic_viability_analysis()
        results['viability_analysis'] = viability_analysis
        
        self.results['step2_antimatter'] = results
        logger.info(f"Step 2 completed: {len(antimatter_masses)} antimatter masses analyzed")
        
        return results
    
    def run_step3_conversion_efficiency(self,
                                      mu_range: Tuple[float, float] = (0.1, 10.0),
                                      num_points: int = 30,
                                      test_mass: float = 1e-12) -> Dict:
        """
        Step 3: Energy conversion efficiency analysis
        
        Args:
            mu_range: Range of polymer scale parameters
            num_points: Number of analysis points
            test_mass: Test mass for conversion analysis (kg)
            
        Returns:
            Step 3 analysis results
        """
        logger.info("Running Step 3: Energy Conversion Efficiency Analysis")
        
        # Initialize conversion efficiency pipeline
        pipeline = ConversionEfficiencyPipeline(self.west_baseline)
        
        results = {
            'test_mass_kg': test_mass,
            'mu_range': mu_range,
            'conversion_methods': ['tpv_lab', 'tpv_system', 'thermionic', 'direct'],
            'method_comparison': {},
            'efficiency_impact': {},
            'west_benchmark': {}
        }
        
        # Run conversion method comparison
        logger.info(f"  Comparing conversion methods for {test_mass*1e15:.0f} fg antimatter")
        method_comparison = pipeline.run_conversion_method_comparison(
            antimatter_mass_kg=test_mass,
            mu_range=mu_range,
            num_points=num_points
        )
        results['method_comparison'] = method_comparison
        
        # Efficiency impact analysis
        logger.info("  Analyzing efficiency impact on economics")
        efficiency_impact = pipeline.efficiency_impact_analysis(test_mass)
        results['efficiency_impact'] = efficiency_impact
        
        # WEST benchmark analysis
        logger.info("  Benchmarking against WEST tokamak")
        west_benchmark = pipeline.west_benchmark_analysis()
        results['west_benchmark'] = west_benchmark
        
        # Detailed analysis of best conversion method
        best_method = 'direct'  # Assume direct conversion is best
        converter = RealisticAntimatterConverter(
            polymer_scale_mu=5.0,
            conversion_method=best_method,
            polymer_enhanced=True
        )
        
        realistic_energy = converter.realistic_energy_conversion(test_mass)
        cost_analysis = converter.comprehensive_cost_analysis(test_mass)
        west_comparison = converter.west_benchmark_comparison(test_mass)
        
        results['optimal_analysis'] = {
            'best_method': best_method,
            'realistic_energy_kwh': realistic_energy['realistic_energy_kwh'],
            'conversion_efficiency': realistic_energy['conversion_efficiency'],
            'total_efficiency': realistic_energy['total_efficiency'],
            'cost_per_kwh': cost_analysis['cost_per_kwh_usd'],
            'west_energy_ratio': west_comparison['comparison_metrics']['energy_ratio'],
            'efficiency_loss_factor': realistic_energy['efficiency_loss_factor']
        }
        
        self.results['step3_conversion'] = results
        logger.info(f"Step 3 completed: {len(results['conversion_methods'])} methods analyzed")
        
        return results\n    \n    def run_integrated_analysis(self) -> Dict:\n        \"\"\"\n        Integrated analysis combining all three steps\n        \n        Returns:\n            Integrated analysis results\n        \"\"\"\n        logger.info(\"Running Integrated Analysis: Combining all three steps\")\n        \n        # Check that all steps have been run\n        required_steps = ['step1_theoretical', 'step2_antimatter', 'step3_conversion']\n        for step in required_steps:\n            if step not in self.results or not self.results[step]:\n                logger.error(f\"Step {step} must be run before integrated analysis\")\n                return {}\n        \n        results = {\n            'analysis_timestamp': datetime.now().isoformat(),\n            'west_baseline_reference': {\n                'confinement_time_s': self.west_baseline.confinement_time,\n                'plasma_temperature_c': self.west_baseline.plasma_temperature,\n                'heating_power_w': self.west_baseline.heating_power,\n                'date': self.west_baseline.date\n            },\n            'cross_step_comparison': {},\n            'economic_conclusions': {},\n            'technical_feasibility': {}\n        }\n        \n        # Compare theoretical vs realistic energy yields\n        step1_data = self.results['step1_theoretical']\n        step3_data = self.results['step3_conversion']\n        \n        # Example comparison for 1 mg mass (theoretical) vs 1 pg antimatter (realistic)\n        theoretical_1mg = step1_data['mass_analyses'].get('mass_1000mg', {})\n        antimatter_1pg = self.results['step2_antimatter']['mass_analyses'].get('mass_1fg', {})\n        realistic_1pg = step3_data['optimal_analysis']\n        \n        results['cross_step_comparison'] = {\n            'theoretical_regular_matter': {\n                'mass_kg': 1e-3,\n                'energy_kwh': theoretical_1mg.get('theoretical_energy_kwh', 0),\n                'cost_per_kwh': theoretical_1mg.get('cost_per_kwh', 0)\n            },\n            'antimatter_theoretical': {\n                'mass_kg': 1e-12,\n                'energy_kwh': antimatter_1pg.get('theoretical_energy_kwh', 0),\n                'cost_per_kwh': antimatter_1pg.get('cost_per_kwh', 0)\n            },\n            'antimatter_realistic': {\n                'mass_kg': 1e-12,\n                'energy_kwh': realistic_1pg.get('realistic_energy_kwh', 0),\n                'cost_per_kwh': realistic_1pg.get('cost_per_kwh', 0),\n                'efficiency_loss': realistic_1pg.get('efficiency_loss_factor', 0)\n            }\n        }\n        \n        # Economic conclusions\n        grid_threshold = 0.10\n        premium_threshold = 1.00\n        space_threshold = 1000.00\n        \n        results['economic_conclusions'] = {\n            'regular_matter_viable': theoretical_1mg.get('cost_per_kwh', float('inf')) < grid_threshold,\n            'antimatter_theoretical_viable': antimatter_1pg.get('cost_per_kwh', float('inf')) < space_threshold,\n            'antimatter_realistic_viable': realistic_1pg.get('cost_per_kwh', float('inf')) < space_threshold,\n            'grid_competitive_pathway': None,  # To be determined\n            'space_application_pathway': 'antimatter' if realistic_1pg.get('cost_per_kwh', float('inf')) < space_threshold else None\n        }\n        \n        # Technical feasibility assessment\n        west_energy_kwh = (self.west_baseline.heating_power * self.west_baseline.confinement_time) / 3.6e6\n        \n        results['technical_feasibility'] = {\n            'west_baseline_energy_kwh': west_energy_kwh,\n            'polymer_enhancement_required': True,\n            'conversion_efficiency_critical': True,\n            'antimatter_production_bottleneck': True,\n            'recommended_research_priorities': [\n                'Polymer enhancement factor validation',\n                'Energy conversion efficiency improvement',\n                'Antimatter production cost reduction',\n                'System integration challenges'\n            ]\n        }\n        \n        self.results['integrated_analysis'] = results\n        logger.info(\"Integrated analysis completed\")\n        \n        return results\n    \n    def generate_comprehensive_visualization(self, save_plots: bool = True) -> None:\n        \"\"\"\n        Generate comprehensive visualization of all analysis results\n        \n        Args:\n            save_plots: Whether to save plots to files\n        \"\"\"\n        logger.info(\"Generating comprehensive visualization\")\n        \n        # Create master figure with subplots for all three steps\n        fig = plt.figure(figsize=(20, 16))\n        \n        # Step 1: Theoretical energy yields\n        ax1 = plt.subplot(3, 3, 1)\n        if 'step1_theoretical' in self.results:\n            step1_data = self.results['step1_theoretical']\n            for mass_key, sweep_data in step1_data['polymer_sweep'].items():\n                mu_vals = sweep_data['mu_values']\n                energy_vals = sweep_data['energy_yields_kwh']\n                ax1.loglog(mu_vals, energy_vals, label=mass_key.replace('_', ' '))\n        \n        ax1.set_xlabel('Polymer Scale μ')\n        ax1.set_ylabel('Energy Yield (kWh)')\n        ax1.set_title('Step 1: Theoretical Energy Density')\n        ax1.legend()\n        ax1.grid(True, alpha=0.3)\n        \n        # Step 1: Cost analysis\n        ax2 = plt.subplot(3, 3, 2)\n        if 'step1_theoretical' in self.results:\n            step1_data = self.results['step1_theoretical']\n            for mass_key, sweep_data in step1_data['polymer_sweep'].items():\n                mu_vals = sweep_data['mu_values']\n                cost_vals = sweep_data['cost_per_kwh_values']\n                ax2.semilogy(mu_vals, cost_vals, label=mass_key.replace('_', ' '))\n        \n        ax2.axhline(y=0.10, color='g', linestyle='--', label='Grid Competitive')\n        ax2.set_xlabel('Polymer Scale μ')\n        ax2.set_ylabel('Cost per kWh ($)')\n        ax2.set_title('Step 1: Economic Analysis')\n        ax2.legend()\n        ax2.grid(True, alpha=0.3)\n        \n        # Step 2: Antimatter cost scaling\n        ax3 = plt.subplot(3, 3, 3)\n        if 'step2_antimatter' in self.results:\n            step2_data = self.results['step2_antimatter']\n            masses = []\n            costs = []\n            for mass_key, analysis in step2_data['mass_analyses'].items():\n                mass_fg = float(mass_key.split('_')[1].replace('fg', ''))\n                cost_per_kwh = analysis['cost_per_kwh']\n                if cost_per_kwh < 1e10:  # Filter extreme values\n                    masses.append(mass_fg)\n                    costs.append(cost_per_kwh)\n            \n            if masses and costs:\n                ax3.loglog(masses, costs, 'bo-', markersize=8)\n        \n        ax3.axhline(y=1000, color='r', linestyle='--', label='Space App Threshold')\n        ax3.set_xlabel('Antimatter Mass (fg)')\n        ax3.set_ylabel('Cost per kWh ($)')\n        ax3.set_title('Step 2: Antimatter Economics')\n        ax3.legend()\n        ax3.grid(True, alpha=0.3)\n        \n        # Step 3: Conversion efficiency comparison\n        ax4 = plt.subplot(3, 3, 4)\n        if 'step3_conversion' in self.results:\n            step3_data = self.results['step3_conversion']\n            if 'method_comparison' in step3_data:\n                comparison = step3_data['method_comparison']\n                mu_vals = comparison['mu_values']\n                for method, data in comparison['methods'].items():\n                    efficiencies = [e * 100 for e in data['conversion_efficiencies']]\n                    ax4.plot(mu_vals, efficiencies, linewidth=2, \n                           label=method.replace('_', ' ').title())\n        \n        ax4.set_xlabel('Polymer Scale μ')\n        ax4.set_ylabel('Conversion Efficiency (%)')\n        ax4.set_title('Step 3: Conversion Efficiency')\n        ax4.legend()\n        ax4.grid(True, alpha=0.3)\n        \n        # Step 3: Realistic energy output\n        ax5 = plt.subplot(3, 3, 5)\n        if 'step3_conversion' in self.results:\n            step3_data = self.results['step3_conversion']\n            if 'method_comparison' in step3_data:\n                comparison = step3_data['method_comparison']\n                mu_vals = comparison['mu_values']\n                for method, data in comparison['methods'].items():\n                    ax5.semilogy(mu_vals, data['realistic_energies_kwh'], linewidth=2,\n                               label=method.replace('_', ' ').title())\n        \n        ax5.set_xlabel('Polymer Scale μ')\n        ax5.set_ylabel('Realistic Energy (kWh)')\n        ax5.set_title('Step 3: Realistic Energy Output')\n        ax5.legend()\n        ax5.grid(True, alpha=0.3)\n        \n        # WEST benchmark comparison\n        ax6 = plt.subplot(3, 3, 6)\n        west_energy = (self.west_baseline.heating_power * self.west_baseline.confinement_time) / 3.6e6\n        \n        # Create bar chart comparing different approaches\n        approaches = ['WEST\\nBaseline', 'Theoretical\\n1mg', 'Antimatter\\n1pg Theoretical', 'Antimatter\\n1pg Realistic']\n        energies = [west_energy]\n        \n        if 'step1_theoretical' in self.results:\n            step1_data = self.results['step1_theoretical']\n            theoretical_1mg = step1_data['mass_analyses'].get('mass_1000mg', {})\n            energies.append(theoretical_1mg.get('theoretical_energy_kwh', 0))\n        else:\n            energies.append(0)\n        \n        if 'step2_antimatter' in self.results:\n            step2_data = self.results['step2_antimatter']\n            antimatter_1pg = step2_data['mass_analyses'].get('mass_1fg', {})\n            energies.append(antimatter_1pg.get('theoretical_energy_kwh', 0))\n        else:\n            energies.append(0)\n        \n        if 'step3_conversion' in self.results:\n            step3_data = self.results['step3_conversion']\n            realistic_data = step3_data.get('optimal_analysis', {})\n            energies.append(realistic_data.get('realistic_energy_kwh', 0))\n        else:\n            energies.append(0)\n        \n        colors = ['blue', 'green', 'orange', 'red']\n        bars = ax6.bar(approaches, energies, color=colors, alpha=0.7)\n        ax6.set_ylabel('Energy Output (kWh)')\n        ax6.set_title('Energy Output Comparison')\n        ax6.set_yscale('log')\n        \n        # Add value labels on bars\n        for bar, energy in zip(bars, energies):\n            if energy > 0:\n                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),\n                        f'{energy:.2e}', ha='center', va='bottom', rotation=90)\n        \n        # Economic viability summary\n        ax7 = plt.subplot(3, 3, 7)\n        if 'integrated_analysis' in self.results:\n            integrated = self.results['integrated_analysis']\n            conclusions = integrated.get('economic_conclusions', {})\n            \n            viability_data = {\n                'Regular Matter\\n(1mg)': conclusions.get('regular_matter_viable', False),\n                'Antimatter\\nTheoretical': conclusions.get('antimatter_theoretical_viable', False),\n                'Antimatter\\nRealistic': conclusions.get('antimatter_realistic_viable', False)\n            }\n            \n            approaches = list(viability_data.keys())\n            viable = [1 if v else 0 for v in viability_data.values()]\n            colors = ['green' if v else 'red' for v in viability_data.values()]\n            \n            bars = ax7.bar(approaches, viable, color=colors, alpha=0.7)\n            ax7.set_ylabel('Economic Viability')\n            ax7.set_title('Economic Viability Summary')\n            ax7.set_ylim(0, 1.2)\n            \n            # Add threshold labels\n            thresholds = ['Space Apps (<$1000/kWh)', 'Premium (<$1/kWh)', 'Grid (<$0.10/kWh)']\n            for i, threshold in enumerate(thresholds):\n                ax7.text(0.02, 0.98 - i*0.1, threshold, transform=ax7.transAxes, \n                        fontsize=8, verticalalignment='top')\n        \n        # Efficiency loss breakdown\n        ax8 = plt.subplot(3, 3, 8)\n        if 'step3_conversion' in self.results:\n            step3_data = self.results['step3_conversion']\n            if 'efficiency_impact' in step3_data:\n                efficiency_data = step3_data['efficiency_impact']\n                if 'efficiency_gaps' in efficiency_data:\n                    gaps = efficiency_data['efficiency_gaps']\n                    methods = list(gaps.keys())\n                    loss_percentages = [gaps[method]['energy_loss_percentage'] for method in methods]\n                    \n                    bars = ax8.bar([m.replace('_', '\\n') for m in methods], loss_percentages, \n                                  color='orange', alpha=0.7)\n                    ax8.set_ylabel('Energy Loss (%)')\n                    ax8.set_title('Step 3: Energy Loss by Method')\n        \n        # Summary text\n        ax9 = plt.subplot(3, 3, 9)\n        ax9.axis('off')\n        \n        summary_text = f\"\"\"PLAN A SUMMARY (vs WEST Baseline)\n        \nWEST Record (Feb 12, 2025):\n• {self.west_baseline.confinement_time:.0f}s confinement\n• {self.west_baseline.plasma_temperature/1e6:.0f}×10⁶°C temperature  \n• {self.west_baseline.heating_power/1e6:.1f}MW heating power\n\nKey Findings:\n• Regular matter: Low energy density\n• Antimatter: High energy, extreme cost\n• Conversion efficiency: Critical bottleneck\n• Polymer enhancement: Required for viability\n\nEconomic Conclusion:\n• Grid competition: Not viable\n• Space applications: Potentially viable\n• Research priorities: Efficiency & cost\"\"\"\n        \n        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, \n                fontsize=10, verticalalignment='top', fontfamily='monospace')\n        \n        plt.tight_layout()\n        plt.suptitle(f'Plan A: Complete Analysis - WEST-Calibrated Polymer Energy Framework\\n'\n                    f'Analysis Date: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}',\n                    fontsize=16, y=0.98)\n        \n        if save_plots:\n            plot_path = os.path.join(self.output_dir, 'plan_a_complete_analysis.png')\n            plt.savefig(plot_path, dpi=300, bbox_inches='tight')\n            logger.info(f\"Comprehensive visualization saved to {plot_path}\")\n        \n        plt.show()\n    \n    def save_complete_results(self, filename: str = None) -> str:\n        \"\"\"\n        Save all results to a comprehensive JSON file\n        \n        Args:\n            filename: Optional custom filename\n            \n        Returns:\n            Path to saved file\n        \"\"\"\n        if filename is None:\n            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n            filename = f'plan_a_complete_results_{timestamp}.json'\n        \n        filepath = os.path.join(self.output_dir, filename)\n        \n        # Convert any numpy types for JSON serialization\n        def json_serializable(obj):\n            if isinstance(obj, np.integer):\n                return int(obj)\n            elif isinstance(obj, np.floating):\n                return float(obj)\n            elif isinstance(obj, np.ndarray):\n                return obj.tolist()\n            elif isinstance(obj, dict):\n                return {key: json_serializable(value) for key, value in obj.items()}\n            elif isinstance(obj, list):\n                return [json_serializable(item) for item in obj]\n            else:\n                return obj\n        \n        # Prepare comprehensive results\n        comprehensive_results = {\n            'metadata': {\n                'analysis_type': 'Plan A: Direct Mass-Energy Conversion',\n                'framework_version': '1.0',\n                'timestamp': datetime.now().isoformat(),\n                'west_baseline_reference': {\n                    'date': self.west_baseline.date,\n                    'confinement_time_s': self.west_baseline.confinement_time,\n                    'plasma_temperature_c': self.west_baseline.plasma_temperature,\n                    'heating_power_w': self.west_baseline.heating_power\n                }\n            },\n            'results': json_serializable(self.results)\n        }\n        \n        try:\n            with open(filepath, 'w') as f:\n                json.dump(comprehensive_results, f, indent=2)\n            \n            logger.info(f\"Complete results saved to {filepath}\")\n            return filepath\n        \n        except Exception as e:\n            logger.error(f\"Error saving results: {e}\")\n            return None\n    \n    def generate_summary_report(self) -> str:\n        \"\"\"\n        Generate a human-readable summary report\n        \n        Returns:\n            Summary report as string\n        \"\"\"\n        report = []\n        report.append(\"=\" * 80)\n        report.append(\"PLAN A: COMPLETE ANALYSIS SUMMARY REPORT\")\n        report.append(\"Direct Mass-Energy Conversion with WEST Benchmarking\")\n        report.append(\"=\" * 80)\n        report.append(\"\")\n        \n        # WEST baseline information\n        report.append(f\"WEST Tokamak Baseline (February 12, 2025):\")\n        report.append(f\"  Confinement Time: {self.west_baseline.confinement_time:.0f} seconds\")\n        report.append(f\"  Plasma Temperature: {self.west_baseline.plasma_temperature/1e6:.0f}×10⁶ °C\")\n        report.append(f\"  Heating Power: {self.west_baseline.heating_power/1e6:.1f} MW\")\n        report.append(f\"  Total Energy: {(self.west_baseline.heating_power * self.west_baseline.confinement_time)/3.6e6:.2f} kWh\")\n        report.append(\"\")\n        \n        # Step 1 Summary\n        if 'step1_theoretical' in self.results:\n            report.append(\"STEP 1: THEORETICAL ENERGY DENSITY\")\n            report.append(\"-\" * 40)\n            step1 = self.results['step1_theoretical']\n            \n            for mass_key, analysis in step1['mass_analyses'].items():\n                mass_str = mass_key.replace('mass_', '').replace('mg', ' mg')\n                energy = analysis['theoretical_energy_kwh']\n                enhancement = analysis['polymer_enhancement']\n                cost = analysis['cost_per_kwh']\n                \n                report.append(f\"  {mass_str}:\")\n                report.append(f\"    Energy yield: {energy:.2e} kWh\")\n                report.append(f\"    Polymer enhancement: {enhancement:.2f}x\")\n                report.append(f\"    Cost per kWh: ${cost:.2e}\")\n                report.append(f\"    Grid competitive: {'Yes' if cost < 0.10 else 'No'}\")\n            \n            report.append(\"\")\n        \n        # Step 2 Summary\n        if 'step2_antimatter' in self.results:\n            report.append(\"STEP 2: ANTIMATTER PRODUCTION COST\")\n            report.append(\"-\" * 40)\n            step2 = self.results['step2_antimatter']\n            \n            report.append(f\"  NASA Production Cost: ${step2['nasa_cost_per_gram']:.1e} per gram\")\n            \n            for mass_key, analysis in step2['mass_analyses'].items():\n                mass_str = mass_key.replace('mass_', '').replace('fg', ' fg')\n                energy = analysis['theoretical_energy_kwh']\n                cost = analysis['cost_per_kwh']\n                \n                report.append(f\"  {mass_str} antimatter:\")\n                report.append(f\"    Energy yield: {energy:.2e} kWh\")\n                report.append(f\"    Cost per kWh: ${cost:.2e}\")\n                report.append(f\"    Space viable: {'Yes' if cost < 1000 else 'No'}\")\n            \n            report.append(\"\")\n        \n        # Step 3 Summary\n        if 'step3_conversion' in self.results:\n            report.append(\"STEP 3: ENERGY CONVERSION EFFICIENCY\")\n            report.append(\"-\" * 40)\n            step3 = self.results['step3_conversion']\n            \n            if 'optimal_analysis' in step3:\n                optimal = step3['optimal_analysis']\n                report.append(f\"  Best conversion method: {optimal['best_method']}\")\n                report.append(f\"  Conversion efficiency: {optimal['conversion_efficiency']*100:.1f}%\")\n                report.append(f\"  Total efficiency: {optimal['total_efficiency']*100:.1f}%\")\n                report.append(f\"  Realistic energy output: {optimal['realistic_energy_kwh']:.2e} kWh\")\n                report.append(f\"  Cost per kWh: ${optimal['cost_per_kwh']:.2e}\")\n                report.append(f\"  Energy loss factor: {(1-optimal['efficiency_loss_factor'])*100:.1f}%\")\n            \n            report.append(\"\")\n        \n        # Integrated conclusions\n        if 'integrated_analysis' in self.results:\n            report.append(\"INTEGRATED ANALYSIS CONCLUSIONS\")\n            report.append(\"-\" * 40)\n            integrated = self.results['integrated_analysis']\n            \n            if 'economic_conclusions' in integrated:\n                conclusions = integrated['economic_conclusions']\n                report.append(f\"  Regular matter grid competitive: {'Yes' if conclusions.get('regular_matter_viable', False) else 'No'}\")\n                report.append(f\"  Antimatter space viable: {'Yes' if conclusions.get('antimatter_realistic_viable', False) else 'No'}\")\n                \n                if conclusions.get('space_application_pathway'):\n                    report.append(f\"  Recommended pathway: {conclusions['space_application_pathway']}\")\n                else:\n                    report.append(\"  Recommended pathway: None economically viable\")\n            \n            if 'technical_feasibility' in integrated:\n                feasibility = integrated['technical_feasibility']\n                report.append(\"\")\n                report.append(\"  Research Priorities:\")\n                for priority in feasibility.get('recommended_research_priorities', []):\n                    report.append(f\"    • {priority}\")\n        \n        report.append(\"\")\n        report.append(\"=\" * 80)\n        report.append(f\"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\")\n        report.append(\"=\" * 80)\n        \n        return \"\\n\".join(report)\n\n\ndef run_complete_plan_a_demonstration():\n    \"\"\"\n    Run the complete Plan A demonstration with all three steps\n    \"\"\"\n    print(\"Starting Complete Plan A Demonstration...\")\n    print(\"This will run all three steps with WEST benchmarking\\n\")\n    \n    # Initialize demonstration\n    demo = CompletePlanADemonstration(output_dir=\"plan_a_complete_results\")\n    \n    try:\n        # Step 1: Theoretical analysis\n        print(\"Running Step 1: Theoretical Energy Density Analysis...\")\n        step1_results = demo.run_step1_theoretical_analysis(\n            mu_range=(0.1, 10.0),\n            num_points=40,\n            test_masses=[1e-6, 1e-3, 1e-1]  # μg, mg, 100mg\n        )\n        print(f\"✓ Step 1 completed: {len(step1_results['test_masses_kg'])} masses analyzed\\n\")\n        \n        # Step 2: Antimatter cost analysis\n        print(\"Running Step 2: Antimatter Production Cost Assessment...\")\n        step2_results = demo.run_step2_antimatter_analysis(\n            mu_range=(0.1, 10.0),\n            num_points=40,\n            antimatter_masses=[1e-15, 1e-12, 1e-9]  # fg, pg, ng\n        )\n        print(f\"✓ Step 2 completed: {len(step2_results['antimatter_masses_kg'])} antimatter masses analyzed\\n\")\n        \n        # Step 3: Conversion efficiency analysis\n        print(\"Running Step 3: Energy Conversion Efficiency Analysis...\")\n        step3_results = demo.run_step3_conversion_efficiency(\n            mu_range=(0.1, 10.0),\n            num_points=30,\n            test_mass=1e-12  # 1 pg\n        )\n        print(f\"✓ Step 3 completed: {len(step3_results['conversion_methods'])} conversion methods analyzed\\n\")\n        \n        # Integrated analysis\n        print(\"Running Integrated Analysis...\")\n        integrated_results = demo.run_integrated_analysis()\n        print(\"✓ Integrated analysis completed\\n\")\n        \n        # Generate visualization\n        print(\"Generating comprehensive visualization...\")\n        demo.generate_comprehensive_visualization(save_plots=True)\n        print(\"✓ Visualization generated\\n\")\n        \n        # Save results\n        print(\"Saving complete results...\")\n        results_file = demo.save_complete_results()\n        print(f\"✓ Results saved to: {results_file}\\n\")\n        \n        # Generate summary report\n        print(\"Generating summary report...\")\n        summary = demo.generate_summary_report()\n        \n        # Save summary report\n        summary_file = os.path.join(demo.output_dir, 'plan_a_summary_report.txt')\n        with open(summary_file, 'w') as f:\n            f.write(summary)\n        \n        print(f\"✓ Summary report saved to: {summary_file}\\n\")\n        \n        # Print summary to console\n        print(\"\\n\" + \"=\" * 60)\n        print(\"EXECUTIVE SUMMARY\")\n        print(\"=\" * 60)\n        \n        # Extract key findings\n        if integrated_results and 'economic_conclusions' in integrated_results:\n            conclusions = integrated_results['economic_conclusions']\n            print(f\"Regular matter grid competitive: {'Yes' if conclusions.get('regular_matter_viable', False) else 'No'}\")\n            print(f\"Antimatter space applications viable: {'Yes' if conclusions.get('antimatter_realistic_viable', False) else 'No'}\")\n            \n            if conclusions.get('space_application_pathway'):\n                print(f\"Recommended research pathway: {conclusions['space_application_pathway']} systems\")\n            else:\n                print(\"Recommended research pathway: Focus on efficiency improvements\")\n        \n        print(f\"\\nDetailed results available in: {demo.output_dir}/\")\n        print(\"Analysis complete!\")\n        \n        return demo\n        \n    except Exception as e:\n        print(f\"Error during demonstration: {e}\")\n        import traceback\n        traceback.print_exc()\n        return None\n\n\nif __name__ == \"__main__\":\n    # Run the complete demonstration\n    demo = run_complete_plan_a_demonstration()\n
