"""
Unified Polymer-Enhanced Energy Pipeline Integration
==================================================

This module integrates both Plan A (Direct Mass-Energy Conversion) and 
Plan B (Polymer-Enhanced Fusion) to provide comprehensive comparative analysis
and identify optimal pathways for experimental focus.

Calibrated against WEST tokamak world record (February 12, 2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime

from plan_a_direct_mass_energy import (
    DirectMassEnergyConverter, PolymerMassEnergyPipeline
)
from plan_b_polymer_fusion import (
    WESTBaseline, PolymerEnhancedFusion, PolymerFusionPipeline, FusionReaction
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedPolymerEnergyAnalysis:
    """Unified analysis framework for both polymer energy pathways"""
    
    def __init__(self, west_baseline: WESTBaseline = None):
        """
        Initialize unified analysis
        
        Args:
            west_baseline: WEST tokamak baseline (default creates new instance)
        """
        self.west_baseline = west_baseline or WESTBaseline()
        
        # Initialize individual pipelines
        self.plan_a_pipeline = PolymerMassEnergyPipeline(self.west_baseline)
        self.plan_b_pipeline = PolymerFusionPipeline(self.west_baseline)
        
        # Results storage
        self.unified_results = {}
        
    def run_comprehensive_analysis(self,
                                 mu_range: Tuple[float, float] = (0.1, 10.0),
                                 num_points: int = 50,
                                 mass_kg: float = 0.001,
                                 production_cost_per_kg: float = 1000.0,
                                 reactor_cost_usd: float = 20e9) -> Dict:
        """
        Run comprehensive analysis for both pathways
        
        Args:
            mu_range: Polymer scale parameter range
            num_points: Number of analysis points
            mass_kg: Mass for Plan A analysis (kg)
            production_cost_per_kg: Production cost for Plan A ($/kg)
            reactor_cost_usd: Reactor cost for Plan B ($)
            
        Returns:
            Unified results dictionary
        """
        logger.info("Running comprehensive polymer energy analysis...")
        
        # Run Plan A analysis
        logger.info("Analyzing Plan A: Direct Mass-Energy Conversion")
        plan_a_results = self.plan_a_pipeline.run_polymer_scale_sweep(
            mu_range=mu_range,
            num_points=num_points,
            mass_kg=mass_kg,
            production_cost_per_kg=production_cost_per_kg
        )
        
        # Run Plan B analysis
        logger.info("Analyzing Plan B: Polymer-Enhanced Fusion")
        plan_b_results = self.plan_b_pipeline.run_polymer_scale_sweep(
            mu_range=mu_range,
            num_points=num_points,
            reactor_cost_usd=reactor_cost_usd
        )
        
        # Store results
        self.unified_results = {
            'analysis_parameters': {
                'mu_range': mu_range,
                'num_points': num_points,
                'mass_kg': mass_kg,
                'production_cost_per_kg': production_cost_per_kg,
                'reactor_cost_usd': reactor_cost_usd,
                'analysis_date': datetime.now().isoformat()
            },
            'west_baseline': {
                'confinement_time_s': self.west_baseline.confinement_time,
                'plasma_temperature_c': self.west_baseline.plasma_temperature,
                'heating_power_w': self.west_baseline.heating_power,
                'date': self.west_baseline.date,
                'location': self.west_baseline.location
            },
            'plan_a_results': plan_a_results,
            'plan_b_results': plan_b_results
        }
        
        # Perform comparative analysis
        self._perform_comparative_analysis()
        
        return self.unified_results
    
    def _perform_comparative_analysis(self):
        """Perform detailed comparative analysis between both plans"""
        plan_a = self.unified_results['plan_a_results']
        plan_b = self.unified_results['plan_b_results']
        
        mu_values = plan_a['mu_values']
        
        # Economic comparison
        plan_a_costs = plan_a['cost_per_kwh_values']
        plan_b_costs = plan_b['cost_per_kwh_values']
        
        # Find crossover points
        economic_crossovers = []
        for i, mu in enumerate(mu_values):
            if (plan_a_costs[i] < 0.10 or plan_b_costs[i] < 0.10):
                economic_crossovers.append({
                    'mu': mu,
                    'plan_a_cost': plan_a_costs[i],
                    'plan_b_cost': plan_b_costs[i],
                    'plan_a_viable': plan_a_costs[i] < 0.10,
                    'plan_b_viable': plan_b_costs[i] < 0.10
                })
        
        # Energy density comparison
        plan_a_energies = plan_a['energy_yields_kwh']
        plan_b_net_powers = plan_b['net_powers_mw']
        
        # Convert Plan B to equivalent energy (assuming 1 hour operation)
        plan_b_energies = [p * 1000 for p in plan_b_net_powers]  # MW to kWh for 1 hour
        
        # Technology readiness assessment
        tech_readiness = self._assess_technology_readiness()
        
        # Compile comparative analysis
        comparative = {
            'economic_crossovers': economic_crossovers,
            'optimal_pathways': self._identify_optimal_pathways(),
            'energy_density_comparison': {
                'mu_values': mu_values,
                'plan_a_energy_yields_kwh': plan_a_energies,
                'plan_b_energy_yields_kwh': plan_b_energies
            },
            'technology_readiness': tech_readiness,
            'experimental_focus_recommendations': self._generate_experimental_recommendations()
        }
        
        self.unified_results['comparative_analysis'] = comparative
    
    def _assess_technology_readiness(self) -> Dict:
        """Assess technology readiness levels (TRL) for both pathways"""
        return {
            'plan_a_direct_conversion': {
                'current_trl': 2,  # Technology concept formulated
                'description': 'Theoretical framework established, requires proof-of-concept',
                'key_challenges': [
                    'Polymer-enhanced mass-energy conversion mechanism',
                    'Containment and control systems',
                    'Radiation safety and waste management'
                ],
                'required_experiments': [
                    'Small-scale polymer cross-section validation',
                    'Energy extraction efficiency measurement',
                    'Safety protocol development'
                ]
            },
            'plan_b_polymer_fusion': {
                'current_trl': 4,  # Technology validated in lab
                'description': 'Building on established fusion technology with polymer enhancements',
                'key_challenges': [
                    'Polymer integration with plasma confinement',
                    'Material compatibility at high temperatures',
                    'Enhanced confinement stability'
                ],
                'required_experiments': [
                    'Polymer-plasma interaction studies',
                    'Enhanced confinement demonstration',
                    'Material testing in fusion environment'
                ]
            }
        }
    
    def _identify_optimal_pathways(self) -> Dict:
        """Identify optimal pathways based on economic and technical analysis"""
        plan_a = self.unified_results['plan_a_results']
        plan_b = self.unified_results['plan_b_results']
        
        # Find minimum cost points
        plan_a_min_cost_idx = np.argmin(plan_a['cost_per_kwh_values'])
        plan_b_valid_costs = [c for c in plan_b['cost_per_kwh_values'] if c < float('inf')]
        
        optimal_pathways = {
            'plan_a_optimal': {
                'mu': plan_a['mu_values'][plan_a_min_cost_idx],
                'cost_per_kwh': plan_a['cost_per_kwh_values'][plan_a_min_cost_idx],
                'economically_viable': plan_a['cost_per_kwh_values'][plan_a_min_cost_idx] < 0.10
            }
        }
        
        if plan_b_valid_costs:
            plan_b_min_cost_idx = np.argmin(plan_b['cost_per_kwh_values'])
            optimal_pathways['plan_b_optimal'] = {
                'mu': plan_b['mu_values'][plan_b_min_cost_idx],
                'cost_per_kwh': plan_b['cost_per_kwh_values'][plan_b_min_cost_idx],
                'q_factor': plan_b['q_factors'][plan_b_min_cost_idx],
                'economically_viable': plan_b['cost_per_kwh_values'][plan_b_min_cost_idx] < 0.10
            }
        
        return optimal_pathways
    
    def _generate_experimental_recommendations(self) -> Dict:
        """Generate specific experimental focus recommendations"""
        comparative = self.unified_results.get('comparative_analysis', {})
        optimal = comparative.get('optimal_pathways', {})
        
        recommendations = {
            'immediate_priority': 'plan_b_polymer_fusion',
            'reasoning': 'Higher technology readiness level and building on proven fusion concepts',
            'timeline': {
                'phase_1_months_0_12': {
                    'focus': 'Plan B Proof-of-Concept',
                    'objectives': [
                        'Validate polymer-plasma interaction models',
                        'Demonstrate enhanced confinement in laboratory plasma',
                        'Characterize polymer material properties at fusion temperatures'
                    ],
                    'required_facilities': [
                        'Small-scale plasma confinement device',
                        'High-temperature materials testing lab',
                        'Computational modeling cluster'
                    ]
                },
                'phase_2_months_12_24': {
                    'focus': 'Plan B Scale-up and Plan A Exploration',
                    'objectives': [
                        'Scale polymer-enhanced fusion to larger plasma volumes',
                        'Begin theoretical and computational work on Plan A',
                        'Develop integrated control systems'
                    ]
                },
                'phase_3_months_24_36': {
                    'focus': 'Parallel Development and Decision Point',
                    'objectives': [
                        'Full-scale polymer-enhanced fusion demonstration',
                        'Plan A proof-of-concept if theoretical validation successful',
                        'Economic viability assessment with real-world data'
                    ]
                }
            },
            'resource_allocation': {
                'plan_b_polymer_fusion': 0.7,  # 70% of resources
                'plan_a_direct_conversion': 0.2,  # 20% of resources
                'supporting_research': 0.1  # 10% for cross-cutting research
            }
        }
        
        return recommendations
    
    def generate_unified_visualization(self, save_path: Optional[str] = None):
        """Generate comprehensive visualization comparing both pathways"""
        if not self.unified_results:
            logger.error("No unified results available. Run comprehensive analysis first.")
            return
        
        plan_a = self.unified_results['plan_a_results']
        plan_b = self.unified_results['plan_b_results']
        
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Cost comparison
        ax1 = fig.add_subplot(gs[0, 0])
        mu_values = plan_a['mu_values']
        ax1.semilogy(mu_values, plan_a['cost_per_kwh_values'], 'b-', linewidth=2, label='Plan A: Direct Conversion')
        
        # Filter extreme values for Plan B
        valid_b_costs = []
        valid_b_mu = []
        for mu, cost in zip(mu_values, plan_b['cost_per_kwh_values']):
            if cost < 10.0:  # Filter extreme values
                valid_b_costs.append(cost)
                valid_b_mu.append(mu)
        
        if valid_b_costs:
            ax1.semilogy(valid_b_mu, valid_b_costs, 'r-', linewidth=2, label='Plan B: Polymer Fusion')
        
        ax1.axhline(y=0.10, color='g', linestyle='--', alpha=0.7, label='Economic Threshold ($0.10/kWh)')
        ax1.set_xlabel('Polymer Scale μ')
        ax1.set_ylabel('Cost per kWh ($)')
        ax1.set_title('Economic Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Energy yield comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.loglog(mu_values, plan_a['energy_yields_kwh'], 'b-', linewidth=2, label='Plan A: Energy Yield')
        
        # Convert Plan B net power to energy (1 hour equivalent)
        plan_b_energies = [max(0, p * 1000) for p in plan_b['net_powers_mw']]
        ax2.loglog(mu_values, plan_b_energies, 'r-', linewidth=2, label='Plan B: Net Energy (1h)')
        
        ax2.set_xlabel('Polymer Scale μ')
        ax2.set_ylabel('Energy Yield (kWh)')
        ax2.set_title('Energy Yield Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Plan B Q-factor
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(mu_values, plan_b['q_factors'], 'r-', linewidth=2)
        ax3.axhline(y=1.0, color='g', linestyle='--', label='Breakeven (Q=1)')
        ax3.axhline(y=10.0, color='orange', linestyle='--', label='Ignition (Q=10)')
        ax3.set_xlabel('Polymer Scale μ')
        ax3.set_ylabel('Q-Factor')
        ax3.set_title('Fusion Q-Factor (Plan B)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Enhancement factors
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(mu_values, plan_a['enhancement_factors'], 'b-', linewidth=2, label='Plan A: Enhancement')
        ax4.plot(mu_values, plan_b['confinement_improvements'], 'r-', linewidth=2, label='Plan B: Confinement')
        ax4.set_xlabel('Polymer Scale μ')
        ax4.set_ylabel('Enhancement Factor')
        ax4.set_title('Polymer Enhancement Factors')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Economic viability regions
        ax5 = fig.add_subplot(gs[1, 1])
        viable_a = [1 if viable else 0 for viable in plan_a['economic_viability']]
        viable_b = [1 if viable else 0 for viable in plan_b['economic_viability']]
        
        ax5.fill_between(mu_values, 0, viable_a, alpha=0.3, color='blue', label='Plan A Viable')
        ax5.fill_between(mu_values, 1, [1 + v for v in viable_b], alpha=0.3, color='red', label='Plan B Viable')
        ax5.set_xlabel('Polymer Scale μ')
        ax5.set_ylabel('Economic Viability')
        ax5.set_title('Economic Viability Regions')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Technology readiness assessment
        ax6 = fig.add_subplot(gs[1, 2])
        trl_data = self.unified_results['comparative_analysis']['technology_readiness']
        
        plans = ['Plan A\nDirect Conversion', 'Plan B\nPolymer Fusion']
        trls = [trl_data['plan_a_direct_conversion']['current_trl'], 
                trl_data['plan_b_polymer_fusion']['current_trl']]
        colors = ['blue', 'red']
        
        bars = ax6.bar(plans, trls, color=colors, alpha=0.7)
        ax6.set_ylabel('Technology Readiness Level')
        ax6.set_title('Current Technology Readiness')
        ax6.set_ylim(0, 9)
        
        # Add TRL labels on bars
        for bar, trl in zip(bars, trls):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'TRL {trl}', ha='center', va='bottom')
        
        # Plot 7-9: Timeline and recommendations (text-based)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Add timeline text
        timeline_text = self._format_timeline_text()
        ax7.text(0.02, 0.95, timeline_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Unified Polymer-Enhanced Energy Analysis\nCalibrated against WEST Tokamak Baseline (Feb 12, 2025)', 
                     fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Unified visualization saved to {save_path}")
        else:
            plt.show()
    
    def _format_timeline_text(self) -> str:
        """Format timeline recommendations as text"""
        recs = self.unified_results['comparative_analysis']['experimental_focus_recommendations']
        
        text = "EXPERIMENTAL FOCUS TIMELINE:\n\n"
        text += f"Immediate Priority: {recs['immediate_priority'].replace('_', ' ').title()}\n"
        text += f"Reasoning: {recs['reasoning']}\n\n"
        
        for phase, details in recs['timeline'].items():
            phase_name = phase.replace('_', ' ').replace('months', 'Months').title()
            text += f"{phase_name}: {details['focus']}\n"
            text += f"  Objectives: {', '.join(details['objectives'][:2])}...\n\n"        
        allocation = recs['resource_allocation']
        text += f"Resource Allocation:\n"
        text += f"  Plan B (Polymer Fusion): {allocation['plan_b_polymer_fusion']*100:.0f}%\n"
        text += f"  Plan A (Direct Conversion): {allocation['plan_a_direct_conversion']*100:.0f}%\n"
        text += f"  Supporting Research: {allocation['supporting_research']*100:.0f}%"
        
        return text
    
    def export_comprehensive_report(self, filepath: str):
        """Export comprehensive analysis report"""
        
        # Generate summary statistics
        summary = self._generate_summary_statistics()
        
        report_data = {
            **self.unified_results,
            'summary_statistics': summary,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert the entire report data
        report_data = convert_numpy_types(report_data)
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Comprehensive report exported to {filepath}")
    
    def _generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for the analysis"""
        plan_a = self.unified_results['plan_a_results']
        plan_b = self.unified_results['plan_b_results']
        
        return {
            'plan_a_summary': {
                'min_cost_per_kwh': min(plan_a['cost_per_kwh_values']),
                'max_energy_yield_kwh': max(plan_a['energy_yields_kwh']),
                'economic_crossover_found': plan_a['economic_crossover_mu'] is not None,
                'viable_mu_range': self._find_viable_range(plan_a['mu_values'], plan_a['economic_viability'])
            },
            'plan_b_summary': {
                'max_q_factor': max(plan_b['q_factors']),
                'max_net_power_mw': max(plan_b['net_powers_mw']),
                'breakeven_achieved': plan_b['breakeven_mu'] is not None,
                'economic_viability_achieved': plan_b['economic_crossover_mu'] is not None
            },
            'west_baseline_comparison': {
                'plan_a_energy_advantage': max(plan_a['energy_yields_kwh']) / 1000,  # Rough comparison
                'plan_b_q_improvement': max(plan_b['q_factors'])  # vs WEST Q~0
            }
        }
    
    def _find_viable_range(self, mu_values: List[float], viability: List[bool]) -> Tuple[float, float]:
        """Find the range of viable mu values"""
        viable_indices = [i for i, v in enumerate(viability) if v]
        if not viable_indices:
            return (None, None)
        return (mu_values[viable_indices[0]], mu_values[viable_indices[-1]])

def run_unified_demonstration():
    """Run complete unified demonstration"""
    print("=" * 80)
    print("UNIFIED POLYMER-ENHANCED ENERGY ANALYSIS")
    print("Calibrated against WEST Tokamak World Record (February 12, 2025)")
    print("=" * 80)
    
    # Initialize analysis
    analyzer = UnifiedPolymerEnergyAnalysis()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(
        mu_range=(0.1, 5.0),
        num_points=30,
        mass_kg=0.001,  # 1 gram
        production_cost_per_kg=1000.0,  # $1000/kg
        reactor_cost_usd=20e9  # $20B reactor
    )
    
    # Display summary
    print("\nANALYSIS SUMMARY:")
    print("-" * 40)
    
    summary = results['comparative_analysis']['experimental_focus_recommendations']
    print(f"Recommended Priority: {summary['immediate_priority'].replace('_', ' ').title()}")
    print(f"Reasoning: {summary['reasoning']}")
    
    optimal = results['comparative_analysis']['optimal_pathways']
    if 'plan_a_optimal' in optimal:
        plan_a_opt = optimal['plan_a_optimal']
        print(f"\nPlan A Optimal: μ={plan_a_opt['mu']:.2f}, Cost=${plan_a_opt['cost_per_kwh']:.3f}/kWh")
    
    if 'plan_b_optimal' in optimal:
        plan_b_opt = optimal['plan_b_optimal']
        print(f"Plan B Optimal: μ={plan_b_opt['mu']:.2f}, Q={plan_b_opt['q_factor']:.2f}, Cost=${plan_b_opt['cost_per_kwh']:.3f}/kWh")
    
    # Generate visualizations
    print("\nGenerating comprehensive visualization...")
    analyzer.generate_unified_visualization("polymer_energy_unified_analysis.png")
    
    # Export report
    print("Exporting comprehensive report...")
    analyzer.export_comprehensive_report("polymer_energy_comprehensive_report.json")
    
    print("\nAnalysis complete! Check generated files for detailed results.")

if __name__ == "__main__":
    run_unified_demonstration()
