"""
Polymer-Enhanced Energy Research Main Runner
==========================================

This script provides the main entry point for running polymer-enhanced energy
research using both Plan A (Direct Mass-Energy Conversion) and Plan B 
(Polymer-Enhanced Fusion) pathways.

Usage:
    python main.py [options]

Options:
    --plan {a,b,unified}    Run specific plan or unified analysis (default: unified)
    --mu-range MIN MAX      Polymer scale parameter range (default: 0.1 10.0)
    --points N              Number of analysis points (default: 50)
    --mass KG               Mass for Plan A analysis in kg (default: 0.001)
    --cost-per-kg USD       Production cost per kg for Plan A (default: 1000.0)
    --reactor-cost USD      Reactor cost for Plan B (default: 20e9)
    --output-dir DIR        Output directory for results (default: results/)
    --visualize            Generate visualizations
    --export-report        Export comprehensive JSON report
"""

import argparse
import os
import sys
from datetime import datetime
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plan_a_direct_mass_energy import demonstrate_plan_a, PolymerMassEnergyPipeline, WESTBaseline
from plan_b_polymer_fusion import demonstrate_plan_b, PolymerFusionPipeline
from unified_polymer_energy_analysis import UnifiedPolymerEnergyAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_directory(output_dir: str) -> str:
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    return output_dir

def run_plan_a(args):
    """Run Plan A: Direct Mass-Energy Conversion"""
    logger.info("Running Plan A: Direct Mass-Energy Conversion")
    
    if args.demo:
        demonstrate_plan_a()
        return
    
    # Initialize pipeline
    west = WESTBaseline()
    pipeline = PolymerMassEnergyPipeline(west)
    
    # Run analysis
    results = pipeline.run_polymer_scale_sweep(
        mu_range=(args.mu_range[0], args.mu_range[1]),
        num_points=args.points,
        mass_kg=args.mass,
        production_cost_per_kg=args.cost_per_kg
    )
    
    # Generate outputs
    output_dir = create_output_directory(args.output_dir)
    
    if args.visualize:
        viz_path = os.path.join(output_dir, f"plan_a_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        pipeline.generate_visualization(viz_path)
    
    if args.export_report:
        report_path = os.path.join(output_dir, f"plan_a_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        pipeline.save_results(report_path)
    
    # Print summary
    print_plan_a_summary(results)

def run_plan_b(args):
    """Run Plan B: Polymer-Enhanced Fusion"""
    logger.info("Running Plan B: Polymer-Enhanced Fusion")
    
    if args.demo:
        demonstrate_plan_b()
        return
    
    # Initialize pipeline
    west = WESTBaseline()
    pipeline = PolymerFusionPipeline(west)
    
    # Run analysis
    results = pipeline.run_polymer_scale_sweep(
        mu_range=(args.mu_range[0], args.mu_range[1]),
        num_points=args.points,
        reactor_cost_usd=args.reactor_cost
    )
    
    # Generate outputs
    output_dir = create_output_directory(args.output_dir)
    
    if args.visualize:
        viz_path = os.path.join(output_dir, f"plan_b_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        pipeline.generate_fusion_visualization(viz_path)
    
    if args.export_report:
        report_path = os.path.join(output_dir, f"plan_b_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        pipeline.save_results(report_path)
    
    # Print summary
    print_plan_b_summary(results)

def run_unified(args):
    """Run Unified Analysis of both plans"""
    logger.info("Running Unified Polymer-Enhanced Energy Analysis")
    
    # Initialize analyzer
    analyzer = UnifiedPolymerEnergyAnalysis()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(
        mu_range=(args.mu_range[0], args.mu_range[1]),
        num_points=args.points,
        mass_kg=args.mass,
        production_cost_per_kg=args.cost_per_kg,
        reactor_cost_usd=args.reactor_cost
    )
    
    # Generate outputs
    output_dir = create_output_directory(args.output_dir)
    
    if args.visualize:
        viz_path = os.path.join(output_dir, f"unified_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        analyzer.generate_unified_visualization(viz_path)
    
    if args.export_report:
        report_path = os.path.join(output_dir, f"unified_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        analyzer.export_comprehensive_report(report_path)
    
    # Print summary
    print_unified_summary(results)

def print_plan_a_summary(results):
    """Print Plan A summary"""
    print("\n" + "="*60)
    print("PLAN A: DIRECT MASS-ENERGY CONVERSION SUMMARY")
    print("="*60)
    
    if results['economic_crossover_mu'] is not None:
        print(f"Economic Crossover: μ = {results['economic_crossover_mu']:.3f}")
        print(f"Crossover Cost: ${results['crossover_cost_per_kwh']:.4f}/kWh")
    else:
        print("No economic crossover found in tested range")
    
    max_energy_idx = results['energy_yields_kwh'].index(max(results['energy_yields_kwh']))
    print(f"Maximum Energy Yield: {max(results['energy_yields_kwh']):.2e} kWh at μ={results['mu_values'][max_energy_idx]:.2f}")
    
    min_cost = min(results['cost_per_kwh_values'])
    min_cost_idx = results['cost_per_kwh_values'].index(min_cost)
    print(f"Minimum Cost: ${min_cost:.4f}/kWh at μ={results['mu_values'][min_cost_idx]:.2f}")

def print_plan_b_summary(results):
    """Print Plan B summary"""
    print("\n" + "="*60)
    print("PLAN B: POLYMER-ENHANCED FUSION SUMMARY")
    print("="*60)
    
    if results['breakeven_mu'] is not None:
        print(f"Fusion Breakeven (Q≥1): μ = {results['breakeven_mu']:.3f}")
    else:
        print("No fusion breakeven achieved in tested range")
    
    if results['economic_crossover_mu'] is not None:
        print(f"Economic Viability: μ = {results['economic_crossover_mu']:.3f}")
    else:
        print("No economic viability achieved in tested range")
    
    max_q = max(results['q_factors'])
    max_q_idx = results['q_factors'].index(max_q)
    print(f"Maximum Q-Factor: {max_q:.3f} at μ={results['mu_values'][max_q_idx]:.2f}")
    
    max_power = max(results['net_powers_mw'])
    max_power_idx = results['net_powers_mw'].index(max_power)
    print(f"Maximum Net Power: {max_power:.2f} MW at μ={results['mu_values'][max_power_idx]:.2f}")

def print_unified_summary(results):
    """Print unified analysis summary"""
    print("\n" + "="*80)
    print("UNIFIED POLYMER-ENHANCED ENERGY ANALYSIS SUMMARY")
    print("Calibrated against WEST Tokamak World Record (February 12, 2025)")
    print("="*80)
    
    west = results['west_baseline']
    print(f"\nWEST Baseline:")
    print(f"  Confinement Time: {west['confinement_time_s']:.0f} s")
    print(f"  Plasma Temperature: {west['plasma_temperature_c']/1e6:.0f}×10⁶ °C")
    print(f"  Heating Power: {west['heating_power_w']/1e6:.1f} MW")
    
    comparative = results['comparative_analysis']
    recommendations = comparative['experimental_focus_recommendations']
    
    print(f"\nRecommended Experimental Focus:")
    print(f"  Priority: {recommendations['immediate_priority'].replace('_', ' ').title()}")
    print(f"  Reasoning: {recommendations['reasoning']}")
    
    resource_allocation = recommendations['resource_allocation']
    print(f"\nResource Allocation:")
    print(f"  Plan B (Polymer Fusion): {resource_allocation['plan_b_polymer_fusion']*100:.0f}%")
    print(f"  Plan A (Direct Conversion): {resource_allocation['plan_a_direct_conversion']*100:.0f}%")
    print(f"  Supporting Research: {resource_allocation['supporting_research']*100:.0f}%")
    
    optimal = comparative['optimal_pathways']
    if 'plan_a_optimal' in optimal:
        plan_a_opt = optimal['plan_a_optimal']
        print(f"\nPlan A Optimal Point:")
        print(f"  μ = {plan_a_opt['mu']:.3f}")
        print(f"  Cost = ${plan_a_opt['cost_per_kwh']:.4f}/kWh")
        print(f"  Economically Viable: {plan_a_opt['economically_viable']}")
    
    if 'plan_b_optimal' in optimal:
        plan_b_opt = optimal['plan_b_optimal']
        print(f"\nPlan B Optimal Point:")
        print(f"  μ = {plan_b_opt['mu']:.3f}")
        print(f"  Q-Factor = {plan_b_opt['q_factor']:.3f}")
        print(f"  Cost = ${plan_b_opt['cost_per_kwh']:.4f}/kWh")
        print(f"  Economically Viable: {plan_b_opt['economically_viable']}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Polymer-Enhanced Energy Research Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--plan', 
        choices=['a', 'b', 'unified'], 
        default='unified',
        help='Run specific plan or unified analysis (default: unified)'
    )
    
    parser.add_argument(
        '--mu-range', 
        nargs=2, 
        type=float, 
        default=[0.1, 10.0],
        metavar=('MIN', 'MAX'),
        help='Polymer scale parameter range (default: 0.1 10.0)'
    )
    
    parser.add_argument(
        '--points', 
        type=int, 
        default=50,
        help='Number of analysis points (default: 50)'
    )
    
    parser.add_argument(
        '--mass', 
        type=float, 
        default=0.001,
        help='Mass for Plan A analysis in kg (default: 0.001)'
    )
    
    parser.add_argument(
        '--cost-per-kg', 
        type=float, 
        default=1000.0,
        help='Production cost per kg for Plan A (default: 1000.0)'
    )
    
    parser.add_argument(
        '--reactor-cost', 
        type=float, 
        default=20e9,
        help='Reactor cost for Plan B (default: 2e10)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='results',
        help='Output directory for results (default: results/)'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--export-report', 
        action='store_true',
        help='Export comprehensive JSON report'
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run demonstration mode'
    )
    
    args = parser.parse_args()
    
    # Run selected analysis
    try:
        if args.plan == 'a':
            run_plan_a(args)
        elif args.plan == 'b':
            run_plan_b(args)
        elif args.plan == 'unified':
            run_unified(args)
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
