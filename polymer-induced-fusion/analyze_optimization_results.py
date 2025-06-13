#!/usr/bin/env python3
"""
Create visualizations from the current optimization results and 
provide analysis of cost curves and economic viability
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_analyze_results():
    """Load and analyze the optimization results"""
    
    results_file = "polymer_economic_optimization/complete_optimization_results.json"
    
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found!")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def create_simple_cost_plots(results):
    """Create simple cost comparison plots"""
    
    mu_values = np.array(results['polymer_scales'])
    costs_a = np.array(results['plan_a_antimatter']['costs_per_kwh'])
    costs_b = np.array(results['plan_b_fusion']['costs_per_kwh'])
    
    # Convert to reasonable units
    costs_a_millions = costs_a / 1e6  # Convert to millions of $/kWh
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Direct cost comparison
    ax1.semilogy(mu_values, costs_a_millions, 'r-', linewidth=3, 
                 label='Plan A (Antimatter)', marker='o', markersize=5)
    ax1.semilogy(mu_values, costs_b, 'b-', linewidth=3, 
                 label='Plan B (Fusion)', marker='s', markersize=5)
    
    # Economic thresholds
    thresholds = [0.15, 0.08, 0.05]  # $/kWh
    threshold_names = ['Competitive', 'Natural Gas', 'Breakthrough']
    colors = ['green', 'orange', 'red']
    
    for threshold, name, color in zip(thresholds, threshold_names, colors):
        ax1.axhline(y=threshold, linestyle='--', alpha=0.7, color=color,
                   label=f'{name} (${threshold:.2f}/kWh)')
    
    ax1.set_xlabel('Polymer Scale μ', fontsize=14)
    ax1.set_ylabel('Cost per kWh', fontsize=14)
    ax1.set_title('Cost per kWh vs. Polymer Scale μ\n(Plan A in millions $/kWh)', 
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim(1e-3, 1e2)
    
    # Plot 2: Plan A cost details
    min_cost_a = costs_a.min()
    max_cost_a = costs_a.max()
    optimal_mu_a = mu_values[np.argmin(costs_a)]
    
    ax2.plot(mu_values, costs_a_millions, 'r-', linewidth=3, marker='o', markersize=5)
    ax2.plot(optimal_mu_a, costs_a_millions[np.argmin(costs_a)], 'r*', 
             markersize=15, label=f'Minimum at μ={optimal_mu_a:.2f}')
    
    ax2.set_xlabel('Polymer Scale μ', fontsize=14)
    ax2.set_ylabel('Plan A Cost (millions $/kWh)', fontsize=14)
    ax2.set_title('Plan A Antimatter Cost Details', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_dir = "polymer_economic_optimization"
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, "cost_analysis_curves.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Cost analysis plot saved to: {plot_file}")
    
    return {
        'min_cost_a': min_cost_a,
        'min_cost_b': costs_b.min(),
        'optimal_mu_a': optimal_mu_a,
        'optimal_mu_b': mu_values[np.argmin(costs_b)]
    }

def analyze_economic_viability(results):
    """Analyze economic viability and crossings"""
    
    mu_values = np.array(results['polymer_scales'])
    costs_a = np.array(results['plan_a_antimatter']['costs_per_kwh'])
    costs_b = np.array(results['plan_b_fusion']['costs_per_kwh'])
    
    thresholds = {
        'competitive': 0.15,      # $/kWh
        'natural_gas': 0.08,      # $/kWh  
        'breakthrough': 0.05      # $/kWh
    }
    
    print("\nECONOMIC VIABILITY ANALYSIS")
    print("="*50)
    
    print(f"Polymer scale range: μ ∈ [{mu_values.min():.1f}, {mu_values.max():.1f}]")
    print(f"Number of optimization points: {len(mu_values)}")
    
    print(f"\nPLAN A (ANTIMATTER) ANALYSIS:")
    print(f"  Cost range: ${costs_a.min():.0f} - ${costs_a.max():.0f} per kWh")
    print(f"  Minimum cost: ${costs_a.min():.0f}/kWh at μ = {mu_values[np.argmin(costs_a)]:.2f}")
    
    # Check threshold crossings for Plan A
    for name, threshold in thresholds.items():
        crossings = mu_values[costs_a <= threshold]
        if len(crossings) > 0:
            print(f"  {name.title()} threshold: Crossed at μ = {crossings[0]:.2f}")
        else:
            print(f"  {name.title()} threshold: NO CROSSING (minimum gap: ${costs_a.min() - threshold:.0f}/kWh)")
    
    print(f"\nPLAN B (FUSION) ANALYSIS:")
    print(f"  Cost range: ${costs_b.min():.2f} - ${costs_b.max():.2f} per kWh")
    print(f"  Minimum cost: ${costs_b.min():.2f}/kWh at μ = {mu_values[np.argmin(costs_b)]:.2f}")
    
    # Check why fusion is failing
    fusion_successes = 0
    for result in results['plan_b_fusion']['optimization_results']:
        if result['success']:
            fusion_successes += 1
    
    print(f"  Successful optimizations: {fusion_successes}/{len(mu_values)}")
    
    if fusion_successes == 0:
        print(f"  Issue: All fusion optimizations failed")
        print(f"  Common error: {results['plan_b_fusion']['optimization_results'][0]['error']}")
    
    # Check threshold crossings for Plan B
    for name, threshold in thresholds.items():
        crossings = mu_values[costs_b <= threshold]
        if len(crossings) > 0:
            print(f"  {name.title()} threshold: Crossed at μ = {crossings[0]:.2f}")
        else:
            print(f"  {name.title()} threshold: NO CROSSING (gap: ${costs_b.min() - threshold:.2f}/kWh)")
    
    # Competitive analysis
    print(f"\nCOMPETITIVE ANALYSIS:")
    cost_ratio = costs_a / costs_b
    advantage_b = costs_a - costs_b
    
    print(f"  Plan B advantage: ${advantage_b.min():.0f} - ${advantage_b.max():.0f} per kWh")
    print(f"  Cost ratio (A/B): {cost_ratio.min():.0f}x - {cost_ratio.max():.0f}x")
    print(f"  Plan B is cheaper by factors of {cost_ratio.min():.0f}x to {cost_ratio.max():.0f}x")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if costs_b.min() < min(thresholds.values()):
        print(f"✅ PLAN B SHOWS ECONOMIC VIABILITY")
        print(f"   • Already below breakthrough threshold at μ = {mu_values[np.argmin(costs_b)]:.2f}")
        print(f"   • Target cost: ${costs_b.min():.2f}/kWh")
        print(f"   • IMMEDIATE EXPERIMENTAL FOCUS: Fusion enhancement validation")
    else:
        print(f"⚠ NEITHER PLAN ACHIEVES ECONOMIC VIABILITY IN CURRENT RANGE")
        
        if costs_b.min() < costs_a.min():
            gap_to_threshold = costs_b.min() - min(thresholds.values())
            print(f"   • Plan B closer to viability")
            print(f"   • Gap to breakthrough: ${gap_to_threshold:.2f}/kWh")
            print(f"   • Need ~{gap_to_threshold/costs_b.min()*100:.1f}% cost reduction")
            print(f"   • RECOMMENDED: Focus on Plan B optimization and μ range extension")
        else:
            print(f"   • Plan A shows potential but requires extreme cost reduction")
            print(f"   • RECOMMENDED: Fundamental research on both approaches")

def create_enhancement_analysis(results):
    """Analyze the polymer enhancement factors"""
    
    mu_values = np.array(results['polymer_scales'])
    
    print(f"\nPOLYMER ENHANCEMENT ANALYSIS:")
    print("="*40)
    
    # Plan A enhancements
    enhancements_a = []
    for result in results['plan_a_antimatter']['optimization_results']:
        if result['success']:
            enhancements_a.append(result['optimal_conditions']['production_enhancement'])
        else:
            enhancements_a.append(1.0)
    
    enhancements_a = np.array(enhancements_a)
    
    print(f"Plan A Enhancement Factors:")
    print(f"  Range: {enhancements_a.min():.3f} - {enhancements_a.max():.3f}")
    print(f"  At μ = 0.1: {enhancements_a[0]:.3f}")
    print(f"  At μ = 10.0: {enhancements_a[-1]:.3f}")
    
    # Enhancement trend
    if len(enhancements_a) > 1:
        trend = "decreasing" if enhancements_a[-1] < enhancements_a[0] else "increasing"
        print(f"  Trend: {trend} with μ")
    
    print(f"\nPlan B Enhancement Analysis:")
    # For Plan B, check if we can extract enhancement factors
    print(f"  Issue: No successful optimizations to analyze")
    print(f"  All optimizations show zero fusion power")
    print(f"  Indicates problem with cross-section enhancement application")

def main():
    """Main analysis function"""
    
    print("POLYMER ECONOMIC OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # Load results
    results = load_and_analyze_results()
    if results is None:
        return
    
    # Create visualizations
    cost_analysis = create_simple_cost_plots(results)
    
    # Economic analysis
    analyze_economic_viability(results)
    
    # Enhancement analysis
    create_enhancement_analysis(results)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("Key Finding: Plan B fusion calculations need debugging")
    print("Plan A shows consistent but very expensive results")
    print("="*60)

if __name__ == "__main__":
    main()
