"""
Quick Test to Check Result Structure
===================================
"""

from integrated_gut_polymer_optimization import *

def check_result_structure():
    framework = IntegratedPolymerEconomicFramework()
    result = framework.optimize_reactor_for_polymer_scale(5.0, "fusion")
    
    print("Result structure:")
    print(result)
    
    if result['success']:
        print("\nKeys in result:")
        for key in result.keys():
            print(f"  {key}: {type(result[key])}")
        
        if 'economics' in result:
            print("\nKeys in economics:")
            for key in result['economics'].keys():
                print(f"  {key}: {type(result['economics'][key])}")

if __name__ == "__main__":
    check_result_structure()
