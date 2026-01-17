import sys
import os
sys.path.append(os.getcwd())

from cashflow import CashFlowKOR
import numpy as np

def test_inflated_breakdown():
    cf = CashFlowKOR(cost_inflation_rate=0.10) # 10% inflation for easy checking
    
    # Mock development data
    cost_years = [2025, 2026]
    annual_capex = {2025: 100.0, 2026: 200.0}
    capex_breakdown = {
        'drilling': {2025: 60.0, 2026: 120.0},
        'subsea': {2025: 40.0, 2026: 80.0}
    }
    
    dev = {
        'cost_years': cost_years,
        'annual_capex': annual_capex,
        'annual_opex': {},
        'annual_abex': {},
        'capex_breakdown': capex_breakdown
    }
    
    cf.set_development_costs(dev, output=False)
    
    # Check inflated capex
    # Year 0: 100 * (1.1^0) = 100
    # Year 1: 200 * (1.1^1) = 220
    assert cf.annual_capex_inflated[2025] == 100.0
    assert abs(cf.annual_capex_inflated[2026] - 220.0) < 1e-6
    
    # Check inflated breakdown
    # drilling 2025: 60 * 1 = 60
    # drilling 2026: 120 * 1.1 = 132
    assert cf.capex_breakdown_inflated['drilling'][2025] == 60.0
    assert abs(cf.capex_breakdown_inflated['drilling'][2026] - 132.0) < 1e-6
    
    # subsea 2025: 40 * 1 = 40
    # subsea 2026: 80 * 1.1 = 88
    assert cf.capex_breakdown_inflated['subsea'][2025] == 40.0
    assert abs(cf.capex_breakdown_inflated['subsea'][2026] - 88.0) < 1e-6
    
    # Check sum
    for y in cost_years:
        breakdown_sum = sum(v.get(y, 0.0) for v in cf.capex_breakdown_inflated.values())
        assert abs(breakdown_sum - cf.annual_capex_inflated[y]) < 1e-6
    
    print("Inflated breakdown verification passed!")

if __name__ == "__main__":
    test_inflated_breakdown()
