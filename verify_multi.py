import pandas as pd
from cashflow import CashFlowKOR, CompanyConfig, MultiCompanyCashFlow
from development import DevelopmentCost

def test_multi_company():
    # 1. Setup a simple project
    dev_param = {
        'test_case': {
            'drilling_cost': 100.0, # Large drilling cost for exploration
            'Subsea_cost': 0.0,
            'OPEX_fixed': 10.0,
            'OPEX_per_bcf': 1.0,
            'ABEX_per_well': 0.0,
        }
    }
    dev = DevelopmentCost(dev_start_year=2024, dev_param=dev_param, development_case='test_case')
    dev.set_drilling_schedule(2025, {0: 1}) # 1 well in 2025
    # Add exploration cost manually to test farm-in
    dev.exploration_costs = {2024: 10.0, 2025: 10.0} # Total 20 MM$
    
    # Set production
    dev.set_annual_production({2026: 10.0}, {2026: 0.0}) # 10 BCF in 2026
    dev.calculate_total_costs()
    
    cf = CashFlowKOR(
        base_year=2024,
        oil_price_by_year={2026: 0.0},
        gas_price_by_year={2026: 5.0}, # Revenue 50 MM$
        cost_inflation_rate=0.0,
        discount_rate=0.1,
        exchange_rate=1350.0
    )
    cf.set_development_costs(dev, output=False)
    cf.calculate_annual_revenue(output=False)
    cf.calculate_royalty()
    cf.calculate_taxes(output=False)
    cf.calculate_net_cash_flow(output=False)
    cf.calculate_npv(output=False)
    
    print(f"Project Total NPV: {cf.npv:.2f}")
    
    # 2. Setup Companies
    # Company A: 51% PI, 100% carry on first 15 MM$ of exploration
    compA = CompanyConfig(name="CompA", pi=0.51, farm_in_expo_share=1.0, farm_in_expo_cap=15.0)
    # Company B: 49% PI
    compB = CompanyConfig(name="CompB", pi=0.49)
    
    multi = MultiCompanyCashFlow(cf, [compA, compB])
    results = multi.calculate()
    
    summ = multi.get_summary_df()
    print("\nMulti-Company Summary:")
    print(summ)
    
    # Checks
    # Total Revenue check
    sum_rev = results['CompA'].total_revenue + results['CompB'].total_revenue
    assert abs(sum_rev - cf.total_revenue) < 0.01, f"Revenue mismatch: {sum_rev} vs {cf.total_revenue}"
    
    # Exploration cost split check
    # Project Explo: 10 (2024), 10 (2025). Total 20.
    # Cap is 15.
    # Year 2024: Proj=10. CumBefore=0. Under=10, Over=0. CompA=10*1.0 = 10.
    # Year 2025: Proj=10. CumBefore=10. Under=5, Over=5. CompA=5*1.0 + 5*0.51 = 5 + 2.55 = 7.55.
    # Total CompA Explo: 10 + 7.55 = 17.55.
    # CompB Explo: 20 - 17.55 = 2.45.
    
    # Wait, CompB is 49% PI.
    # Year 2024: CompB = 10 * 0 = 0.
    # Year 2025: CompB = 5 * 0 + 5 * 0.49 = 2.45.
    # Total CompB Explo: 0 + 2.45 = 2.45. Correct.
    
    compA_explo = sum(results['CompA'].capex_breakdown['exploration'].values())
    print(f"\nCompA Exploration Cost: {compA_explo:.2f}")
    assert abs(compA_explo - 17.55) < 0.01, f"CompA explo mismatch: {compA_explo}"

    print("\n✅ Verification Successful!")

if __name__ == "__main__":
    test_multi_company()
