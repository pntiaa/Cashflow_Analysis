import pandas as pd
from typing import Dict
from development import QuestorDevelopmentCost

# Mock class to bypass excel loading
class MockQuestorDev(QuestorDevelopmentCost):
    def __init__(self, dev_start_year: int):
        # Bypass super().__init__ file loading by calling grandparent init and manual setup
        # But since QuestorDevelopmentCost.__init__ calls load_questor_file, we need to mock it effectively.
        # Easiest way is to override load_questor_file to do nothing, then call super().__init__
        pass 
        
    def load_questor_file(self, excel_file_path: str, sheet_name: str):
        # Simulate data loading
        self.questor_raw = {
            'PROJECT CAPEX': {2026: 100.0, 2027: 50.0},
            'Facilities CAPEX': {2026: 10.0, 2027: 10.0},
            'OPEX': {2028: 5.0, 2029: 5.0},
            'Gas Bscf': {2028: 100, 2029: 100}
        }
        self._set_annual_production(self.questor_raw)
        self._set_annual_costs(self.questor_raw)
        self.update_summary_metrics()

def test_exploration_inclusion():
    print("--- Testing Exploration Cost Inclusion ---")
    
    # 1. Initialize Mock
    # We'll create a subclass instance but we need to initialize it properly.
    # Since we can't easily change the class inheritance at runtime for the init call without modifying source,
    # let's just make a dummy file path and ensure our overridden load_questor_file is called.
    
    # Actually, Python resolves methods dynamically. If I define MockQuestorDev with overridden load_questor_file, 
    # and call super().__init__, it will use the overridden method.
    
    class TestDev(QuestorDevelopmentCost):
        def load_questor_file(self, excel_file_path: str, sheet_name: str):
            print("[Test] Mocking load_questor_file")
            self.questor_raw = {
                'PROJECT CAPEX': {0: 100.0, 1: 0.0, 2: 0.0}, 
                'OPEX': {0: 0.0, 1: 0.0, 2: 10.0}
            }
            self._set_annual_production(self.questor_raw)
            self._set_annual_costs(self.questor_raw)
            self.update_summary_metrics()
            
    dev = TestDev(dev_start_year=2026, excel_file_path="dummy", sheet_name="dummy")
    
    print(f"Initial Total CAPEX: {dev.total_capex}")
    print(f"Annual CAPEX: {dev.annual_capex}")
    
    # 2. Set Exploration Costs
    exploration_costs = {2024: 5.0, 2025: 5.0} # Years before dev_start
    print(f"Setting Exploration Costs: {exploration_costs}")
    
    dev.set_exploration_stage(exploration_costs=exploration_costs, output=True)
    
    # 3. Check Results
    print(f"Total CAPEX after exploration: {dev.total_capex}")
    print(f"Annual CAPEX after exploration: {dev.annual_capex}")
    
    # Verification
    expected_capex = 100.0 + 5.0 + 5.0
    if abs(dev.total_capex - expected_capex) < 0.01:
        print("✅ SUCCESS: Exploration costs included in Total CAPEX")
    else:
        print(f"❌ FAILURE: Expected {expected_capex}, got {dev.total_capex}")

    if 2024 in dev.annual_capex and dev.annual_capex[2024] == 5.0:
         print("✅ SUCCESS: Exploration cost for 2024 found in Annual CAPEX")
    else:
         print("❌ FAILURE: Exploration cost for 2024 NOT found in Annual CAPEX")

if __name__ == "__main__":
    test_exploration_inclusion()
