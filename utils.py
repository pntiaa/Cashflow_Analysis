import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List

# --- Session State Initialization ---

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
DEFAULTS_FILE = DATA_DIR / "defaults.json"

def load_defaults():
    if DEFAULTS_FILE.exists():
        with open(DEFAULTS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_defaults(defaults):
    with open(DEFAULTS_FILE, "w") as f:
        json.dump(defaults, f, indent=4)

def ensure_state_init():
    if "current_project" not in st.session_state:
        st.session_state.current_project = None
    if "production_cases" not in st.session_state:
        st.session_state.production_cases = {}
    if "development_cases" not in st.session_state:
        st.session_state.development_cases = {}
    if "price_cases" not in st.session_state:
        st.session_state.price_cases = {}
    if "cashflow_results" not in st.session_state:
        st.session_state.cashflow_results = {}
    # UI transient states
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "tc_data" not in st.session_state:
        st.session_state.tc_data = None
    if "prod_data" not in st.session_state:
        st.session_state.prod_data = None
    if "drilling_plan_results" not in st.session_state:
        st.session_state.drilling_plan_results = None
    if "defaults" not in st.session_state:
        st.session_state.defaults = load_defaults()
    
    # Initialize input parameters from defaults if not present
    if st.session_state.defaults:
        d = st.session_state.defaults
        p_keys = {
            "qi_input": d["production"]["qi_mmcfd"],
            "well_eur_input": d["production"]["well_eur_bcf"],
            "prod_dur_input": d["production"]["prod_duration"],
            "giip_input": d["production"]["giip_bcf"],
            "oiip_mmbbl": d["production"]["oiip_mmbbl"],
            "drilling_rate_input": d["production"]["drilling_rate"],
            "max_rate_input": d["production"]["max_prod_rate"],
            "sunk_cost_input": d["development"]["sunk_cost"],
            "exp_start_year_input": d["development"]["exp_start_year"],
            "dev_start_year_input": d["development"]["dev_start_year"],
            "drill_start_year_input": d["development"]["drill_start_year"],
            "dev_case_input": d["development"]["dev_case"],
            "feas_study_input": d["development"]["study_costs"]["feasibility"]["cost"],
            "feas_study_t": d["development"]["study_costs"]["feasibility"]["timing"],
            "feas_study_d": d["development"]["study_costs"]["feasibility"]["duration"],
            "concept_study_input": d["development"]["study_costs"]["concept"]["cost"],
            "concept_study_t": d["development"]["study_costs"]["concept"]["timing"],
            "concept_study_d": d["development"]["study_costs"]["concept"]["duration"],
            "feed_cost_input": d["development"]["study_costs"]["feed_fpso"]["cost"],
            "feed_cost_t": d["development"]["study_costs"]["feed_fpso"]["timing"],
            "feed_cost_d": d["development"]["study_costs"]["feed_fpso"]["duration"],
            "pm_others_input": d["development"]["study_costs"]["pm_others_per_well"],
            "drilling_cost_input": d["development"]["facility_capex"]["drilling_per_well"],
            "subsea_cost_input": d["development"]["facility_capex"]["subsea_per_well"],
            "fpso_cost_input": d["development"]["facility_capex"]["fpso_fpso"]["cost"],
            "fpso_cost_t": d["development"]["facility_capex"]["fpso_fpso"]["timing"],
            "fpso_cost_d": d["development"]["facility_capex"]["fpso_fpso"]["duration"],
            "pipeline_cost_input": d["development"]["facility_capex"]["pipeline_fpso"]["cost"],
            "pipeline_cost_t": d["development"]["facility_capex"]["pipeline_fpso"]["timing"],
            "pipeline_cost_d": d["development"]["facility_capex"]["pipeline_fpso"]["duration"],
            "terminal_cost_input": d["development"]["facility_capex"]["terminal_fpso"]["cost"],
            "terminal_cost_t": d["development"]["facility_capex"]["terminal_fpso"]["timing"],
            "terminal_cost_d": d["development"]["facility_capex"]["terminal_fpso"]["duration"],
            "opex_per_bcf_input": d["development"]["opex_abex"]["opex_per_bcf"],
            "opex_fixed_input": d["development"]["opex_abex"]["opex_fixed"],
            "abex_per_well_input": d["development"]["opex_abex"]["abex_per_well"],
            "abex_fpso_input": d["development"]["opex_abex"]["abex_fpso_fpso"],
            "base_year_input": d["economics"]["base_year"],
            "discount_rate_input": d["economics"]["discount_rate"],
            "exchange_rate_input": d["economics"]["exchange_rate"],
            "cost_inflation_input": d["economics"]["cost_inflation"],
            "useful_life_input": d["economics"]["useful_life"],
            "depreciation_method_input": d["economics"]["depreciation_method"],
            "oil_init_input": d["price_deck"]["oil_init"],
            "gas_init_input": d["price_deck"]["gas_init"],
            "price_start_year_input": d["price_deck"]["start_year"],
            "price_end_year_input": d["price_deck"]["end_year"],
            "price_inflation_input": d["price_deck"]["inflation_pct"],
            "price_cap_2x_input": d["price_deck"]["cap_2x"],
            "q_dev_start_year": d["development"]["dev_start_year"],
            "q_sunk_cost": d["development"]["sunk_cost"],
            "q_exp_start_year": d["development"]["exp_start_year"],
        }
        for k, v in p_keys.items():
            if k not in st.session_state:
                st.session_state[k] = v
        
        # Also initialize price_deck_oil and gas if not present
        if "price_deck_oil" not in st.session_state:
            start = d["price_deck"]["start_year"]
            end = d["price_deck"]["end_year"]
            init_oil = d["price_deck"]["oil_init"]
            init_gas = d["price_deck"]["gas_init"]
            st.session_state.price_deck_oil = {y: init_oil for y in range(start, end + 1)}
            st.session_state.price_deck_gas = {y: init_gas for y in range(start, end + 1)}

def sanitize_data(data):
    """
    Recursively converts numpy types (int64, float64, ndarray) to native Python types (int, float, list).
    Ensures dictionary keys are also converted to native strings or ints.
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            # Sanitize key
            if isinstance(k, (np.integer, np.int64)):
                k = int(k)
            elif isinstance(k, (np.floating, np.float64)):
                k = float(k)
            # Sanitize value
            new_dict[k] = sanitize_data(v)
        return new_dict
    elif isinstance(data, list):
        return [sanitize_data(i) for i in data]
    elif isinstance(data, (np.integer, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return sanitize_data(data.tolist())
    else:
        return data

def save_project(project_name: str):
    if not project_name:
        return
    file_path = DATA_DIR / f"{project_name}.json"
    
    # We only save the major case dictionaries. 
    # Note: development_cases contains objects that need special handling.
    # For now, let's filter out non-serializable objects or convert them.
    
    raw_data = {
        "production_cases": st.session_state.production_cases,
        "development_cases": serialize_dev_cases(st.session_state.development_cases),
        "price_cases": st.session_state.price_cases,
        "cashflow_results": serialize_cashflow_results(st.session_state.cashflow_results)
    }
    
    # Sanitize all data to ensure no numpy types cause JSON errors
    data_to_save = sanitize_data(raw_data)
    
    with open(file_path, "w") as f:
        json.dump(data_to_save, f, indent=4)

def load_project(project_name: str):
    file_path = DATA_DIR / f"{project_name}.json"
    if not file_path.exists():
        return
    
    with open(file_path, "r") as f:
        data = json.load(f)
        
    # Convert string keys back to int for production profiles
    production_cases = data.get("production_cases", {})
    for case in production_cases.values():
        if "profiles" in case:
            for prof_type in ["gas", "oil", "drilling_plan"]:
                if prof_type in case["profiles"]:
                    case["profiles"][prof_type] = {int(k): v for k, v in case["profiles"][prof_type].items()}
    
    # Convert string keys back to int for price scenarios
    price_cases = data.get("price_cases", {})
    for case in price_cases.values():
        for price_type in ["oil", "gas"]:
            if price_type in case:
                case[price_type] = {int(k): v for k, v in case[price_type].items()}
                
    st.session_state.production_cases = production_cases
    st.session_state.development_cases = deserialize_dev_cases(data.get("development_cases", {}))
    st.session_state.price_cases = price_cases
    st.session_state.cashflow_results = deserialize_cashflow_results(data.get("cashflow_results", {}))
    st.session_state.current_project = project_name

def list_projects():
    # Bypass 'defaults' and other non-project files
    return [f.stem for f in DATA_DIR.glob("*.json") if f.stem != "defaults"]

def delete_project(project_name: str):
    file_path = DATA_DIR / f"{project_name}.json"
    if file_path.exists():
        os.remove(file_path)
        if st.session_state.get("current_project") == project_name:
            st.session_state.current_project = None
            st.session_state.production_cases = {}
            st.session_state.development_cases = {}
            st.session_state.price_cases = {}
            st.session_state.cashflow_results = {}
        return True
    return False

def delete_case(project_name: str, case_type: str, case_name: str):
    file_path = DATA_DIR / f"{project_name}.json"
    if not file_path.exists():
        return False
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Mapping readable types to dict keys
    type_map = {
        "Production": "production_cases",
        "Development": "development_cases",
        "Price": "price_cases",
        "Cash Flow Result": "cashflow_results"
    }
    
    key = type_map.get(case_type)
    if key and key in data and case_name in data[key]:
        del data[key][case_name]
        
        # Save back to file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        
        # If it's the current project, update session state too
        if st.session_state.get("current_project") == project_name:
            if key == "production_cases": del st.session_state.production_cases[case_name]
            elif key == "development_cases": del st.session_state.development_cases[case_name]
            elif key == "price_cases": del st.session_state.price_cases[case_name]
            elif key == "cashflow_results": del st.session_state.cashflow_results[case_name]
        
        return True
    return False

def render_project_sidebar():
    """Renders the project management section in the sidebar. Shared across all pages."""
    with st.sidebar:
        st.title("📁 Project Management")
        
        project_option = st.radio("Action", ["Select Existing", "Create New"], key="project_action_radio")

        if project_option == "Create New":
            new_project_name = st.text_input("New Project Name", key="new_project_name_input")
            if st.button("➕ Create Project", key="create_project_btn"):
                if new_project_name:
                    # Initialize empty state for new project
                    st.session_state.current_project = new_project_name
                    st.session_state.production_cases = {}
                    st.session_state.development_cases = {}
                    st.session_state.price_cases = {}
                    st.session_state.cashflow_results = {}
                    save_project(new_project_name)
                    st.success(f"Project '{new_project_name}' created!")
                    st.rerun()
                else:
                    st.error("Please enter a name.")

        else:
            existing_projects = list_projects()
            if existing_projects:
                # Determine current index
                try:
                    current_idx = existing_projects.index(st.session_state.current_project) if st.session_state.current_project in existing_projects else 0
                except ValueError:
                    current_idx = 0
                    
                selected_project = st.selectbox("Select Project", existing_projects, index=current_idx, key="select_project_selectbox")
                
                if st.button("📂 Load Project", key="load_project_btn"):
                    load_project(selected_project)
                    st.success(f"Loaded '{selected_project}'")
                    st.rerun()
            else:
                st.info("No projects found. Create one!")

        if st.session_state.get("current_project"):
            st.markdown(f"---")
            st.markdown(f"**Current Project:** `{st.session_state.current_project}`")
        else:
            st.warning("⚠️ No project active. Data will NOT be saved.")
        
        st.divider()

def serialize_dev_cases(dev_cases):
    """Converts DevelopmentCost objects and production data to serializable dicts."""
    serialized = {}
    for name, case in dev_cases.items():
        # Keep the structure but handle the 'dev_obj'
        new_case = case.copy()
        if "dev_obj" in new_case:
            dev_obj = new_case["dev_obj"]
            # Extract essential parameters to recreate it
            new_case["dev_params_info"] = {
                "dev_start_year": dev_obj.dev_start_year,
                "dev_param": dev_obj.dev_param,
                "development_case": dev_obj.development_case,
                "drill_start_year": dev_obj.drill_start_year,
                "yearly_drilling_schedule": dev_obj.yearly_drilling_schedule,
                "annual_gas_production": dev_obj.annual_gas_production,
                "annual_oil_production": dev_obj.annual_oil_production,
                # Persistence for pre-calculated costs (Questor/Direct)
                "annual_capex": dev_obj.annual_capex,
                "annual_opex": dev_obj.annual_opex,
                "annual_abex": dev_obj.annual_abex,
                "total_annual_costs": dev_obj.total_annual_costs,
                "cost_years": dev_obj.cost_years
            }
            del new_case["dev_obj"]
        
        # Ensure production parameters are also serialized if they are in the case root
        # (Combined case structure)
        serialized[name] = new_case
    return serialized

def deserialize_dev_cases(serialized_cases):
    """Reconstructs DevelopmentCost objects and handles production profile data."""
    from development import DevelopmentCost
    deserialized = {}
    for name, case in serialized_cases.items():
        new_case = case.copy()
        
        # Handle production profile string-to-int key conversion if embedded
        if "prod_profiles" in new_case:
            for prof_type in ["gas", "oil", "drilling_plan"]:
                if prof_type in new_case["prod_profiles"]:
                    new_case["prod_profiles"][prof_type] = {int(k): v for k, v in new_case["prod_profiles"][prof_type].items()}

        if "dev_params_info" in new_case:
            info = new_case["dev_params_info"]
            dev_obj = DevelopmentCost(
                dev_start_year=info["dev_start_year"],
                dev_param=info["dev_param"],
                development_case=info["development_case"]
            )
            # Re-apply schedule and production (only if schedule exists)
            schedule = {int(k): v for k, v in info.get("yearly_drilling_schedule", {}).items()}
            if schedule:
                dev_obj.set_drilling_schedule(
                    drill_start_year=info.get("drill_start_year", info["dev_start_year"]),
                    yearly_drilling_schedule=schedule,
                    already_shifted=True,
                    output=False
                )
            
            dev_obj.set_annual_production(
                annual_gas_production={int(k): v for k, v in info["annual_gas_production"].items()},
                annual_oil_production={int(k): v for k, v in info["annual_oil_production"].items()},
                already_shifted=True,
                output=False
            )
            
            # Restore pre-calculated costs if they exist (for cases without drilling schedule like Questor)
            if "annual_capex" in info and info["annual_capex"]:
                dev_obj.annual_capex = {int(k): float(v) for k, v in info["annual_capex"].items()}
                dev_obj.annual_opex = {int(k): float(v) for k, v in info["annual_opex"].items()}
                dev_obj.annual_abex = {int(k): float(v) for k, v in info["annual_abex"].items()}
                dev_obj.total_annual_costs = {int(k): float(v) for k, v in info["total_annual_costs"].items()}
                dev_obj.cost_years = [int(y) for y in info.get("cost_years", sorted(list(dev_obj.annual_capex.keys())))]
                
                # Re-calculate cumulative costs for the restored data
                cum = 0.0
                dev_obj.cumulative_costs = {}
                for y in sorted(dev_obj.total_annual_costs.keys()):
                    cum += dev_obj.total_annual_costs[y]
                    dev_obj.cumulative_costs[y] = cum
            
            # Only calculate if we have a drilling schedule to work with and NO pre-calculated costs
            if schedule and not ("annual_capex" in info and info["annual_capex"]):
                dev_obj.calculate_total_costs(output=False)
            
            new_case["dev_obj"] = dev_obj
            del new_case["dev_params_info"]
        deserialized[name] = new_case
    return deserialized

def serialize_cashflow_results(results):
    """Converts CashFlowKOR result objects to serializable dicts."""
    serialized = {}
    for name, item in results.items():
        # item['cf'] is the CashFlowKOR object
        # item['inputs'] is dict of {prod_name, dev_name, price_name, global_params}
        
        entry = {'inputs': item.get('inputs', {})}
        
        if 'cf' in item:
            cf = item['cf']
            if hasattr(cf, 'model_dump'):
                 cf_dict = cf.model_dump()
            else:
                 cf_dict = cf.dict() # For Pydantic v1
            
            # Remove complex objects or redundant large objects
            if 'development_cost' in cf_dict:
                del cf_dict['development_cost']
            
            entry['cf_data'] = cf_dict
        
        if 'result_summary' in item:
            entry['result_summary'] = item['result_summary']
            
        serialized[name] = entry
    return serialized

def deserialize_cashflow_results(data):
    """Reconstructs CashFlowKOR objects from data."""
    from cashflow import CashFlowKOR
    results = {}
    for name, item in data.items():
        cf_data = item.get('cf_data')
        inputs = item.get('inputs', {})
        
        try:
            res_item = {'inputs': inputs}
            if cf_data:
                cf = CashFlowKOR(**cf_data)
                res_item['cf'] = cf
            
            if 'result_summary' in item:
                res_item['result_summary'] = item['result_summary']
            
            if 'cf' in res_item or 'result_summary' in res_item:
                results[name] = res_item
                
        except Exception as e:
            print(f"Error deserializing cashflow result '{name}': {e}")
            
    return results

# --- PriceDeck Setup ---

def rounding_decorator(func):
    def wrapper(self, *args, **kwargs) -> Dict[int, float]:
        price_by_year = func(self, *args, **kwargs)
        if not isinstance(price_by_year, dict):
            raise TypeError("The decorated function must return Dict[int, float].")
        return {year: round(price, 2) for year, price in price_by_year.items()}
    return wrapper

class PriceDeck:
    def __init__(self,
         start_year: int = 2025, end_year: int = 2075,
         oil_price_initial: float = 70.0, gas_price_initial: float = 8.0, inflation_rate: float = 0.015,
         flat_after_year: int = None,
         oil_price_by_year: Dict[int, float] = None,
         gas_price_by_year: Dict[int, float] = None):

        self.start_year = start_year
        self.end_year = end_year
        self.inflation_rate = inflation_rate
        self.flat_after_year = flat_after_year
        self.years: List[int] = range(start_year, end_year+1, 1)
        self.oil_price_by_year: Dict[int, float] = {}
        self.gas_price_by_year: Dict[int, float] = {}

        self.oil_price_by_year = self._setting_initial_oil_price(oil_price_initial)
        self.gas_price_by_year = self._setting_initial_gas_price(gas_price_initial)

        if oil_price_by_year:
            for y, price in oil_price_by_year.items():
                self.oil_price_by_year[y] = price

        if gas_price_by_year:
            for y, price in gas_price_by_year.items():
                self.gas_price_by_year[y] = price

        if flat_after_year:
            self._setting_flat_price(flat_after_year)

    @rounding_decorator
    def _setting_initial_oil_price(self, oil_price_initial) -> Dict[int, float]:
        calculated_prices = {}
        for y in self.years:
            years_from_base = (y - self.start_year)
            calculated_prices[y] = oil_price_initial * ((1 + self.inflation_rate) ** years_from_base)
        return calculated_prices

    @rounding_decorator
    def _setting_initial_gas_price(self, gas_price_initial) -> Dict[int, float]:
        calculated_prices = {}
        for y in self.years:
            years_from_base = (y - self.start_year)
            calculated_prices[y] = gas_price_initial * ((1 + self.inflation_rate) ** years_from_base)
        return calculated_prices

    def _setting_flat_price(self, flat_after_year: int=None):
        for y in self.years:
            if y > flat_after_year:
                self.oil_price_by_year[y] = self.oil_price_by_year.get(flat_after_year, self.oil_price_by_year[max(self.oil_price_by_year.keys())])
                self.gas_price_by_year[y] = self.gas_price_by_year.get(flat_after_year, self.gas_price_by_year[max(self.gas_price_by_year.keys())])
        return self.oil_price_by_year, self.gas_price_by_year


def get_all_projects_summary() -> pd.DataFrame:
    """
    Scans all project JSON files in DATA_DIR and returns a summary DataFrame.
    Columns: Project Name, Gas Reserves (BCF), Oil Reserves (MMbbl), 
             Total Revenue, Total CAPEX, Net Cash Flow, NPV (Discounted), IRR
    """
    rows = []
    projects = list_projects()
    
    for proj_name in projects:
        file_path = DATA_DIR / f"{proj_name}.json"
        
        # Default Row Dictionary
        row = {
            "Project Name": proj_name,
            "Gas Reserves (BCF)": "not calculated yet",
            "Oil Reserves (MMbbl)": "not calculated yet",
            "Total Revenue ($MM)": "not calculated yet",
            "Total CAPEX ($MM)": "not calculated yet",
            "Net Cash Flow ($MM)": "not calculated yet",
            "NPV (Discounted) ($MM)": "not calculated yet",
            "IRR (%)": "not calculated yet"
        }
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Check for Cashflow Results
            cf_results = data.get("cashflow_results", {})
            
            if cf_results:
                # Pick the first one
                first_key = list(cf_results.keys())[0]
                res_item = cf_results[first_key]
                summary = res_item.get("result_summary", {})
                inputs = res_item.get("inputs", {})
                
                # Metrics
                if summary:
                    row["Total Revenue ($MM)"] = f"{summary.get('total_revenue', 0):,.0f}"
                    row["Total CAPEX ($MM)"] = f"{summary.get('total_capex', 0):,.0f}"
                    # final_cumulative is Net Cash Flow (Total) as per pages/4_Cash_Flow.py logic
                    row["Net Cash Flow ($MM)"] = f"{summary.get('final_cumulative', 0):,.0f}"
                    row["NPV (Discounted) ($MM)"] = f"{summary.get('npv', 0):,.0f}"
                    
                    irr_val = summary.get('irr', 'N/A')
                    if isinstance(irr_val, (int, float)):
                        row["IRR (%)"] = f"{irr_val*100:.1f}%"
                    else:
                        row["IRR (%)"] = "N/A"
                
                # Reserves (Need to look up Linked Development Case)
                dev_name = inputs.get("dev_name")
                if dev_name:
                    dev_cases = data.get("development_cases", {})
                    dev_data = dev_cases.get(dev_name, {})
                    
                    # Try to find profiles
                    # In serialized form, it might be in 'profiles' key (as saved in pages/1_Development.py)
                    # or 'dev_params_info' -> 'annual_gas_production' if purely serialized object
                    
                    profiles = dev_data.get("profiles", {})
                    gas_prof = profiles.get("gas", {})
                    oil_prof = profiles.get("oil", {})
                    
                    # Fallback to dev_params_info if profiles not found
                    if not gas_prof and "dev_params_info" in dev_data:
                        gas_prof = dev_data["dev_params_info"].get("annual_gas_production", {})
                        oil_prof = dev_data["dev_params_info"].get("annual_oil_production", {})

                    if gas_prof:
                        total_gas = sum(float(v) for v in gas_prof.values())
                        row["Gas Reserves (BCF)"] = f"{total_gas:,.1f}"
                    
                    if oil_prof:
                        total_oil = sum(float(v) for v in oil_prof.values())
                        row["Oil Reserves (MMbbl)"] = f"{total_oil:,.1f}"
                        
        except Exception as e:
            print(f"Error reading project {proj_name}: {e}")
            
        rows.append(row)
        
    if not rows:
        return pd.DataFrame(columns=[
            "Project Name", "Gas Reserves (BCF)", "Oil Reserves (MMbbl)", 
            "Total Revenue ($MM)", "Total CAPEX ($MM)", "Net Cash Flow ($MM)", 
            "NPV (Discounted) ($MM)", "IRR (%)"
        ])
        
    return pd.DataFrame(rows)


