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

def save_project(project_name: str):
    if not project_name:
        return
    file_path = DATA_DIR / f"{project_name}.json"
    
    # We only save the major case dictionaries. 
    # Note: development_cases contains objects that need special handling.
    # For now, let's filter out non-serializable objects or convert them.
    
    data_to_save = {
        "production_cases": st.session_state.production_cases,
        "development_cases": serialize_dev_cases(st.session_state.development_cases),
        "price_cases": st.session_state.price_cases,
        "cashflow_results": serialize_cashflow_results(st.session_state.cashflow_results)
    }
    
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
    return [f.stem for f in DATA_DIR.glob("*.json")]

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

