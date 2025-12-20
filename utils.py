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
        "price_cases": st.session_state.price_cases
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
    """Converts DevelopmentCost objects to serializable dicts."""
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
                "annual_oil_production": dev_obj.annual_oil_production
            }
            del new_case["dev_obj"]
        serialized[name] = new_case
    return serialized

def deserialize_dev_cases(serialized_cases):
    """Reconstructs DevelopmentCost objects from dicts."""
    from development import DevelopmentCost
    deserialized = {}
    for name, case in serialized_cases.items():
        new_case = case.copy()
        if "dev_params_info" in new_case:
            info = new_case["dev_params_info"]
            dev_obj = DevelopmentCost(
                dev_start_year=info["dev_start_year"],
                dev_param=info["dev_param"],
                development_case=info["development_case"]
            )
            # Re-apply schedule and production (keys in JSON are strings, convert to int)
            dev_obj.set_drilling_schedule(
                drill_start_year=info["drill_start_year"],
                yearly_drilling_schedule={int(k): v for k, v in info["yearly_drilling_schedule"].items()},
                already_shifted=True,
                output=False
            )
            dev_obj.set_annual_production(
                annual_gas_production={int(k): v for k, v in info["annual_gas_production"].items()},
                annual_oil_production={int(k): v for k, v in info["annual_oil_production"].items()},
                already_shifted=True,
                output=False
            )
            dev_obj.calculate_total_costs(output=False)
            new_case["dev_obj"] = dev_obj
            del new_case["dev_params_info"]
        deserialized[name] = new_case
    return deserialized

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

