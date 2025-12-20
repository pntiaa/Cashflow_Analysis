import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
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

# --- Plotting Functions ---

def plot_cash_flow_profile_plotly(cf, width=1200, height=800):
    years = cf.all_years
    rev = np.array([cf.annual_revenue.get(y, 0.0) for y in years])
    royalty = np.array([cf.annual_royalty.get(y, 0.0) for y in years])
    cap = np.array([cf.annual_capex.get(y, 0.0) for y in years])
    opx = np.array([cf.annual_opex.get(y, 0.0) for y in years])
    abx = np.array([cf.annual_abex.get(y, 0.0) for y in years])
    tax = np.array([cf.annual_total_tax.get(y, 0.0) for y in years])
    net = np.array([cf.annual_net_cash_flow.get(y, 0.0) for y in years])
    cum = np.array([cf.cumulative_cash_flow.get(y, 0.0) for y in years])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Annual Cash Flow Components",
            "Cumulative Cash Flow (After Tax)",
            "Production Profile",
            "Commodity Prices"
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy", "secondary_y": True}]
        ],
        column_widths=[0.6, 0.4],
        row_heights =[0.6, 0.4],
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    # 1. Annual
    fig.add_trace( go.Bar( x=years, y=rev, name='Revenue', marker_color='rgb(102,194,165)'), row=1, col=1)
    fig.add_trace( go.Bar( x=years, y=-royalty, name='Royalty', marker_color='rgb(252,141,98)' ), row=1, col=1)
    fig.add_trace( go.Bar( x=years, y=-(cap + opx + abx), name='Costs', marker_color='rgb(141,160,203)' ), row=1, col=1)
    fig.add_trace( go.Bar( x=years, y=-tax, name='Taxes', marker_color='rgb(231,138,195)' ), row=1, col=1)
    fig.add_trace(go.Scatter(x=years, y=net, name='Net Flow', mode='lines+markers', line=dict(color='black', width=2)), row=1, col=1)
    fig.update_layout(barmode='relative')

    # 2. Cumulative
    fig.add_trace(go.Scatter(
        x=years, y=cum, name='Cumulative', mode='lines',
        line=dict(color='purple', width=3), fill='tozeroy'
    ), row=1, col=2)
    fig.add_hline(y=0.0, line_dash="dash", line_color="red", row=1, col=2)

    # 3. Production
    oil_prod = [cf.oil_production_by_year.get(y, 0.0) for y in years]
    gas_prod = [cf.gas_production_by_year.get(y, 0.0) for y in years]
    fig.add_trace(go.Bar(x=years, y=gas_prod, name='Gas (BCF)', marker_color='lightblue'), row=2, col=1)
    
    # 4. Prices
    oilp = [cf.oil_price_by_year.get(y, 0.0) for y in years]
    gasp = [cf.gas_price_by_year.get(y, 0.0) for y in years]
    fig.add_trace(go.Scatter(x=years, y=oilp, name='Oil ($/bbl)', line=dict(color='green')), row=2, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=years, y=gasp, name='Gas ($/mcf)', line=dict(color='red')), row=2, col=2, secondary_y=True)

    fig.update_layout(height=height, width=width, title_text="Economic Results", template='plotly_white')
    return fig

def summary_plot(cf, width=1200, height=700):
    years = cf.all_years
    rev = np.array([cf.annual_revenue.get(y, 0.0) for y in years])
    cap = np.array([cf.annual_capex.get(y, 0.0) for y in years])
    opx = np.array([cf.annual_opex.get(y, 0.0) for y in years])
    abx = np.array([cf.annual_abex.get(y, 0.0) for y in years])
    tax = np.array([cf.annual_total_tax.get(y, 0.0) for y in years])
    net = np.array([cf.annual_net_cash_flow.get(y, 0.0) for y in years])
    cum = np.array([cf.cumulative_cash_flow.get(y, 0.0) for y in years])

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Key Metrics", "Price Trends", "Annual & Cumulative Cash Flow"),
        specs=[
            [{"type": "table"}, {"type": "xy","secondary_y": True}],
            [{"type": "xy", "colspan": 2, "secondary_y": True}, None],
        ],
        column_widths=[0.4, 0.6],
        row_heights =[0.4, 0.6],
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    # Table
    metrics = ['Net Cash Flow', 'NPV', 'IRR']
    values = [
        f"{cum[-1]:,.1f} MM$",
        f"{cf.npv:,.1f} MM$",
        f"{cf.irr*100:.1f}%" if cf.irr else "N/A"
    ]
    fig.add_trace(go.Table(
        header=dict(values=['Metric', 'Value'], fill_color='paleturquoise', align='left'),
        cells=dict(values=[metrics, values], fill_color='lavender', align='left')
    ), row=1, col=1)

    # Prices
    oilp = [cf.oil_price_by_year.get(y, 0.0) for y in years]
    gasp = [cf.gas_price_by_year.get(y, 0.0) for y in years]
    fig.add_trace(go.Scatter(x=years, y=oilp, name='Oil Price', line=dict(color='green')), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=years, y=gasp, name='Gas Price', line=dict(color='red')), row=1, col=2, secondary_y=True)

    # Main Chart
    color_palette = px.colors.qualitative.Set3
    fig.add_trace(go.Bar(x=years, y=rev, name='Revenue', marker_color=color_palette[0]), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=-cap, name='CAPEX', marker_color=color_palette[1]), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=-opx, name='OPEX', marker_color=color_palette[2]), row=2, col=1)
    fig.add_trace(go.Bar(x=years, y=-tax, name='Tax', marker_color=color_palette[3]), row=2, col=1)
    fig.add_trace(go.Scatter(x=years, y=net, name='NCF', line=dict(color='black', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=years, y=cum, name='Cumulative', line=dict(color='purple', dash='dot')), row=2, col=1, secondary_y=True)

    fig.update_layout(height=height, width=width, template='plotly_white', barmode='relative')
    return fig
