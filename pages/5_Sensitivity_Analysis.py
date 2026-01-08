import streamlit as st
import pandas as pd
import numpy as np
import copy
from cashflow import CashFlowKOR
from utils import ensure_state_init, render_project_sidebar
from plotting import plot_tornado_chart

st.set_page_config(page_title="Sensitivity Analysis", layout="wide")

st.title("🌪️ Sensitivity Analysis")

# --- Initialize Session State & Sidebar ---
ensure_state_init()
render_project_sidebar()

# --- Dependency Check ---
if not st.session_state.get('cashflow_results'):
    st.warning("⚠️ No cash flow results found. Please run a Cash Flow Analysis first.")
    st.stop()

# --- Base Scenario Selection ---
st.subheader("🏁 Select Base Scenario")
res_keys = list(st.session_state.cashflow_results.keys())
selected_res_name = st.selectbox("Select Scenario to Analyze", res_keys, index=len(res_keys)-1)
res = st.session_state.cashflow_results[selected_res_name]

# Helper to run the model
def run_peturbed_model(base_res, modifications=None):
    inputs = base_res['inputs']
    global_params = inputs['global_params']
    
    # Extract base technical data
    # We use dev_obj from development_cases to get original profiles
    dev_name = inputs['dev_name']
    dev_data = st.session_state.development_cases[dev_name]
    dev_obj = copy.deepcopy(dev_data['dev_obj'])
    
    price_name = inputs['price_name']
    price_data = st.session_state.price_cases[price_name]
    oil_price = copy.deepcopy(price_data['oil'])
    gas_price = copy.deepcopy(price_data['gas'])
    
    # Apply modifications
    if modifications:
        for param, delta in modifications.items():
            if param == "Gas Volume":
                dev_obj.annual_gas_production = {k: v * (1 + delta) for k, v in dev_obj.annual_gas_production.items()}
            elif param == "Oil Volume":
                dev_obj.annual_oil_production = {k: v * (1 + delta) for k, v in dev_obj.annual_oil_production.items()}
            elif param == "Gas Price":
                gas_price = {k: v * (1 + delta) for k, v in gas_price.items()}
            elif param == "Oil Price":
                oil_price = {k: v * (1 + delta) for k, v in oil_price.items()}
            elif param == "CAPEX":
                dev_obj.annual_capex = {k: v * (1 + delta) for k, v in dev_obj.annual_capex.items()}
                # Also scale capex breakdown if it exists for depreciation
                if hasattr(dev_obj, 'capex_breakdown'):
                    for k in dev_obj.capex_breakdown:
                        if isinstance(dev_obj.capex_breakdown[k], dict):
                            dev_obj.capex_breakdown[k] = {y: val * (1 + delta) for y, val in dev_obj.capex_breakdown[k].items()}
            elif param == "OPEX":
                dev_obj.annual_opex = {k: v * (1 + delta) for k, v in dev_obj.annual_opex.items()}

    # Setup Cash Flow Instance
    cf = CashFlowKOR(
        base_year=global_params['base_year'],
        oil_price_by_year=oil_price,
        gas_price_by_year=gas_price,
        cost_inflation_rate=global_params['cost_inflation'],
        discount_rate=global_params['discount_rate'],
        exchange_rate=global_params['exchange_rate']
    )
    
    cf.set_development_costs(dev_obj, output=False)
    
    # Production start year handling (same as pages/4_Cash_Flow.py)
    def ensure_actual_years(d, start_year):
        if not d: return {}
        min_key = min(d.keys())
        if min_key < 1000: return {int(k) + start_year: v for k, v in d.items()}
        return {int(k): v for k, v in d.items()}

    oil_shifted = ensure_actual_years(dev_obj.annual_oil_production, dev_obj.drill_start_year)
    gas_shifted = ensure_actual_years(dev_obj.annual_gas_production, dev_obj.drill_start_year)
    cf.set_production_profile_from_dicts(oil_dict=oil_shifted, gas_dict=gas_shifted)
    
    # Run Calculation Sequence
    cf.calculate_annual_revenue(output=False)    
    cf.determine_cop_year(output=False)
    cf.apply_cop_adjustments(output=False)
    cf.calculate_royalty(output=False)
    cf.calculate_depreciation(method='unit_of_production', useful_life=global_params['useful_life'], output=False)
    cf.calculate_taxes(output=False)
    cf.calculate_net_cash_flow(output=False)
    cf.calculate_npv(output=False)
    
    return cf.npv

# --- Sensitivity Parameters UI ---
st.divider()
st.subheader("⚙️ Sensitivity Parameters")

params_list = ["Gas Volume", "Oil Volume", "Gas Price", "Oil Price", "CAPEX", "OPEX"]
sensitivity_settings = []

col_header = st.columns([0.5, 2, 2, 2])
col_header[0].write("**Run?**")
col_header[1].write("**Parameter**")
col_header[2].write("**Low Range (%)**")
col_header[3].write("**High Range (%)**")

for i, p in enumerate(params_list):
    cols = st.columns([0.5, 2, 2, 2])
    with cols[0]:
        enabled = st.checkbox("", value=True, key=f"enable_{i}")
    with cols[1]:
        st.write(p)
    with cols[2]:
        low_val = st.number_input(f"Low_{i}", value=-10.0, step=1.0, key=f"low_{i}", label_visibility="collapsed")
    with cols[3]:
        high_val = st.number_input(f"High_{i}", value=10.0, step=1.0, key=f"high_{i}", label_visibility="collapsed")
    
    if enabled:
        sensitivity_settings.append({
            "Parameter": p,
            "Low %": low_val / 100.0,
            "High %": high_val / 100.0
        })

run_btn = st.button("🚀 Run Sensitivity Analysis", type="primary")

if run_btn:
    if not sensitivity_settings:
        st.error("Please select at least one parameter.")
    else:
        with st.spinner("Calculating sensitivity cases..."):
            base_npv = run_peturbed_model(res)
            
            results = []
            for setting in sensitivity_settings:
                p = setting["Parameter"]
                
                # Low case
                low_npv = run_peturbed_model(res, {p: setting["Low %"]})
                # High case
                high_npv = run_peturbed_model(res, {p: setting["High %"]})
                
                results.append({
                    "Parameter": p,
                    "Low_NPV": low_npv,
                    "High_NPV": high_npv
                })
            
            df_results = pd.DataFrame(results)
            
            st.divider()
            st.subheader("📊 Sensitivity Results")
            
            # Display Tornado Chart
            fig = plot_tornado_chart(df_results, base_npv)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display Results Table
            with st.expander("📄 Detailed NPV Impacts"):
                df_display = df_results.copy()
                df_display['Low Impact'] = df_display['Low_NPV'] - base_npv
                df_display['High Impact'] = df_display['High_NPV'] - base_npv
                st.dataframe(df_display.style.format({
                    'Low_NPV': "{:,.1f}",
                    'High_NPV': "{:,.1f}",
                    'Low Impact': "{:,.1f}",
                    'High Impact': "{:,.1f}"
                }))
else:
    st.info("💡 Adjust the parameters and click the button to run the analysis.")
