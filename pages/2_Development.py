import streamlit as st
import pandas as pd
from development import DevelopmentCost
import plotly.graph_objects as go
from utils import ensure_state_init, save_project

st.set_page_config(page_title="Development Costs", layout="wide")

st.title("🏗️ Development Cost Generation")

# --- Initialize Session State ---
ensure_state_init()

# --- Scenario Selection (Dependency) ---   
if not st.session_state.production_cases:
    st.warning("⚠️ No production cases found. Please go to the **Production** page and save a case first.")
    st.stop()

selected_prod_name = st.selectbox("Select Production Case for Drilling Plan", list(st.session_state.production_cases.keys()))
selected_prod = st.session_state.production_cases[selected_prod_name]

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Project Timing")
    dev_start_year = st.number_input("Development Start Year", value=2026)
    drill_start_year = st.number_input("Production Drilling Start Year", value=2033)
    
    st.divider()
    dev_case = st.radio("Development Case", options=["FPSO_case", "tie-back_case"])
    
    st.header("CAPEX Parameters (MM$)")
    drilling_cost = st.number_input("Drilling Cost per Well", value=95.0)
    subsea_cost = st.number_input("Subsea Cost", value=41.1)
    fpso_cost = st.number_input("FPSO / Facility Cost", value=1570.0 if dev_case == "FPSO_case" else 0.0)
    pipeline_cost = st.number_input("Export Pipeline Cost", value=244.0 if dev_case == "FPSO_case" else 0.0)

# Build dev_param dict (matching notebook/script structure)
dev_param = {
    dev_case: {
        'drilling_cost': drilling_cost,
        'feasability_study': 3.0,
        'concept_study_cost': 3.0,
        'FEED_cost': 42.0 if dev_case == "FPSO_case" else 3.0,
        'EIA_cost': 2.0 if dev_case == "FPSO_case" else 1.0,
        'Subsea_cost': subsea_cost,
        'FPSO_cost': fpso_cost,
        'export_pipeline_cost': pipeline_cost,
        'terminal_cost': 51.0 if dev_case == "FPSO_case" else 0.0,
        'PM_others_cost': 10.1,
        'OPEX_per_bcf': 2.093,
        'OPEX_fixed': 10422.6,
        'ABEX_per_well': 17.4,
        'ABEX_FPSO': 114.7 if dev_case == "FPSO_case" else 90.0,
        'ABEX_subsea': 14.0,
        'ABEX_onshore_pipeline': 0.5,
        'ABEX_offshore_pipeline': 11.00
    }
}

# --- Calculation ---
dev = DevelopmentCost(dev_start_year=dev_start_year, dev_param=dev_param, development_case=dev_case)

# Link to selected production case's drilling plan and volumes
dev.set_drilling_schedule(
    drill_start_year=drill_start_year,
    yearly_drilling_schedule=selected_prod['profiles']['drilling_plan']
)
dev.set_annual_production(
    annual_gas_production=selected_prod['profiles']['gas'],
    annual_oil_production=selected_prod['profiles']['oil']
)

results = dev.calculate_total_costs()

# --- UI Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Cost Profile (Annual)")
    years = sorted(dev.total_annual_costs.keys())
    cost_data = pd.DataFrame({
        'Year': years,
        'CAPEX': [dev.annual_capex.get(y, 0.0) for y in years],
        'OPEX': [dev.annual_opex.get(y, 0.0) for y in years],
        'ABEX': [dev.annual_abex.get(y, 0.0) for y in years]
    })
    
    fig_annual = go.Figure()
    fig_annual.add_trace(go.Bar(x=cost_data['Year'], y=cost_data['CAPEX'], name='CAPEX'))
    fig_annual.add_trace(go.Bar(x=cost_data['Year'], y=cost_data['OPEX'], name='OPEX'))
    fig_annual.add_trace(go.Bar(x=cost_data['Year'], y=cost_data['ABEX'], name='ABEX'))
    fig_annual.update_layout(barmode='stack', title="Annual Expenditures")
    st.plotly_chart(fig_annual, use_container_width=True)

with col2:
    st.subheader("Summary Metrics")
    total_capex = sum(dev.annual_capex.values())
    total_opex = sum(dev.annual_opex.values())
    total_abex = sum(dev.annual_abex.values())
    
    st.metric("Total CAPEX", f"{total_capex:,.1f} MM$")
    st.metric("Total OPEX", f"{total_opex:,.1f} MM$")
    st.metric("Total ABEX", f"{total_abex:,.1f} MM$")
    st.metric("Grand Total", f"{(total_capex + total_opex + total_abex):,.1f} MM$")

# --- Case Management ---
st.divider()
st.subheader("📁 Case Management")
case_name = st.text_input("Enter Cost Case Name", value="Base Cost")

if st.button("💾 Save Cost Case"):
    if not st.session_state.current_project:
        st.error("⚠️ No active project! Please create or select a project in the Sidebar first.")
    else:
        cost_case = {
            "dev_obj": dev,
            "selected_prod": selected_prod_name,
            "summary": {
                "total_capex": total_capex,
                "total_opex": total_opex,
                "total_abex": total_abex
            }
        }
        st.session_state.development_cases[case_name] = cost_case
        save_project(st.session_state.current_project)
        st.success(f"Cost case '{case_name}' saved to project '{st.session_state.current_project}'!")

# --- Export ---
csv = cost_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Cost Profile CSV",
    data=csv,
    file_name=f'cost_profile_{case_name}.csv',
    mime='text/csv',
)
