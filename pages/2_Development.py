import streamlit as st
import pandas as pd
from development import DevelopmentCost
import plotly.graph_objects as go
from utils import ensure_state_init, save_project, render_project_sidebar

st.set_page_config(page_title="Development Costs", layout="wide")

st.title("🏗️ Development Cost Generation")

# --- Initialize Session State & Sidebar ---
ensure_state_init()
render_project_sidebar()

# --- Scenario Selection (Dependency) ---   
if not st.session_state.production_cases:
    st.warning("⚠️ No production cases found. Please go to the **Production** page and save a case first.")
    st.stop()

selected_prod_name = st.selectbox("Select Production Case for Drilling Plan", list(st.session_state.production_cases.keys()))
selected_prod = st.session_state.production_cases[selected_prod_name]

# --- Sidebar Inputs (Timing) ---
with st.sidebar:
    st.header("Project Timing")
    dev_start_year = st.number_input("Development Start Year", value=2026, step=1)
    drill_start_year = st.number_input("Production Drilling Start Year", value=2033, step=1)
    
    st.divider()
    dev_case = st.radio("Development Case", options=["FPSO_case", "tie-back_case"])

# --- Detailed Parameter Editor ---
with st.expander("🛠️ Detailed Development Parameter Editor", expanded=True):
    st.info("Edit technical and economic parameters for the development scenario below.")
    
    col_ed0, col_ed1, col_ed2, col_ed3 = st.columns(4)

    with col_ed0:
        st.subheader("🔍 Exploration Costs")
        sunk_cost = st.number_input("Sunk Cost", value=0.0)
        exploration_start_year = st.number_input("Exploration Start Year", value=2024, step=1)
        
        exploration_init_df = pd.DataFrame({
            "year": range(exploration_start_year, exploration_start_year + 10),
            "exploration costs (MM$)": [0.0] * 10
        })
        exploration_df = st.data_editor(exploration_init_df, hide_index=True, use_container_width=True)


    with col_ed1:
        st.subheader("📋 Study & PM Costs")
        feas_study = st.number_input("Feasibility Study", value=3.0)
        concept_study = st.number_input("Concept Study", value=3.0)
        feed_cost = st.number_input("FEED Cost", value=42.0 if dev_case == "FPSO_case" else 3.0)
        eia_cost = st.number_input("EIA Cost", value=2.0 if dev_case == "FPSO_case" else 1.0)
        pm_others = st.number_input("PM & Others", value=10.1)

    with col_ed2:
        st.subheader("🏗️ Facility CAPEX")
        drilling_cost = st.number_input("Drilling Cost per Well", value=95.0)
        subsea_cost = st.number_input("Subsea Cost", value=41.1)
        fpso_cost = st.number_input("FPSO / Facility Cost", value=1570.0 if dev_case == "FPSO_case" else 0.0)
        pipeline_cost = st.number_input("Export Pipeline Cost", value=244.0 if dev_case == "FPSO_case" else 0.0)
        terminal_cost = st.number_input("Terminal Cost", value=51.0 if dev_case == "FPSO_case" else 0.0)

    with col_ed3:
        st.subheader("💸 OPEX & ABEX")
        opex_per_bcf = st.number_input("OPEX per BCF", value=1.047, format="%.3f")
        opex_fixed = st.number_input("OPEX Fixed (k$/y)", value=347.424)
        abex_per_well = st.number_input("ABEX per Well", value=17.4)
        abex_fpso = st.number_input("ABEX FPSO", value=114.7 if dev_case == "FPSO_case" else 90.0)
        abex_subsea = st.number_input("ABEX Subsea", value=14.0)
        abex_onshore = st.number_input("ABEX Onshore Pipeline", value=0.5)
        abex_offshore = st.number_input("ABEX Offshore Pipeline", value=11.0)

# Build dev_param dict from inputs
dev_param = {
    dev_case: {
        'drilling_cost': drilling_cost,
        'feasability_study': feas_study,
        'concept_study_cost': concept_study,
        'FEED_cost': feed_cost,
        'EIA_cost': eia_cost,
        'Subsea_cost': subsea_cost,
        'FPSO_cost': fpso_cost,
        'export_pipeline_cost': pipeline_cost,
        'terminal_cost': terminal_cost,
        'PM_others_cost': pm_others,
        'OPEX_per_bcf': opex_per_bcf,
        'OPEX_fixed': opex_fixed,
        'ABEX_per_well': abex_per_well,
        'ABEX_FPSO': abex_fpso,
        'ABEX_subsea': abex_subsea,
        'ABEX_onshore_pipeline': abex_onshore,
        'ABEX_offshore_pipeline': abex_offshore
    }
}

# --- Calculation Trigger ---
st.divider()
apply_button = st.button("🔄 Apply Parameters & Calculate", width='content', type="primary")

# Use session state to keep results visible after calculation
if 'dev_results_ready' not in st.session_state:
    st.session_state.dev_results_ready = False

if apply_button:
    st.session_state.dev_results_ready = True

if st.session_state.dev_results_ready:
    # --- Calculation ---
    dev = DevelopmentCost(dev_start_year=dev_start_year, dev_param=dev_param, development_case=dev_case)

    # Set exploration stage costs
    exploration_costs_dict = exploration_df.set_index('year')['exploration costs (MM$)'].to_dict()
    dev.set_exploration_stage(
        exploration_start_year = exploration_start_year,
        exploration_costs=exploration_costs_dict
        sunk_cost=sunk_cost,
    )

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
        fig_annual.update_layout(barmode='stack', title="Annual Expenditures (MM$)", xaxis_title="Year", yaxis_title="MM$")
        st.plotly_chart(fig_annual, width='stretch')

        # --- Added Gas Production Chart ---
        st.subheader("Annual Gas Production")
        prod_years = sorted(dev.annual_gas_production.keys())
        if prod_years:
            prod_data = pd.DataFrame({
                'Year': prod_years,
                'Gas Production': [dev.annual_gas_production.get(y, 0.0) for y in prod_years]
            })
            fig_prod = go.Figure()
            fig_prod.add_trace(go.Bar(x=prod_data['Year'], y=prod_data['Gas Production'], name='Gas Production', marker_color='green'))
            fig_prod.update_layout(title="Annual Gas Production (BCF/y)", xaxis_title="Year", yaxis_title="BCF/y")
            st.plotly_chart(fig_prod, width='stretch')
        else:
            st.info("No gas production data available for the selected drilling plan.")

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
else:
    st.info("💡 Edit parameters in the editor above and click 'Apply Parameters & Calculate' to view results.")
