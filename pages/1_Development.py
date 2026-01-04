from streamlit import session_state
import streamlit as st
import pandas as pd
import numpy as np
import math
from production import YearlyProductionProfile
from development import DevelopmentCost
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import ensure_state_init, save_project, render_project_sidebar
from plotting import plot_dev_cost_profile

st.set_page_config(page_title="Development & Production", layout="wide")

st.title("Development & Production Setup")

st.space(size="small")

st.markdown("""
자원량을 기준으로 시추공수를 계산하고,  
그에 따른 개발비를 계산합니다.

""")
st.space(size="small")

# --- Initialize Session State & Sidebar ---
ensure_state_init()
render_project_sidebar()

st.subheader("🛢️ Production Profile Generation")

with st.expander("Production Setup", expanded=True):
    # Nested tabs for TC and Field Profile
    t1, t2 = st.columns([2,5], gap="medium", vertical_alignment="top")# (["📈 Type Curve", "🏭 Field Production Profile"])
    
    with t1:
        with st.container(horizontal=True, gap="small"):
            qi_mmcfd = st.number_input("Initial Rate (MMcf/d)", min_value=1.0, value=40.0, key="qi_input")
            well_eur_bcf = st.number_input("Well EUR (BCF)", min_value=1.0, value=60.0, key="well_eur_input")
        
            tc_duration = st.session_state.get("prod_dur_input", 30)
        
            if st.button("🚀 Generate Type Curve", width='stretch'):
                profile = YearlyProductionProfile(production_duration=int(tc_duration))
                profile.generate_type_curve_from_exponential(
                    qi_mmcfd=qi_mmcfd,
                    EUR_target_mmcf=well_eur_bcf * 1000,
                    T_years=int(tc_duration)
                )
                st.session_state.profile = profile
                st.session_state.tc_data = pd.DataFrame({
                    'Year': range(1, len(profile.yearly_type_rate) + 1),
                    'Annual Rate (MMcf/y)': profile.yearly_type_rate,
                    'Cumulative Production (MMcf)': profile.yearly_type_cum
                })
                st.success("Type Curve Generated!")

        if st.session_state.tc_data is not None:
            tc_df = st.session_state.tc_data
            st.plotly_chart(px.line(tc_df, x='Year', y='Annual Rate (MMcf/y)', title="Annual Rate vs. Years"), width='stretch')

    with t2:
        with st.container(horizontal=True, gap="small"):
            giip_bcf = st.number_input("Gas Reserves (BCF)", min_value=1.0, value=4980.0, step=100.0, key="giip_input")
            oiip_mmbbl = st.number_input("Oil Reserves (MMbbl)", min_value=0.0, value=329.0, step=10.0, key="oiip_input")
            prod_duration = st.number_input("Prod. Period (Years)", min_value=1, value=30, key="prod_dur_input")
            drilling_rate = st.number_input("Drilling Rate (Wells/Year)", min_value=1, value=12, key="drilling_rate_input")
            max_prod_rate = st.number_input("Max Prod. Rate (MMcf/y)", min_value=0, value=250_000, key="max_rate_input")

        if st.button("🚀 Generate Field Production Profile", width='stretch'):
            if st.session_state.profile is None:
                st.error("⚠️ Please generate a Type Curve first.")
            else:                
                profile = st.session_state.profile
                profile.production_duration = int(prod_duration)
                wells_to_drill = math.ceil(giip_bcf / well_eur_bcf)
                st.session_state.wells_to_drill = wells_to_drill

                drilling_plan = profile.make_drilling_plan(total_wells_number=wells_to_drill, drilling_rate=drilling_rate)
                gas_profile = profile.make_production_profile_yearly(peak_production_annual=max_prod_rate if max_prod_rate > 0 else None)
                cgr = (oiip_mmbbl / giip_bcf) * 1000
                oil_profile = {year: gas * cgr / 1000 for year, gas in gas_profile.items()}
                
                st.session_state.prod_data = pd.DataFrame({
                    'Year': list(gas_profile.keys()),
                    'Gas Production (BCF/y)': list(gas_profile.values()),
                    'Oil Production (MMbbl/y)': list(oil_profile.values())
                })
                st.session_state.drilling_plan_results = drilling_plan
                st.session_state.current_cgr = cgr
                # st.success("Field Production Profile Generated!")

        if st.session_state.prod_data is not None:
            st.plotly_chart(px.bar(st.session_state.prod_data, x='Year', y='Gas Production (BCF/y)', title="Annual Field Gas Production"), width='stretch')
            st.info(f"🔢 **Estimated Total Wells: {st.session_state.wells_to_drill}** (based on {giip_bcf:,.1f} BCF Reserves / {well_eur_bcf:,.1f} BCF Well EUR)")

st.space(size="small")

st.subheader("🛠️ Development Cost Generation")
with st.expander("Detailed Development Parameter Editor", expanded=True):
    
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        st.text("🔍 Exploration Costs", width=200)
        sunk_cost = st.number_input("Sunk Cost", value=0.0)
        exploration_start_year = st.number_input("Exploration Start Year", value=2024, step=1)
        years_range = list(range(int(exploration_start_year), int(exploration_start_year) + 10))
        if st.button("🔄 Exploration Costs Manual Input"):
            exploration_data = {
                "Year": years_range,
                "Exploration Costs (MM$)": [0.0] * 10
            }
            exploration_df = pd.DataFrame(exploration_data).set_index("Year")
            exploration_df.index = exploration_df.index.astype(int)
            st.session_state.exploration_data = exploration_df.T
    
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        if "exploration_data" in st.session_state:
            st.session_state.exploration_data = st.data_editor(st.session_state.exploration_data, width='stretch')

    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        st.text("🗓️ Project Timing", width=200)
        dev_start_year = st.number_input("Development Start Year", value=2026, step=1, width=200)
        drill_start_year = st.number_input("Production Drilling Start Year", value=2033, step=1, width=200)
        dev_case = st.radio("Development Case", options=["FPSO_case", "tie-back_case"])

    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        st.text("📋 Study & PM Costs", width=200)
        feas_study = st.number_input("Feasibility Study", value=3.0)
        concept_study = st.number_input("Concept Study", value=3.0)
        feed_cost = st.number_input("FEED Cost", value=42.0 if dev_case == "FPSO_case" else 3.0)
        pm_others = st.number_input("PM & Others", value=10.1)

    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        st.text("🏗️ Facility CAPEX", width=200)
        drilling_cost = st.number_input("Drilling Cost per Well", value=95.0)
        subsea_cost = st.number_input("Subsea Cost", value=41.1)
        fpso_cost = st.number_input("FPSO / Facility Cost", value=1570.0 if dev_case == "FPSO_case" else 0.0)
        pipeline_cost = st.number_input("Export Pipeline Cost", value=244.0 if dev_case == "FPSO_case" else 0.0)

    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        st.text("💸 OPEX & ABEX", width=200)
        opex_per_bcf = st.number_input("OPEX per BCF", value=1.047, format="%.3f")
        opex_fixed = st.number_input("OPEX Fixed (k$/y)", value=347.424)
        abex_per_well = st.number_input("ABEX per Well", value=17.4)
        abex_fpso = st.number_input("ABEX FPSO", value=114.7 if dev_case == "FPSO_case" else 90.0)

dev_param = {dev_case: {
    'drilling_cost': drilling_cost, 'feasability_study': feas_study, 'concept_study_cost': concept_study,
    'FEED_cost': feed_cost, 'Subsea_cost': subsea_cost, 'FPSO_cost': fpso_cost,
    'export_pipeline_cost': pipeline_cost, 'PM_others_cost': pm_others,
    'OPEX_per_bcf': opex_per_bcf, 'OPEX_fixed': opex_fixed,
    'ABEX_per_well': abex_per_well, 'ABEX_FPSO': abex_fpso,
    'ABEX_subsea': 14.0, 'ABEX_onshore_pipeline': 0.5, 'ABEX_offshore_pipeline': 11.0
}}

if st.button("🔄 Apply Parameters & Calculate", width='content', type="primary"):
    if st.session_state.prod_data is None:
        st.error("⚠️ Please generate a Production Profile in the first tab first.")
    else:
        st.session_state.exploration_df = st.session_state.exploration_data.T
        st.session_state.exploration_df.index = st.session_state.exploration_df.index.astype(int)
        exploration_costs_dict = st.session_state.exploration_df['Exploration Costs (MM$)'].to_dict()
        dev = DevelopmentCost(dev_start_year=dev_start_year, dev_param=dev_param, development_case=dev_case)
        dev.set_drilling_schedule(drill_start_year=drill_start_year, yearly_drilling_schedule=st.session_state.drilling_plan_results)
        dev.set_annual_production(
            annual_gas_production=dict(zip(st.session_state.prod_data['Year'], st.session_state.prod_data['Gas Production (BCF/y)'])),
            annual_oil_production=dict(zip(st.session_state.prod_data['Year'], st.session_state.prod_data['Oil Production (MMbbl/y)']))
        )
        dev.set_exploration_stage(
            exploration_start_year = exploration_start_year,
            exploration_costs=exploration_costs_dict,
            sunk_cost=sunk_cost,
        )
        dev.calculate_total_costs()
        st.session_state.current_dev_obj = dev
        st.session_state.dev_results_ready = True

if st.session_state.get('dev_results_ready'):
    dev = st.session_state.current_dev_obj
    # years = sorted(set(dev.cost_years + list(dev.exploration_costs.keys()) + list(dev.annual_gas_production.keys()) + list(dev.annual_oil_production.keys()) + list(dev.annual_capex.keys()) + list(dev.annual_abex.keys())))
    # cost_df = pd.DataFrame({
    #     'Year': years,  
    #     'Exp_Cost': [dev.exploration_costs.get(y, 0.0) for y in years],
    #     'CAPEX': [dev.annual_capex.get(y, 0.0) for y in years],
    #     'OPEX': [dev.annual_opex.get(y, 0.0) for y in years],
    #     'ABEX': [dev.annual_abex.get(y, 0.0) for y in years]
    # })
    # st.plotly_chart(px.bar(cost_df, 
    #     x='Year', 
    #     y=['Exp_Cost','CAPEX', 'OPEX', 'ABEX'], 
    #     title="Annual Expenditures (MM$)"), theme=None, width='stretch')
    st.plotly_chart(plot_dev_cost_profile(dev))

# --- Integrated Case Management ---
st.divider()
st.subheader("📁 Case Management")
case_name = st.text_input("Enter Combined Case Name", value="Base Case")

if st.button("💾 Save Combined Development Case"):
    if not st.session_state.current_project:
        st.error("⚠️ No active project! Select a project in the Sidebar.")
    elif st.session_state.prod_data is None or not st.session_state.get('dev_results_ready'):
        st.error("⚠️ Please ensure both Production and Development results are calculated.")
    else:
        case_data = {
            "prod_params": {
                "giip_bcf": st.session_state.giip_input,
                "oiip_mmbbl": st.session_state.oiip_input,
                "well_eur_bcf": st.session_state.well_eur_input,
                "drilling_rate": st.session_state.drilling_rate_input,
                "max_prod_rate": st.session_state.max_rate_input
            },
            "cost_summary": {
                "total_capex": dev.total_capex,
                "total_opex": dev.total_opex,
                "total_abex": dev.total_abex
            },
            "profiles": {
                "gas": dict(zip(st.session_state.prod_data['Year'], st.session_state.prod_data['Gas Production (BCF/y)'])),
                "oil": dict(zip(st.session_state.prod_data['Year'], st.session_state.prod_data['Oil Production (MMbbl/y)'])),
                "drilling_plan": st.session_state.drilling_plan_results
            },
            "dev_obj": st.session_state.current_dev_obj
        }
        st.session_state.development_cases[case_name] = case_data
        save_project(st.session_state.current_project)
        st.success(f"Combined case '{case_name}' saved!")

if st.session_state.development_cases:
    st.write("Saved Combined Case(s):", list(st.session_state.development_cases.keys()))
