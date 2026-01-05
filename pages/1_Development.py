from openpyxl.styles.alignment import horizontal_alignments
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
from plotting import plot_dev_cost_profile, plot_detailed_cost_breakdown

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

# --- Top-Level Case Management ---
st.divider()
st.subheader("📁 Case Management")
col_c1, col_c2, col_c3 = st.columns([2, 2, 1])

with col_c1:
    # Load Case
    existing_cases = list(st.session_state.development_cases.keys())
    cases_options = ["Select a case..."] + existing_cases
    selected_case = st.selectbox("Load Saved Case", options=cases_options, key="dev_load_case_selector")

    if selected_case != "Select a case...":
        if st.button("📂 Load Selected Case"):
            case_data = st.session_state.development_cases[selected_case]
            
            # 1. Restore Input Parameters
            if "input_params" in case_data:
                params = case_data["input_params"]
                # Production params
                st.session_state.qi_input = params.get("qi_input", 40.0)
                st.session_state.well_eur_input = params.get("well_eur_input", 60.0)
                st.session_state.prod_dur_input = params.get("prod_dur_input", 30)
                st.session_state.giip_input = params.get("giip_input", 4980.0)
                st.session_state.oiip_input = params.get("oiip_input", 329.0)
                st.session_state.drilling_rate_input = params.get("drilling_rate_input", 12)
                st.session_state.max_rate_input = params.get("max_rate_input", 250000)
                
                # Development params
                st.session_state.sunk_cost_input = params.get("sunk_cost_input", 0.0)
                st.session_state.exp_start_year_input = params.get("exp_start_year_input", 2024)
                st.session_state.dev_start_year_input = params.get("dev_start_year_input", 2026)
                st.session_state.drill_start_year_input = params.get("drill_start_year_input", 2033)
                st.session_state.dev_case_input = params.get("dev_case_input", "FPSO_case")
                
                # Cost params (Simple + Granular)
                # Helper to set safe fallback
                def safe_set(key, default):
                    st.session_state[key] = params.get(key, default)

                safe_set("feas_study_input", 3.0)
                safe_set("feas_study_t", -2)
                safe_set("feas_study_d", 2)
                
                safe_set("concept_study_input", 3.0)
                safe_set("concept_study_t", -2)
                safe_set("concept_study_d", 2)

                safe_set("feed_cost_input", 42.0)
                safe_set("feed_cost_t", 0)
                safe_set("feed_cost_d", 1)

                safe_set("pm_others_input", 10.1)
                safe_set("pm_others_t", 0)
                safe_set("pm_others_d", 5)

                safe_set("drilling_cost_input", 95.0)
                
                safe_set("subsea_cost_input", 41.1)
                
                safe_set("fpso_cost_input", 1570.0)
                safe_set("fpso_cost_t", 0)
                safe_set("fpso_cost_d", 3)

                safe_set("pipeline_cost_input", 244.0)
                safe_set("pipeline_cost_t", 0)
                safe_set("pipeline_cost_d", 2)

                safe_set("opex_per_bcf_input", 1.047)
                safe_set("opex_fixed_input", 347.424)
                safe_set("abex_per_well_input", 17.4)
                safe_set("abex_fpso_input", 114.7)

            # 2. Restore Results & Visualizations
            if "dev_obj" in case_data:
                st.session_state.current_dev_obj = case_data["dev_obj"]
                st.session_state.dev_results_ready = True
            
            if "profiles" in case_data:
                # Restore Prod Data for Chart
                profiles = case_data["profiles"]
                gas_p = profiles.get("gas", {})
                oil_p = profiles.get("oil", {})
                years = sorted([int(k) for k in gas_p.keys()])
                if years:
                    st.session_state.prod_data = pd.DataFrame({
                        'Year': years,
                        'Gas Production (BCF/y)': [gas_p.get(str(y), gas_p.get(y, 0)) for y in years],
                        'Oil Production (MMbbl/y)': [oil_p.get(str(y), oil_p.get(y, 0)) for y in years]
                    })
                    st.session_state.drilling_plan_results = profiles.get("drilling_plan", {})
                    st.session_state.wells_to_drill = sum(st.session_state.drilling_plan_results.values()) if st.session_state.drilling_plan_results else 0

            st.success(f"Case '{selected_case}' loaded!")
            st.rerun()

with col_c2:
    # Save Case
    new_case_name = st.text_input("New Case Name", value="Base Case", key="new_case_name_input")
    if st.button("💾 Save Current Case"):
        if not st.session_state.current_project:
            st.error("⚠️ No active project! Select a project in the Sidebar.")
        elif st.session_state.prod_data is None or not st.session_state.get('dev_results_ready'):
            st.error("⚠️ Please calculate results before saving.")
        else:
            # Collect Input Params
            input_params = {
                "qi_input": st.session_state.get("qi_input"),
                "well_eur_input": st.session_state.get("well_eur_input"),
                "prod_dur_input": st.session_state.get("prod_dur_input"),
                "giip_input": st.session_state.get("giip_input"),
                "oiip_input": st.session_state.get("oiip_input"),
                "drilling_rate_input": st.session_state.get("drilling_rate_input"),
                "max_rate_input": st.session_state.get("max_rate_input"),
                
                "sunk_cost_input": st.session_state.get("sunk_cost_input"),
                "exp_start_year_input": st.session_state.get("exp_start_year_input"),
                "dev_start_year_input": st.session_state.get("dev_start_year_input"),
                "drill_start_year_input": st.session_state.get("drill_start_year_input"),
                "dev_case_input": st.session_state.get("dev_case_input"),
                
                # Granular Timing Params
                "feas_study_input": st.session_state.get("feas_study_input"),
                "feas_study_t": st.session_state.get("feas_study_t"),
                "feas_study_d": st.session_state.get("feas_study_d"),
                
                "concept_study_input": st.session_state.get("concept_study_input"),
                "concept_study_t": st.session_state.get("concept_study_t"),
                "concept_study_d": st.session_state.get("concept_study_d"),

                "feed_cost_input": st.session_state.get("feed_cost_input"),
                "feed_cost_t": st.session_state.get("feed_cost_t"),
                "feed_cost_d": st.session_state.get("feed_cost_d"),

                "pm_others_input": st.session_state.get("pm_others_input"),
                "pm_others_t": st.session_state.get("pm_others_t"),
                "pm_others_d": st.session_state.get("pm_others_d"),

                "drilling_cost_input": st.session_state.get("drilling_cost_input"),
                "subsea_cost_input": st.session_state.get("subsea_cost_input"),
                
                "fpso_cost_input": st.session_state.get("fpso_cost_input"),
                "fpso_cost_t": st.session_state.get("fpso_cost_t"),
                "fpso_cost_d": st.session_state.get("fpso_cost_d"),

                "pipeline_cost_input": st.session_state.get("pipeline_cost_input"),
                "pipeline_cost_t": st.session_state.get("pipeline_cost_t"),
                "pipeline_cost_d": st.session_state.get("pipeline_cost_d"),

                "opex_per_bcf_input": st.session_state.get("opex_per_bcf_input"),
                "opex_fixed_input": st.session_state.get("opex_fixed_input"),
                "abex_per_well_input": st.session_state.get("abex_per_well_input"),
                "abex_fpso_input": st.session_state.get("abex_fpso_input"),
            }
            
            # Construct Case Data
            dev = st.session_state.current_dev_obj
            case_data = {
                "input_params": input_params, # New: Save inputs
                "prod_params": { # Keep for backward compatibility/reference
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
                "dev_obj": dev
            }
            
            st.session_state.development_cases[new_case_name] = case_data
            save_project(st.session_state.current_project)
            st.success(f"Case '{new_case_name}' saved!")

st.divider()

# --- Main Content ---

st.subheader("🛢️ Production Profile Generation")

with st.expander("Production Setup", expanded=True):
    # Nested tabs for TC and Field Profile
    t1, t2 = st.columns([2,5], gap="medium", vertical_alignment="top")
    
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

    st.markdown("---")

st.subheader("🛠️ Development Cost Generation")
with st.expander("Detailed Development Parameter Editor", expanded=True):
    
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        st.text("🔍 Exploration Costs", width=200)
        sunk_cost = st.number_input("Sunk Cost", value=0.0, key="sunk_cost_input")
        exploration_start_year = st.number_input("Exploration Start Year", value=2024, step=1, key="exp_start_year_input")
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

    st.markdown("---")

    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        st.text("🗓️ Development Year", width=200)
        dev_start_year = st.number_input("Development Start Year", value=2026, step=1, width=200, key="dev_start_year_input")
        drill_start_year = st.number_input("Production Drilling Start Year", value=2033, step=1, width=200, key="drill_start_year_input")
        dev_case = st.radio("Development Case", options=["FPSO_case", "tie-back_case"], key="dev_case_input")

    st.markdown("---")
    
    # ---------------- Study & PM Costs (Granular) ----------------
    # 1. Feasibility Study
    with st.container(horizontal=True, width='content', horizontal_alignment='left', vertical_alignment="bottom", gap="medium"):
        st.text("📋 Study & PM Costs", width=200)

        with st.container(horizontal=True, horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            feas_study = st.number_input("Feasibility Study", value=3.0, key="feas_study_input", width=100)
            feas_t = st.number_input("Timing", value=-2, key="feas_study_t", help="Relative to Dev Start Year", width=60)
            feas_d = st.number_input("Duration", min_value=1, value=2, key="feas_study_d", width=60)

        # 2. Concept Study
        with st.container(horizontal=True, horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            concept_study = st.number_input("Concept Study", value=3.0, key="concept_study_input", width=100)
            concept_t = st.number_input("Timing", value=-2, key="concept_study_t", help="Relative to Dev Start Year", width=60)
            concept_d = st.number_input("Duration", min_value=1, value=2, key="concept_study_d", width=60)
        
        # 3. FEED Cost
        with st.container(horizontal=True, horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            feed_val = 42.0 if dev_case == "FPSO_case" else 3.0
            feed_cost = st.number_input("FEED Cost", value=feed_val, key="feed_cost_input", width=100)
            feed_t = st.number_input("Timing", value=0, key="feed_cost_t", help="Relative to Dev Start Year", width=60)
            feed_d = st.number_input("Duration", min_value=1, value=1, key="feed_cost_d", width=60)

        # 4. PM & Others
        with st.container(horizontal=True, horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            pm_others = st.number_input("PM & Others", value=10.1, key="pm_others_input", width=100)
            pm_t = st.number_input("Timing", value=0, key="pm_others_t", help="Relative to Dev Start Year", width=60)
            pm_d = st.number_input("Duration", min_value=1, value=5, key="pm_others_d", width=60)

    st.markdown("---")

    # ---------------- Facility CAPEX (Granular) ----------------

    with st.container(horizontal=True, width='content', horizontal_alignment='left', vertical_alignment="bottom", gap="medium"):
        st.text("🏗️ Facility CAPEX", width=200)

        with st.container(horizontal=True, width='content', horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            st.text("Drilling Cost", width=100)
            drilling_cost = st.number_input("Drilling Cost per Well", value=95.0, key="drilling_cost_input", help="Driven by drilling schedule", width=160)
            subsea_cost = st.number_input("Subsea Cost per Well", value=41.1, key="subsea_cost_input", help="Driven by drilling schedule", width=160)

        with st.container(horizontal=True, width='content', horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            st.text("FPSO Cost", width=100)
            fpso_val = 1570.0 if dev_case == "FPSO_case" else 0.0
            fpso_cost = st.number_input("FPSO / Facility Cost", value=fpso_val, key="fpso_cost_input", width=160)
            fpso_t = st.number_input("Timing (FPSO)", value=0, key="fpso_cost_t", width=100)
            fpso_d = st.number_input("Duration (FPSO)", min_value=1, value=3, key="fpso_cost_d", width=100)
        
        with st.container(horizontal=True, width='content', horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            st.text("Pipeline Cost", width=100)
            pipeline_val = 244.0 if dev_case == "FPSO_case" else 0.0
            pipeline_cost = st.number_input("Export Pipeline Cost", value=pipeline_val, key="pipeline_cost_input", width=160)
            pipe_t = st.number_input("Timing (Pipe)", value=0, key="pipeline_cost_t", width=100)
            pipe_d = st.number_input("Duration (Pipe)", min_value=1, value=2, key="pipeline_cost_d", width=100)

    st.markdown("---")

    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        st.text("💸 OPEX & ABEX", width=200)
        opex_per_bcf = st.number_input("OPEX per BCF", value=1.047, format="%.3f", key="opex_per_bcf_input")
        opex_fixed = st.number_input("OPEX Fixed (k$/y)", value=347.424, key="opex_fixed_input")
        abex_per_well = st.number_input("ABEX per Well", value=17.4, key="abex_per_well_input")
        abex_fpso = st.number_input("ABEX FPSO", value=114.7 if dev_case == "FPSO_case" else 90.0, key="abex_fpso_input")

# Pack parameters using the granular format
dev_param = {dev_case: {
    'drilling_cost': drilling_cost, 
    'Subsea_cost': subsea_cost,
    
    # Study items (dict with cost, timing, duration)
    'feasability_study': {'cost': feas_study, 'timing': feas_t, 'duration': feas_d},
    'concept_study_cost': {'cost': concept_study, 'timing': concept_t, 'duration': concept_d},
    'FEED_cost': {'cost': feed_cost, 'timing': feed_t, 'duration': feed_d},
    'PM_others_cost': {'cost': pm_others, 'timing': pm_t, 'duration': pm_d},
    'EIA_cost': {'cost': 0.0, 'timing': 0, 'duration': 1}, # Hidden/Default for now

    # Facility items
    'FPSO_cost': {'cost': fpso_cost, 'timing': fpso_t, 'duration': fpso_d},
    'export_pipeline_cost': {'cost': pipeline_cost, 'timing': pipe_t, 'duration': pipe_d},
    'terminal_cost': {'cost': 0.0, 'timing': 0, 'duration': 1}, # Hidden/Default

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
    st.plotly_chart(plot_dev_cost_profile(dev))
    st.plotly_chart(plot_detailed_cost_breakdown(dev))
    
    st.markdown("### 📊 Detailed CAPEX Breakdown (MM$)")
    breakdown = dev.get_cost_breakdown()
    capex_data = breakdown.get('capex_breakdown', {})
    
    # Create DataFrame: Rows=Items, Columns=Years
    if capex_data:
        df_breakdown = pd.DataFrame(capex_data).T
        # Convert column names (years) to integer and sort
        df_breakdown.columns = df_breakdown.columns.astype(int)
        df_breakdown = df_breakdown.reindex(sorted(df_breakdown.columns), axis=1)
        
        # Display with formatting
        st.dataframe(df_breakdown.style.format("{:,.2f}"), width='stretch')
