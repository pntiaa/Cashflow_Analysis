from openpyxl.styles.alignment import vertical_aligments
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
from plotting import plot_dev_cost_profile, plot_detailed_cost_breakdown, plot_df_profile

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
                # Helper to set safe fallback
                def safe_set(key, default):
                    st.session_state[key] = params.get(key, default)

                d_eco = st.session_state.defaults["economics"]
                d_dev = st.session_state.defaults["development"]
                d_prod = st.session_state.defaults["production"]
                d_study = d_dev["study_costs"]
                d_fac = d_dev["facility_capex"]
                d_opab = d_dev["opex_abex"]

                safe_set("qi_input", d_prod["qi_mmcfd"])
                safe_set("well_eur_input", d_prod["well_eur_bcf"])
                safe_set("prod_dur_input", d_prod["prod_duration"])
                safe_set("giip_input", d_prod["giip_bcf"])
                safe_set("oiip_mmbbl", d_prod["oiip_mmbbl"])
                safe_set("drilling_rate_input", d_prod["drilling_rate"])
                safe_set("max_rate_input", d_prod["max_prod_rate"])
                
                # Development params
                safe_set("sunk_cost_input", d_dev["sunk_cost"])
                safe_set("exp_start_year_input", d_dev["exp_start_year"])
                safe_set("dev_start_year_input", d_dev["dev_start_year"])
                safe_set("drill_start_year_input", d_dev["drill_start_year"])
                safe_set("dev_case_input", d_dev["dev_case"])
                
                # Cost params (Simple + Granular)
                safe_set("feas_study_input", d_study["feasibility"]["cost"])
                safe_set("feas_study_t", d_study["feasibility"]["timing"])
                safe_set("feas_study_d", d_study["feasibility"]["duration"])
                
                safe_set("concept_study_input", d_study["concept"]["cost"])
                safe_set("concept_study_t", d_study["concept"]["timing"])
                safe_set("concept_study_d", d_study["concept"]["duration"])

                safe_set("feed_cost_input", d_study["feed_fpso"]["cost"])
                safe_set("feed_cost_t", d_study["feed_fpso"]["timing"])
                safe_set("feed_cost_d", d_study["feed_fpso"]["duration"])

                safe_set("pm_others_input", d_study["pm_others_per_well"])

                safe_set("drilling_cost_input", d_fac["drilling_per_well"])
                
                safe_set("subsea_cost_input", d_fac["subsea_per_well"])
                
                safe_set("fpso_cost_input", d_fac["fpso_fpso"]["cost"])
                safe_set("fpso_cost_t", d_fac["fpso_fpso"]["timing"])
                safe_set("fpso_cost_d", d_fac["fpso_fpso"]["duration"])

                safe_set("pipeline_cost_input", d_fac["pipeline_fpso"]["cost"])
                safe_set("pipeline_cost_t", d_fac["pipeline_fpso"]["timing"])
                safe_set("pipeline_cost_d", d_fac["pipeline_fpso"]["duration"])

                safe_set("opex_per_bcf_input", d_opab["opex_per_bcf"])
                safe_set("opex_fixed_input", d_opab["opex_fixed"])
                safe_set("abex_per_well_input", d_opab["abex_per_well"])
                safe_set("abex_fpso_input", d_opab["abex_fpso_fpso"])

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
                    "total_abex": dev.total_abex,
                    "capex_breakdown": {
                        "exploration": dev.exploration_costs,
                        "drilling": dev.drilling_costs,
                        "subsea": dev.subsea_costs,
                        "feasibility_study": dev.feasability_study_cost,
                        "concept_study": dev.concept_study_cost,
                        "FEED": dev.FEED_cost,
                        "EIA": dev.EIA_cost,
                        "FPSO": dev.FPSO_cost,
                        "export_pipeline": dev.export_pipeline_cost,
                        "terminal": dev.terminal_cost,
                        "PM_others": dev.PM_others_cost
                    }
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
    t1, t2 = st.columns([1,2], gap="medium", vertical_alignment="top")
    
    with t1:
        with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
            qi_mmcfd = st.number_input("Initial Rate (MMcf/d)", min_value=1.0, value=st.session_state.qi_input, key="qi_input")
            well_eur_bcf = st.number_input("Well EUR (BCF)", min_value=1.0, value=st.session_state.well_eur_input, key="well_eur_input")
        
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
            if st.session_state.tc_data is None:
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
            tc_df = st.session_state.tc_data
            st.plotly_chart(px.line(tc_df, x='Year', y='Annual Rate (MMcf/y)', title="Annual Rate vs. Years"), width='stretch')

    with t2:
        with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
            giip_bcf = st.number_input("Gas Reserves (BCF)", min_value=1.0, value=st.session_state.giip_input, step=100.0, key="giip_input")
            oiip_mmbbl = st.number_input("Oil Reserves (MMbbl)", min_value=0.0, value=st.session_state.oiip_input, step=10.0, key="oiip_input")
            prod_duration = st.number_input("Prod. Period (Years)", min_value=1, value=st.session_state.prod_dur_input, key="prod_dur_input")
            drilling_rate = st.number_input("Drilling Rate (Wells/Year)", min_value=1, value=st.session_state.drilling_rate_input, key="drilling_rate_input")
            max_prod_rate = st.number_input("Max Prod. Rate (MMcf/y)", min_value=0, value=st.session_state.max_rate_input, key="max_rate_input")

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
                    drilling_plan = {year: drilling_plan.get(year, 0) for year in range(1, int(prod_duration) + 1)}
                    st.session_state.prod_data = pd.DataFrame({
                        'Year': list(gas_profile.keys()),
                        'Gas Production (BCF/y)': list(gas_profile.values()),
                        'Oil Production (MMbbl/y)': list(oil_profile.values()),
                        'Drilled Wells': list(drilling_plan.values())
                    })
                    st.session_state.drilling_plan_results = drilling_plan
                    st.session_state.current_cgr = cgr
                    # st.success("Field Production Profile Generated!")

        if st.session_state.prod_data is not None:
            # st.plotly_chart(px.bar(st.session_state.prod_data, x='Year', y='Gas Production (BCF/y)', title="Annual Field Gas Production"), width='stretch')
            years = st.session_state.prod_data['Year']
            gas_production = st.session_state.prod_data['Gas Production (BCF/y)']
            oil_production = st.session_state.prod_data['Oil Production (MMbbl/y)']
            drilled_wells = list(st.session_state.drilling_plan_results.values())
            st.plotly_chart(plot_df_profile(years, gas_production, oil_production, drilled_wells), width='stretch')
            st.info(f"🔢 **Estimated Total Wells: {st.session_state.wells_to_drill}** (based on {giip_bcf:,.1f} BCF Reserves / {well_eur_bcf:,.1f} BCF Well EUR)")

st.subheader("🛠️ Development Cost Generation")
with st.expander("Detailed Development Parameter Editor", expanded=True):
    
    with st.container(horizontal=True, vertical_alignment="top", gap="small"):
        st.text("🔍 Exploration Costs", width=200)
        sunk_cost = st.number_input("Sunk Cost", value=st.session_state.sunk_cost_input, key="sunk_cost_input", width=200)
        exploration_start_year = st.number_input("Exploration Start Year", value=st.session_state.exp_start_year_input, step=1, key="exp_start_year_input", width=200)
        years_range = list(range(int(exploration_start_year), int(exploration_start_year) + 10))
        if "exploration_data" not in st.session_state:
            st.session_state.exploration_data = pd.DataFrame(
                {"Year": years_range, "Exploration Costs (MM$)": [0.0] * 10}
            ).set_index("Year").T

        if st.button("Apply to update years"):
            st.session_state.exploration_data.columns = years_range
    
    with st.container(horizontal=True, vertical_alignment="top", gap="small"):
        if "exploration_data" in st.session_state:
            st.session_state.exploration_data = st.data_editor(st.session_state.exploration_data, width='stretch')

    st.markdown("---")

    with st.container(horizontal=True, vertical_alignment="top", gap="small"):
        st.text("🗓️ Development Year", width=200)
        dev_start_year = st.number_input("Development Start Year", value=2026, step=1, width=200, key="dev_start_year_input")
        drill_start_year = st.number_input("Production Drilling Start Year", value=2033, step=1, width=200, key="drill_start_year_input")
        dev_case = st.radio("Development Case", options=["FPSO_case", "tie-back_case"], key="dev_case_input")

    st.markdown("---")
    
    # ---------------- Study & PM Costs (Granular) ----------------
    # 1. Feasibility Study
    with st.container(horizontal=True, width='content', horizontal_alignment='left', vertical_alignment="top", gap="medium"):
        st.text("📋 Study & PM Costs", width=200)

        with st.container(horizontal=True, horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            feas_study = st.number_input("Feasibility Study", value=3.0, key="feas_study_input", width=100)
            feas_t = st.number_input("Timing", value=0, key="feas_study_t", help="Relative to Dev Start Year", width=60)
            feas_d = st.number_input("Duration", min_value=1, value=6, key="feas_study_d", width=60)

        # 2. Concept Study
        with st.container(horizontal=True, horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            concept_study = st.number_input("Concept Study", value=st.session_state.concept_study_input, key="concept_study_input", width=100)
            concept_t = st.number_input("Timing", value=st.session_state.concept_study_t, key="concept_study_t", help="Relative to Dev Start Year", width=60)
            concept_d = st.number_input("Duration", min_value=1, value=st.session_state.concept_study_d, key="concept_study_d", width=60)
        
        # 3. FEED Cost
        with st.container(horizontal=True, horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            d_study = st.session_state.defaults["development"]["study_costs"]
            f_val = d_study["feed_fpso"]["cost"] if dev_case == "FPSO_case" else d_study["feed_tieback"]["cost"]
            # We only use f_val if there is no session state value yet or it's not from a loaded case
            # But Streamlit key handles it. Just set the default value based on case.
            feed_cost = st.number_input("FEED Cost", value=f_val, key="feed_cost_input", width=100)
            feed_t = st.number_input("Timing", value=st.session_state.feed_cost_t, key="feed_cost_t", help="Relative to Dev Start Year", width=60)
            feed_d = st.number_input("Duration", min_value=1, value=st.session_state.feed_cost_d, key="feed_cost_d", width=60)

        # 4. PM & Others
        with st.container(horizontal=True, horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            pm_others = st.number_input("PM & Others (per Well)", value=10.1, key="pm_others_input", width=160)

    st.markdown("---")

    # ---------------- Facility CAPEX (Granular) ----------------

    with st.container(horizontal=True, width='content', horizontal_alignment='left', vertical_alignment="top", gap="medium"):
        st.text("🏗️ Facility CAPEX", width=200)

        with st.container(horizontal=True, width='content', horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            st.text("Drilling", width='content')
            drilling_cost = st.number_input("Drilling Cost per Well", value=95.0, key="drilling_cost_input", help="Driven by drilling schedule", width=160)
            subsea_cost = st.number_input("Subsea Cost per Well", value=41.1, key="subsea_cost_input", help="Driven by drilling schedule", width=160)

        with st.container(horizontal=True, width='content', horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            st.text("FPSO", width='content')
            fpso_val = 1570.0 if dev_case == "FPSO_case" else 0.0
            fpso_cost = st.number_input("FPSO / Facility Cost", value=fpso_val, key="fpso_cost_input", width=160)
            fpso_t = st.number_input("Timing (FPSO)", value=0, key="fpso_cost_t", width=100)
            fpso_d = st.number_input("Duration (FPSO)", min_value=1, value=7, key="fpso_cost_d", width=100)
        
        with st.container(horizontal=True, width='content', horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            st.text("Pipeline", width='content')
            pipeline_val = 244.0 if dev_case == "FPSO_case" else 0.0
            pipeline_cost = st.number_input("Export Pipeline Cost", value=pipeline_val, key="pipeline_cost_input", width=160)
            pipe_t = st.number_input("Timing (Pipe)", value=0, key="pipeline_cost_t", width=100)
            pipe_d = st.number_input("Duration (Pipe)", min_value=1, value=7, key="pipeline_cost_d", width=100)

        with st.container(horizontal=True, width='content', horizontal_alignment='left', vertical_alignment="bottom", gap="small"):
            st.text("Terminal", width='content')
            terminal_val = 51.0 if dev_case == "FPSO_case" else 0.0
            terminal_cost = st.number_input("Terminal Cost", value=terminal_val, key="terminal_cost_input", width=160)
            terminal_t = st.number_input("Timing (Terminal)", value=4, key="terminal_cost_t", width=100)
            terminal_d = st.number_input("Duration (Terminal)", min_value=1, value=3, key="terminal_cost_d", width=100)

    st.markdown("---")

    with st.container(horizontal=True, vertical_alignment="top", gap="small"):
        st.text("💸 OPEX & ABEX", width=200)
        opex_per_bcf = st.number_input("OPEX per BCF", value=st.session_state.opex_per_bcf_input, format="%.3f", key="opex_per_bcf_input", width=160)
        opex_fixed = st.number_input("OPEX Fixed (MM$/y)", value=st.session_state.opex_fixed_input, key="opex_fixed_input", width=160)
        abex_per_well = st.number_input("ABEX per Well", value=st.session_state.abex_per_well_input, key="abex_per_well_input", width=160)
        d_opab = st.session_state.defaults["development"]["opex_abex"]
        a_val = d_opab["abex_fpso_fpso"] if dev_case == "FPSO_case" else d_opab["abex_fpso_tieback"]
        abex_fpso = st.number_input("ABEX FPSO", value=a_val, key="abex_fpso_input", width=160)

# Pack parameters using the granular format
dev_param = {dev_case: {
    'drilling_cost': drilling_cost, 
    'Subsea_cost': subsea_cost,
    
    # Study items (dict with cost, timing, duration)
    'feasability_study': {'cost': feas_study, 'timing': feas_t, 'duration': feas_d},
    'concept_study_cost': {'cost': concept_study, 'timing': concept_t, 'duration': concept_d},
    'FEED_cost': {'cost': feed_cost, 'timing': feed_t, 'duration': feed_d},
    'PM_others_cost': pm_others,
    'EIA_cost': {'cost': 0.0, 'timing': 0, 'duration': 1}, # Hidden/Default for now

    # Facility items
    'FPSO_cost': {'cost': fpso_cost, 'timing': fpso_t, 'duration': fpso_d},
    'export_pipeline_cost': {'cost': pipeline_cost, 'timing': pipe_t, 'duration': pipe_d},
    'terminal_cost': {'cost': terminal_cost, 'timing': terminal_t, 'duration': terminal_d}, # Hidden/Default

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
    
    st.markdown("### 📊 Development Cost Profile (MM$)")
    st.plotly_chart(plot_dev_cost_profile(dev))

    st.markdown("### 📊 Detailed CAPEX Breakdown (MM$)")
    st.plotly_chart(plot_detailed_cost_breakdown(dev))
    
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
