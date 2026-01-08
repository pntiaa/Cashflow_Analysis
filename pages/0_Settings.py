import streamlit as st
import json
import pandas as pd
from utils import (
    ensure_state_init, save_defaults, render_project_sidebar, 
    list_projects, delete_project, delete_case, save_project
)

st.set_page_config(page_title="Settings", layout="wide")

st.title("⚙️ Global Settings & Defaults")

# --- Initialize Session State & Sidebar ---
ensure_state_init()
render_project_sidebar()

if "defaults" not in st.session_state or not st.session_state.defaults:
    st.error("⚠️ Defaults not loaded. Please check data/defaults.json.")
    st.stop()

defaults = st.session_state.defaults

# --- Project Management Section ---
st.subheader("📁 Project & Case Management")
with st.expander("🛠️ Manage Projects and Cases", expanded=False):
    m_tab1, m_tab2, m_tab3 = st.tabs(["➕ Add Project", "🗑️ Remove Project", "❌ Delete Case"])
    
    with m_tab1:
        st.markdown("#### Create New Project")
        new_p_name = st.text_input("Project Name", key="settings_new_p_name")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            new_p_gas = st.number_input("Gas Reserves (BCF)", value=0.0, step=10.0)
        with col_r2:
            new_p_oil = st.number_input("Oil Reserves (MMbbl)", value=0.0, step=1.0)
        
        if st.button("🚀 Add Project", type="primary"):
            if new_p_name:
                if new_p_name in list_projects():
                    st.error(f"Project '{new_p_name}' already exists.")
                else:
                    # Create Project
                    st.session_state.current_project = new_p_name
                    st.session_state.production_cases = {}
                    st.session_state.development_cases = {}
                    st.session_state.price_cases = {}
                    st.session_state.cashflow_results = {}
                    
                    # If reserves provided, create a skeleton case
                    if new_p_gas > 0 or new_p_oil > 0:
                        st.session_state.production_cases["Base Case"] = {
                            "input_params": {
                                "giip_input": new_p_gas,
                                "oiip_mmbbl": new_p_oil,
                                "well_eur_bcf": defaults["production"]["well_eur_bcf"],
                                "qi_input": defaults["production"]["qi_mmcfd"],
                                "prod_dur_input": defaults["production"]["prod_duration"],
                                "drilling_rate_input": defaults["production"]["drilling_rate"],
                                "max_rate_input": defaults["production"]["max_prod_rate"]
                            },
                            "profiles": {}, # Empty profiles, user needs to run calculation
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") if 'pd' in globals() else ""
                        }
                    
                    save_project(new_p_name)
                    st.success(f"Project '{new_p_name}' created!")
                    st.rerun()
            else:
                st.error("Please enter a project name.")

    with m_tab2:
        st.markdown("#### Delete Entire Project")
        p_to_delete = st.selectbox("Select Project to Remove", list_projects())
        if st.button("⚠️ Delete Project", type="secondary"):
            if delete_project(p_to_delete):
                st.success(f"Project '{p_to_delete}' deleted.")
                st.rerun()
            else:
                st.error("Failed to delete project.")

    with m_tab3:
        st.markdown("#### Delete Specific Case")
        target_p = st.selectbox("Select Target Project", list_projects(), key="target_p_select")
        
        # Load project data to list cases
        if target_p:
            file_path = st.session_state.defaults.get("data_dir", "./data") + f"/{target_p}.json"
            # We can't easily read file from here without opening it or using a util
            # Let's use a temporary load (not changing session state)
            import os
            from pathlib import Path
            import json
            p_file = Path("./data") / f"{target_p}.json"
            if p_file.exists():
                with open(p_file, "r") as f:
                    p_data = json.load(f)
                
                c_type = st.selectbox("Case Category", ["Production", "Development", "Price", "Cash Flow Result"])
                type_map = {
                    "Production": "production_cases",
                    "Development": "development_cases",
                    "Price": "price_cases",
                    "Cash Flow Result": "cashflow_results"
                }
                category_key = type_map[c_type]
                cases_in_cat = list(p_data.get(category_key, {}).keys())
                
                if cases_in_cat:
                    c_to_delete = st.selectbox("Select Case to Delete", cases_in_cat)
                    if st.button("❌ Delete Case"):
                        if delete_case(target_p, c_type, c_to_delete):
                            st.success(f"Case '{c_to_delete}' deleted from '{target_p}'.")
                            st.rerun()
                        else:
                            st.error("Failed to delete case.")
                else:
                    st.info(f"No cases found in {c_type} category.")

st.divider()

st.markdown("Edit the default parameters used across the application. These values will be used when creating new cases.")

# Tabs for different categories
tab1, tab2, tab3, tab4 = st.tabs(["🛢️ Production", "🛠️ Development", "📈 Price Deck", "⚖️ Economics"])

with tab1:
    st.subheader("Field & Production Defaults")
    prod = defaults["production"]
    col1, col2 = st.columns(2)
    with col1:
        prod["qi_mmcfd"] = st.number_input("Initial Rate (MMcf/d)", value=float(prod["qi_mmcfd"]))
        prod["well_eur_bcf"] = st.number_input("Well EUR (BCF)", value=float(prod["well_eur_bcf"]))
        prod["prod_duration"] = st.number_input("Prod. Period (Years)", value=int(prod["prod_duration"]))
    with col2:
        prod["giip_bcf"] = st.number_input("Gas Reserves (BCF)", value=float(prod["giip_bcf"]))
        prod["oiip_mmbbl"] = st.number_input("Oil Reserves (MMbbl)", value=float(prod["oiip_mmbbl"]))
        prod["drilling_rate"] = st.number_input("Drilling Rate (Wells/Year)", value=int(prod["drilling_rate"]))
        prod["max_prod_rate"] = st.number_input("Max Prod. Rate (MMcf/y)", value=int(prod["max_prod_rate"]))

with tab2:
    st.subheader("Development Cost & Timing Defaults")
    dev = defaults["development"]
    
    col1, col2 = st.columns(2)
    with col1:
        dev["sunk_cost"] = st.number_input("Sunk Cost (MM$)", value=float(dev["sunk_cost"]))
        dev["exp_start_year"] = st.number_input("Exploration Start Year", value=int(dev["exp_start_year"]))
    with col2:
        dev["dev_start_year"] = st.number_input("Development Start Year", value=int(dev["dev_start_year"]))
        dev["drill_start_year"] = st.number_input("Drilling Start Year", value=int(dev["drill_start_year"]))

    st.divider()
    st.markdown("#### Study & PM Costs")
    study = dev["study_costs"]
    c1, c2, c3 = st.columns(3)
    with c1:
        study["feasibility"]["cost"] = st.number_input("Feasibility Cost", value=float(study["feasibility"]["cost"]))
        study["feasibility"]["timing"] = st.number_input("Feasibility Timing", value=int(study["feasibility"]["timing"]))
    with c2:
        study["concept"]["cost"] = st.number_input("Concept Cost", value=float(study["concept"]["cost"]))
        study["concept"]["timing"] = st.number_input("Concept Timing", value=int(study["concept"]["timing"]))
    with c3:
        study["pm_others_per_well"] = st.number_input("PM & Others (per well)", value=float(study["pm_others_per_well"]))

    st.divider()
    st.markdown("#### Facility CAPEX")
    fac = dev["facility_capex"]
    c1, c2 = st.columns(2)
    with c1:
        fac["drilling_per_well"] = st.number_input("Drilling Cost per Well", value=float(fac["drilling_per_well"]))
        fac["subsea_per_well"] = st.number_input("Subsea Cost per Well", value=float(fac["subsea_per_well"]))
    with c2:
        fac["fpso_fpso"]["cost"] = st.number_input("FPSO Cost", value=float(fac["fpso_fpso"]["cost"]))
        fac["fpso_fpso"]["duration"] = st.number_input("FPSO Duration", value=int(fac["fpso_fpso"]["duration"]))

    st.divider()
    st.markdown("#### OPEX & ABEX")
    opab = dev["opex_abex"]
    c1, c2 = st.columns(2)
    with c1:
        opab["opex_per_bcf"] = st.number_input("OPEX per BCF", value=float(opab["opex_per_bcf"]), format="%.3f")
        opab["opex_fixed"] = st.number_input("OPEX Fixed (MM$/y)", value=float(opab["opex_fixed"]))
    with c2:
        opab["abex_per_well"] = st.number_input("ABEX per Well", value=float(opab["abex_per_well"]))
        opab["abex_fpso_fpso"] = st.number_input("ABEX FPSO", value=float(opab["abex_fpso_fpso"]))

with tab3:
    st.subheader("Price Deck Defaults")
    price = defaults["price_deck"]
    col1, col2 = st.columns(2)
    with col1:
        price["oil_init"] = st.number_input("Initial Oil Price ($/bbl)", value=float(price["oil_init"]))
        price["gas_init"] = st.number_input("Initial Gas Price ($/mcf)", value=float(price["gas_init"]))
        price["inflation_pct"] = st.number_input("Price Inflation (%)", value=float(price["inflation_pct"]))
    with col2:
        price["start_year"] = st.number_input("Price Start Year", value=int(price["start_year"]))
        price["end_year"] = st.number_input("Price End Year", value=int(price["end_year"]))
        price["cap_2x"] = st.checkbox("Cap at 2x Initial Price", value=bool(price["cap_2x"]))

with tab4:
    st.subheader("Global Economic Defaults")
    eco = defaults["economics"]
    col1, col2 = st.columns(2)
    with col1:
        eco["base_year"] = st.number_input("Base Year (for PV)", value=int(eco["base_year"]))
        eco["discount_rate"] = st.number_input("Discount Rate (fraction)", value=float(eco["discount_rate"]), format="%.2f")
        eco["exchange_rate"] = st.number_input("Exchange Rate (KRW/USD)", value=float(eco["exchange_rate"]))
    with col2:
        eco["cost_inflation"] = st.number_input("Cost Inflation Rate", value=float(eco["cost_inflation"]), format="%.4f")
        eco["depreciation_method"] = st.selectbox("Depreciation Method", ["Unit of Production", "Straight Line", "Declining Balance"], index=["Unit of Production", "Straight Line", "Declining Balance"].index(eco["depreciation_method"]))
        eco["useful_life"] = st.number_input("Useful Life (Years)", value=int(eco["useful_life"]))

st.divider()
if st.button("💾 Save Settings", type="primary"):
    save_defaults(defaults)
    st.session_state.defaults = defaults
    st.success("✅ Settings saved successfully!")
    st.rerun()
