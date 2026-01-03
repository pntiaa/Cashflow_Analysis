import streamlit as st
import pandas as pd
import numpy as np
from utils import PriceDeck, ensure_state_init, save_project, render_project_sidebar
import plotly.express as px

st.set_page_config(page_title="Price Deck Setup", layout="wide")

st.title("📈 Oil & Gas Price Setup")

# --- Initialize Session State & Sidebar ---
ensure_state_init()
render_project_sidebar()

# Persistent state for current price dictionaries
if "price_deck_oil" not in st.session_state:
    st.session_state.price_deck_oil = {y: 70.0 for y in range(2025, 2076)}
if "price_deck_gas" not in st.session_state:
    st.session_state.price_deck_gas = {y: 8.0 for y in range(2025, 2076)}

# --- 1. Manual Input Section ---
st.header("1. Manual Price Input")
with st.expander("🛠️ Manual Price Configuration", expanded=False):
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        m_start_year = st.number_input("Input Start Year", value=2025, key="m_start")
        m_end_year = st.number_input("Input End Year", value=2035, key="m_end")
    with col_m2:
        outbound_policy = st.selectbox("Outbound Policy (post End Year)", ["Same as end year price", "Apply Inflation"], key="m_policy")
        m_inflation = st.number_input("Outbound Inflation Rate (%)", value=2.0, step=0.1, key="m_inf") / 100.0

    # Initialize / Prepare Dataframe for editing
    # User requested: Row 1 years, Row 2 oil, Row 3 gas
    # In Streamlit, it's easier to edit columns, so we'll transpose for the editor if needed, 
    # but the user explicitly asked for rows. We'll use a transposed DF for editing.
    
    years_range = list(range(int(m_start_year), int(m_end_year) + 1))
    
    if st.button("🔄 Initialize/Reset Manual Table"):
        init_data = {
            "Year": years_range,
            "Oil Price ($/bbl)": [st.session_state.price_deck_oil.get(y, 70.0) for y in years_range],
            "Gas Price ($/mcf)": [st.session_state.price_deck_gas.get(y, 8.0) for y in years_range]
        }
        st.session_state.df_manual_edit = pd.DataFrame(init_data).set_index("Year").T
    
    if "df_manual_edit" in st.session_state:
        edited_df = st.data_editor(st.session_state.df_manual_edit, use_container_width=True)
        
        if st.button("🚀 Apply Manual Forecast", type="primary"):
            # Update the range from the table
            new_prices = edited_df.T.to_dict('index')
            
            # Apply to session state and handle outbound
            max_proj_year = 2075 # Target horizon
            temp_oil = st.session_state.price_deck_oil.copy()
            temp_gas = st.session_state.price_deck_gas.copy()
            
            # Update with edited range
            last_year = int(m_end_year)
            for y, prices in new_prices.items():
                temp_oil[y] = prices["Oil Price ($/bbl)"]
                temp_gas[y] = prices["Gas Price ($/mcf)"]
            
            # Handle Outbound
            last_oil = temp_oil[last_year]
            last_gas = temp_gas[last_year]
            
            for y in range(last_year + 1, max_proj_year + 1):
                if outbound_policy == "Same as end year price":
                    temp_oil[y] = last_oil
                    temp_gas[y] = last_gas
                else:
                    years_diff = y - last_year
                    temp_oil[y] = last_oil * ((1 + m_inflation) ** years_diff)
                    temp_gas[y] = last_gas * ((1 + m_inflation) ** years_diff)
            
            st.session_state.price_deck_oil = temp_oil
            st.session_state.price_deck_gas = temp_gas
            st.success("Manual forecast applied!")

# --- 2. Price Setting Section (Auto Generation) ---
st.header("2. Automated Price Generation")
with st.expander("🪄 Auto Generation Parameters", expanded=True):
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        a_start_year = st.number_input("Start Year", value=2025, key="a_start")
        a_end_year = st.number_input("End Year", value=2075, key="a_end")
    with col_a2:
        a_oil_init = st.number_input("Initial Oil Price ($/bbl)", value=70.0, key="a_oil_init")
        a_gas_init = st.number_input("Initial Gas Price ($/mcf)", value=8.0, key="a_gas_init")
    with col_a3:
        a_inflation = st.number_input("Inflation Rate (%)", value=1.5, step=0.1, key="a_inf_pct") / 100.0
        cap_2x = st.checkbox("Stop increasing at 2x initial price", value=True)

    if st.button("⚡ Generate & Apply Prices", type="primary"):
        temp_oil = {}
        temp_gas = {}
        
        oil_limit = a_oil_init * 2 if cap_2x else float('inf')
        gas_limit = a_gas_init * 2 if cap_2x else float('inf')
        
        for y in range(int(a_start_year), int(a_end_year) + 1):
            years_from_base = y - a_start_year
            
            calc_oil = a_oil_init * ((1 + a_inflation) ** years_from_base)
            calc_gas = a_gas_init * ((1 + a_inflation) ** years_from_base)
            
            temp_oil[y] = min(calc_oil, oil_limit)
            temp_gas[y] = min(calc_gas, gas_limit)
            
        st.session_state.price_deck_oil = temp_oil
        st.session_state.price_deck_gas = temp_gas
        st.success(f"Generated prices from {a_start_year} to {a_end_year}.")

# --- Visualization & Case Management ---
st.divider()

col_v1, col_v2 = st.columns([2, 1])

with col_v1:
    st.subheader("Current Price Forecast")
    # Ensure keys are integers to avoid sorting errors between str and int
    st.session_state.price_deck_oil = {int(k): v for k, v in st.session_state.price_deck_oil.items()}
    st.session_state.price_deck_gas = {int(k): v for k, v in st.session_state.price_deck_gas.items()}
    
    all_years = sorted(st.session_state.price_deck_oil.keys())
    price_data = pd.DataFrame({
        'Year': all_years,
        'Oil Price ($/bbl)': [st.session_state.price_deck_oil.get(y, 0.0) for y in all_years],
        'Gas Price ($/mcf)': [st.session_state.price_deck_gas.get(y, 0.0) for y in all_years]
    })
    
    fig = px.line(price_data, x='Year', y=['Oil Price ($/bbl)', 'Gas Price ($/mcf)'],
                 title="Commodity Price Deck", markers=False)
    st.plotly_chart(fig, use_container_width=True)

with col_v2:
    st.subheader("Data Summary")
    st.dataframe(price_data, height=400)

st.divider()
st.subheader("📁 Case Management")
case_name = st.text_input("Enter Price Scenario Name", value="Base Price")

if st.button("💾 Save Price Scenario"):
    if not st.session_state.current_project:
        st.error("⚠️ No active project! Please create or select a project in the Sidebar first.")
    else:
        price_case = {
            "oil": st.session_state.price_deck_oil,
            "gas": st.session_state.price_deck_gas,
            "params": {
                "source": "Manual/Auto Enhanced",
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            }
        }
        st.session_state.price_cases[case_name] = price_case
        save_project(st.session_state.current_project)
        st.success(f"Price scenario '{case_name}' saved to project '{st.session_state.current_project}'!")

if st.session_state.price_cases:
    st.write("Saved Scenarios:", list(st.session_state.price_cases.keys()))
