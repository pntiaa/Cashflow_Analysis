import streamlit as st
import pandas as pd
from utils import PriceDeck
import plotly.express as px

st.set_page_config(page_title="Price Deck Setup", layout="wide")

st.title("📈 Oil & Gas Price Setup")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Price Deck Parameters")
    start_year = st.number_input("Forecast Start Year", value=2025)
    end_year = st.number_input("Forecast End Year", value=2075)
    
    st.divider()
    oil_price_init = st.number_input("Initial Oil Price ($/bbl)", value=70.0)
    gas_price_init = st.number_input("Initial Gas Price ($/mcf)", value=8.0)
    
    st.divider()
    inflation_rate = st.number_input("Price Inflation Rate (fraction)", value=0.0125, format="%.4f")
    flat_after = st.number_input("Flat Price Year (Optional)", value=0)

# --- Calculation ---
flat_year = flat_after if flat_after > start_year else None
price_obj = PriceDeck(
    start_year=start_year,
    end_year=end_year,
    oil_price_initial=oil_price_init,
    gas_price_initial=gas_price_init,
    inflation_rate=inflation_rate,
    flat_after_year=flat_year
)

# --- UI Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Price Forecast")
    price_data = pd.DataFrame({
        'Year': list(price_obj.oil_price_by_year.keys()),
        'Oil Price ($/bbl)': list(price_obj.oil_price_by_year.values()),
        'Gas Price ($/mcf)': list(price_obj.gas_price_by_year.values())
    })
    
    fig = px.line(price_data, x='Year', y=['Oil Price ($/bbl)', 'Gas Price ($/mcf)'],
                 title="Commodity Price Deck", markers=True)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Price Data Sample")
    st.dataframe(price_data.head(20), height=400)

# --- Case Management ---
st.divider()
st.subheader("📁 Case Management")
case_name = st.text_input("Enter Price Scenario Name", value="Base Forecast")

if st.button("💾 Save Price Scenario"):
    price_case = {
        "oil": price_obj.oil_price_by_year,
        "gas": price_obj.gas_price_by_year,
        "params": {
            "oil_init": oil_price_init,
            "gas_init": gas_price_init,
            "inflation": inflation_rate
        }
    }
    st.session_state.price_cases[case_name] = price_case
    st.success(f"Price scenario '{case_name}' saved successfully!")

if st.session_state.price_cases:
    st.write("Saved Scenarios:", list(st.session_state.price_cases.keys()))
