import streamlit as st
import pandas as pd
from cashflow import CashFlow_KOR_Regime
from utils import plot_cash_flow_profile_plotly, summary_plot
import io

st.set_page_config(page_title="Cash Flow Analysis", layout="wide")

st.title("💰 Cash Flow Analysis")

# --- Dependency Check ---
missing_deps = []
if not st.session_state.production_cases: missing_deps.append("Production")
if not st.session_state.development_cases: missing_deps.append("Development")
if not st.session_state.price_cases: missing_deps.append("Price Deck")

if missing_deps:
    st.warning(f"⚠️ Missing saved cases from: {', '.join(missing_deps)}. Please complete those pages first.")
    st.stop()

# --- Scenario Selection ---
st.subheader("🏁 Run Economic Scenario")
col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    prod_name = st.selectbox("Select Production Case", list(st.session_state.production_cases.keys()))
with col_s2:
    dev_name = st.selectbox("Select Development Case", list(st.session_state.development_cases.keys()))
with col_s3:
    price_name = st.selectbox("Select Price Scenario", list(st.session_state.price_cases.keys()))

# --- Global Economic Inputs ---
with st.expander("⚙️ Global Economic Parameters"):
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        base_year = st.number_input("Base Year (for PV)", value=2024)
        discount_rate = st.number_input("Discount Rate (fraction)", value=0.10, format="%.2f")
    with col_e2:
        exchange_rate = st.number_input("Exchange Rate (KRW/USD)", value=1350.0)
        cost_inflation = st.number_input("Cost Inflation Rate", value=0.015)
    with col_e3:
        useful_life = st.number_input("Useful Life (Depreciation)", value=10)

# --- Calculation ---
p_case = st.session_state.production_cases[prod_name]
d_case = st.session_state.development_cases[dev_name]
s_case = st.session_state.price_cases[price_name]

# Setup Cash Flow Instance
cf = CashFlow_KOR_Regime(
    base_year=base_year,
    oil_price_by_year=s_case['oil'],
    gas_price_by_year=s_case['gas'],
    cost_inflation_rate=cost_inflation,
    discount_rate=discount_rate,
    exchange_rate=exchange_rate
)

# Set Development and Production
cf.set_development_costs(d_case['dev_obj'], output=False)
cf.set_production_profile_from_dicts(
    oil_dict=p_case['profiles']['oil'],
    gas_dict=p_case['profiles']['gas']
)

# Run Full Cycle
cf.calculate_annual_revenue(output=False)
cf.calculate_depreciation(method='straight_line', useful_life=useful_life, output=False)
cf.calculate_royalty()
cf.calculate_taxes(output=False)
cf.calculate_net_cash_flow(output=False)
cf.calculate_npv(output=False)

# --- UI Results ---
st.divider()

# Dashboards from utils
st.plotly_chart(summary_plot(cf), use_container_width=True)
st.plotly_chart(plot_cash_flow_profile_plotly(cf), use_container_width=True)

# Detailed Results Expander
with st.expander("📄 Detailed Annual Cash Flow Table"):
    years = cf.all_years
    detail_df = pd.DataFrame({
        'Year': years,
        'Oil Price': [cf.oil_price_by_year.get(y) for y in years],
        'Gas Production': [cf.gas_production_by_year.get(y) for y in years],
        'Revenue': [cf.annual_revenue.get(y) for y in years],
        'CAPEX': [cf.annual_capex.get(y) for y in years],
        'OPEX': [cf.annual_opex.get(y) for y in years],
        'Tax': [cf.annual_total_tax.get(y) for y in years],
        'NCF': [cf.annual_net_cash_flow.get(y) for y in years],
        'Cum CF': [cf.cumulative_cash_flow.get(y) for y in years]
    })
    st.dataframe(detail_df.style.format("{:.2f}"))

# --- Final Export ---
st.subheader("📊 Export Results")

# Create Excel buffer
output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    detail_df.to_excel(writer, index=False, sheet_name='Annual Cash Flow')
    
    # Summary Sheet
    summ = cf.get_project_summary()
    summary_df = pd.DataFrame(list(summ.items()), columns=['Metric', 'Value'])
    summary_df.to_excel(writer, index=False, sheet_name='Summary')

st.download_button(
    label="📥 Download Full Economic Report (Excel)",
    data=output.getvalue(),
    file_name=f"economic_report_{prod_name}_{dev_name}_{price_name}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
