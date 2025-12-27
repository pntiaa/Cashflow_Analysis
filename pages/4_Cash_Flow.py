import streamlit as st
import pandas as pd
from cashflow import CashFlowKOR
from utils import ensure_state_init, render_project_sidebar
from plotting import plot_cashflow, summary_plot, plot_cf_sankey_chart
import io

st.set_page_config(page_title="Cash Flow Analysis", layout="wide")

st.title("💰 Cash Flow Analysis")

# --- Initialize Session State & Sidebar ---
ensure_state_init()
render_project_sidebar()

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
        depreciation_method = st.selectbox("Depreciation Method", ["Unit of Production", "Straight Line", "Declining Balance"], index=0)
        useful_life = st.number_input("Useful Life (Depreciation)", value=10)

# --- Run Button ---
run_button = st.button("🚀 Run Cash Flow Analysis", width='content', type="secondary")

if run_button:
    # --- Calculation ---
    p_case = st.session_state.production_cases[prod_name]
    d_case = st.session_state.development_cases[dev_name]
    s_case = st.session_state.price_cases[price_name]
    
    dev_obj = d_case['dev_obj']

    # Setup Cash Flow Instance
    cf = CashFlowKOR(
        base_year=base_year,
        oil_price_by_year=s_case['oil'],
        gas_price_by_year=s_case['gas'],
        cost_inflation_rate=cost_inflation,
        discount_rate=discount_rate,
        exchange_rate=exchange_rate
    )
    
    # Set Development and Production
    # IMPORTANT: Use the production profiles from the dev_obj because they are already 
    # shifted to the actual calendar years (drill_start_year).
    cf.set_development_costs(dev_obj, output=False)
    
    # Precaution: if for some reason dev_obj has indices instead of years, shift them
    def ensure_actual_years(d, start_year):
        if not d: return {}
        min_key = min(d.keys())
        if min_key < 1000: # It's likely an index (1, 2, 3...)
            return {int(k) + start_year: v for k, v in d.items()}
        return {int(k): v for k, v in d.items()}

    oil_shifted = ensure_actual_years(dev_obj.annual_oil_production, dev_obj.drill_start_year)
    gas_shifted = ensure_actual_years(dev_obj.annual_gas_production, dev_obj.drill_start_year)

    cf.set_production_profile_from_dicts(
        oil_dict=oil_shifted,
        gas_dict=gas_shifted
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
    
    # Summary Section
    st.subheader("📊 Economic Summary")
    summ = cf.get_project_summary()
    
    col_r1, col_r2 = st.columns([1, 2])
    
    # Key Metrics Table
    with col_r1:
        st.markdown("### Key Metrics")
        col_metrics1, col_metrics2 = st.columns([1, 2])
        col_metrics1.metric("NPV (Discounted)", f"{summ['npv']:,.0f} MM$")
        col_metrics1.metric("IRR", f"{summ['irr']*100:.1f}%" if isinstance(summ['irr'], (int, float)) else "N/A")
        col_metrics1.metric("Payback Year", f"{summ['payback_year']}" if summ['payback_year'] else "N/A")
        col_metrics2.metric("Total Revenue", f"{summ['total_revenue']:,.0f} MM$")
        col_metrics2.metric("Total CAPEX", f"{summ['total_capex']:,.0f} MM$")
        col_metrics2.metric("Net Cash Flow (Total)", f"{summ['final_cumulative']:,.0f} MM$")

    # Plot Cash Flow Chart
    with col_r2:
        st.plotly_chart(plot_cashflow(cf), width='content')
        # st.plotly_chart(plot_cf_waterfall_chart(cf, height=500), width='content')

    st.divider()
    st.subheader("📈 Detailed Visualizations")
    
    # Dashboards from plotting
    st.plotly_chart(summary_plot(cf), width='stretch')

    st.subheader("Cash Flow Profile")
    st.plotly_chart(plot_cashflow(cf), width='stretch')
    
    with st.expander("🔗 Cash Flow - Sankey Diagram"):
        st.plotly_chart(plot_cf_sankey_chart(cf, height=700), width='stretch')
    
    # Detailed Results Expander
    with st.expander("📄 Detailed Annual Cash Flow Table"):
        years = cf.all_years
        detail_df = pd.DataFrame({
            'Year': years,
            'Oil Price': [cf.oil_price_by_year.get(y,0.0) for y in years],
            'Gas Price': [cf.gas_price_by_year.get(y,0.0) for y in years],
            'Oil Production': [cf.annual_oil_production.get(y,0.0) for y in years],
            'Gas Production': [cf.annual_gas_production.get(y,0.0) for y in years],
            'Revenue': [cf.annual_revenue.get(y,0.0) for y in years],
            'Royalty': [cf.annual_royalty.get(y,0.0) for y in years],
            'CAPEX': [cf.annual_capex.get(y,0.0) for y in years],
            'OPEX': [cf.annual_opex.get(y,0.0) for y in years],
            'ABEX': [cf.annual_abex.get(y,0.0) for y in years],
            'Tax': [cf.annual_total_tax.get(y,0.0) for y in years],
            'NCF': [cf.annual_net_cash_flow.get(y,0.0) for y in years],
            'Cum CF': [cf.cumulative_cash_flow.get(y,0.0) for y in years]
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
else:
    st.info("💡 Click the button above to run the economic analysis.")
