import streamlit as st
import pandas as pd
import plotly.express as px
from cashflow import CashFlowKOR, CompanyConfig, MultiCompanyCashFlow
from utils import ensure_state_init, render_project_sidebar
from plotting import plot_cashflow, summary_plot, plot_production_profile
import io

st.set_page_config(page_title="Multi-Company Analysis", layout="wide")

st.title("🏢 Multi-Company Analysis")

st.space(size="large")

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
st.subheader("🏁 Run Project Scenario")
col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    prod_name = st.selectbox("Select Production Case", list(st.session_state.production_cases.keys()))
with col_s2:
    dev_name = st.selectbox("Select Development Case", list(st.session_state.development_cases.keys()))
with col_s3:
    price_name = st.selectbox("Select Price Scenario", list(st.session_state.price_cases.keys()))

# --- Global Economic Inputs ---
with st.expander("⚙️ Project Economic Parameters"):
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        base_year = st.number_input("Base Year (for PV)", value=2024)
        discount_rate = st.number_input("Discount Rate (fraction)", value=0.10, format="%.2f")
    with col_e2:
        exchange_rate = st.number_input("Exchange Rate (KRW/USD)", value=1350.0)
        cost_inflation = st.number_input("Cost Inflation Rate", value=0.015)
    with col_e3:
        useful_life = st.number_input("Useful Life (Depreciation)", value=10)

st.divider()

# --- Company Configuration ---
st.subheader("👥 Participating Companies")

if 'companies_config' not in st.session_state:
    st.session_state.companies_config = [
        {'name': 'Company A', 'pi': 0.51, 'farm_in_expo_share': None, 'farm_in_expo_cap': None},
        {'name': 'Company B', 'pi': 0.49, 'farm_in_expo_share': None, 'farm_in_expo_cap': None},
    ]

def add_company():
    st.session_state.companies_config.append({'name': f'Company {len(st.session_state.companies_config)+1}', 'pi': 0.0, 'farm_in_expo_share': None, 'farm_in_expo_cap': None})

def remove_company(index):
    st.session_state.companies_config.pop(index)

for i, comp in enumerate(st.session_state.companies_config):
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 0.5])
        comp['name'] = c1.text_input(f"Name", value=comp['name'], key=f"name_{i}")
        comp['pi'] = c2.number_input(f"PI (fraction)", value=comp['pi'], step=0.01, format="%.2f", key=f"pi_{i}")
        
        has_carry = c3.checkbox("Farm-in Carry?", value=comp['farm_in_expo_share'] is not None, key=f"carry_check_{i}")
        if has_carry:
            comp['farm_in_expo_share'] = c3.number_input("Carry Share", value=comp.get('farm_in_expo_share', 1.0) or 1.0, key=f"share_{i}")
            comp['farm_in_expo_cap'] = c4.number_input("Carry Cap (MM$)", value=comp.get('farm_in_expo_cap', 10.0) or 10.0, key=f"cap_{i}")
        else:
            comp['farm_in_expo_share'] = None
            comp['farm_in_expo_cap'] = None
            
        if c5.button("🗑️", key=f"del_{i}"):
            remove_company(i)
            st.rerun()

st.button("➕ Add Company", on_click=add_company)

total_pi = sum(c['pi'] for c in st.session_state.companies_config)
if abs(total_pi - 1.0) > 0.001:
    st.warning(f"⚠️ Total PI is {total_pi*100:.1f}%. It should ideally be 100%.")

# --- Run Button ---
if 'multi_cf_results' not in st.session_state:
    st.session_state.multi_cf_results = None

run_button = st.button("🚀 Run Multi-Company Analysis", type="secondary")

if run_button:
    # --- Calculation ---
    p_case = st.session_state.production_cases[prod_name]
    d_case = st.session_state.development_cases[dev_name]
    s_case = st.session_state.price_cases[price_name]
    dev_obj = d_case['dev_obj']

    # 1. Setup Project Instance
    project_cf = CashFlowKOR(
        base_year=base_year,
        oil_price_by_year=s_case['oil'],
        gas_price_by_year=s_case['gas'],
        cost_inflation_rate=cost_inflation,
        discount_rate=discount_rate,
        exchange_rate=exchange_rate
    )
    
    project_cf.set_development_costs(dev_obj, output=False)
    
    def ensure_actual_years(d, start_year):
        if not d: return {}
        min_key = min(d.keys())
        if min_key < 1000: return {int(k) + start_year: v for k, v in d.items()}
        return {int(k): v for k, v in d.items()}

    oil_shifted = ensure_actual_years(dev_obj.annual_oil_production, dev_obj.drill_start_year)
    gas_shifted = ensure_actual_years(dev_obj.annual_gas_production, dev_obj.drill_start_year)
    project_cf.set_production_profile_from_dicts(oil_dict=oil_shifted, gas_dict=gas_shifted)
    
    # Run Project calculations
    project_cf.calculate_annual_revenue(output=False)
    project_cf.calculate_royalty()
    project_cf.calculate_taxes(output=False)
    project_cf.calculate_net_cash_flow(output=False)
    project_cf.calculate_npv(output=False)

    # 2. Setup Multi-Company Logic
    comp_configs = [CompanyConfig(**c) for c in st.session_state.companies_config]
    multi_cf = MultiCompanyCashFlow(project_cf, comp_configs)
    multi_cf.calculate()
    
    # Store in session state
    st.session_state.multi_cf_results = {
        'multi_cf': multi_cf,
        'prod_name': prod_name
    }

if st.session_state.multi_cf_results:
    results = st.session_state.multi_cf_results
    multi_cf = results['multi_cf']
    prod_name = results['prod_name']
    comp_configs = multi_cf.companies

    # --- UI Results ---
    st.divider()
    st.subheader("📊 Comparison Summary")
    
    summary_df = multi_cf.get_summary_df()
    st.dataframe(summary_df.style.format({
        'PI (%)': '{:.1f}%',
        'NPV (MM$)': '{:,.2f}',
        'IRR (%)': '{:.2f}%',
        'Total Revenue (MM$)': '{:,.2f}',
        'Total CAPEX (MM$)': '{:,.2f}',
        'Net Cash Flow (MM$)': '{:,.2f}'
    }), width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### NPV Distribution")
        plot_df = summary_df[summary_df['Company'] != 'PROJECT TOTAL']
        fig_npv = px.pie(plot_df, values='NPV (MM$)', names='Company', hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_npv, width='stretch')
    with c2:
        st.markdown("### CAPEX Distribution")
        fig_capex = px.pie(plot_df, values='Total CAPEX (MM$)', names='Company', hole=0.4,
                           color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig_capex, width='stretch')

    st.divider()
    st.subheader("📈 Individual Company Details")
    
    selected_comp_name = st.selectbox("Select Company to view details", [c.name for c in comp_configs])
    ccf = multi_cf.company_results[selected_comp_name]
    
    col_det1, col_det2 = st.columns([1, 2])
    with col_det1:
        st.markdown(f"#### {selected_comp_name} Summary")
        det_summ = ccf.get_project_summary()
        st.metric("Individual NPV", f"{det_summ['npv']:,.2f} MM$")
        st.metric("Individual IRR", f"{det_summ['irr']*100:.2f}%" if isinstance(det_summ['irr'], (int, float)) else "N/A")
        st.metric("Net Cash Flow", f"{det_summ['final_cumulative']:,.2f} MM$")
    
    with col_det2:
        st.plotly_chart(plot_cashflow(ccf), width='stretch')

    with st.expander(f"📄 Detailed Annual Table for {selected_comp_name}"):
        st.dataframe(ccf.to_df(), width='stretch')

    # --- Export ---
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Comparison Summary')
        for name, ccf_inst in multi_cf.company_results.items():
            ccf_inst.to_df().to_excel(writer, sheet_name=f'{name[:30]} Details')
    
    st.download_button(
        label="📥 Download Multi-Company Economic Report (Excel)",
        data=output.getvalue(),
        file_name=f"multi_company_report_{prod_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("💡 Adjust company profiles and click 'Run Multi-Company Analysis' to see results.")
