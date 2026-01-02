import streamlit as st
import pandas as pd
import os
import tempfile
from development import QuestorDevelopmentCost
from utils import ensure_state_init, save_project, render_project_sidebar
from plotting import plot_cost_profile
import plotly.graph_objects as go

st.set_page_config(page_title="Questor Cost Upload", layout="wide")

st.title("📂 QUE$TOR Cost Processing")

# --- Initialize Session State & Sidebar ---
ensure_state_init()
render_project_sidebar()

# --- 1. File Upload Section ---
st.header("1. Upload QUE$TOR Excel File")
uploaded_file = st.file_uploader("Choose a QUE$TOR Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Option for sheet name - Automatically detect sheet names
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    
    # Set default index if "Economic Detail" exists
    default_idx = sheet_names.index("Economic Detail") if "Economic Detail" in sheet_names else 0
    sheet_name = st.selectbox("Select Excel Sheet", options=sheet_names, index=default_idx)
    
    try:
        # Preview data
        df_preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
        st.subheader("Data Preview")
        st.dataframe(df_preview.head(50))
        
        st.divider()
        
        # --- 2. Parameter Setup ---
        st.header("2. Development Parameters")
        colp1, colp2 = st.columns(2)
        with colp1:
            dev_start_year = st.number_input("Development Start Year", value=2024)
        
        # --- 3. Calculation ---
        if st.button("🚀 Apply Parameters & Calculate", type="primary"):
            # Save uploaded file to a temporary location for the class to read
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            try:
                # Initialize QuestorDevelopmentCost
                q_dev = QuestorDevelopmentCost(
                    dev_start_year=dev_start_year,
                    excel_file_path=tmp_path,
                    sheet_name=sheet_name
                )
                
                # Store in session state for saving later
                st.session_state.last_questor_obj = q_dev
                
                st.success("✅ Calculation successful!")
                
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # --- 4. Visualization ---
        if "last_questor_obj" in st.session_state:
            q_dev = st.session_state.last_questor_obj
            
            st.subheader("📊 Annual Cost Profile")
            
            # Using custom Plotly plotting for better stacked bar experience if plot_cost_profile is Matplotlib-based
            # Checking plotting.py again, plot_cost_profile returns a matplotlib fig.
            # I will create a Plotly version here for better Streamlit integration.
            
            years = sorted(q_dev.cost_years)
            capex_vals = [q_dev.annual_capex.get(y, 0.0) for y in years]
            opex_vals = [q_dev.annual_opex.get(y, 0.0) for y in years]
            abex_vals = [q_dev.annual_abex.get(y, 0.0) for y in years]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=years, y=capex_vals, name='CAPEX'))
            fig.add_trace(go.Bar(x=years, y=opex_vals, name='OPEX'))
            fig.add_trace(go.Bar(x=years, y=abex_vals, name='ABEX'))
            
            fig.update_layout(
                barmode='stack',
                title=f"Annual Expenditure Forecast (MM$)",
                xaxis_title="Year",
                yaxis_title="MM$",
                legend_title="Cost Category"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Added Production Plot ---
            st.subheader("📈 Annual Production Profile")
            p_years = sorted(q_dev.production_years)
            gas_vals = [q_dev.annual_gas_production.get(y, 0.0) for y in p_years]
            oil_vals = [q_dev.annual_oil_production.get(y, 0.0) for y in p_years]
            
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=p_years, y=gas_vals, mode='lines+markers', name='Gas (BCF)'))
            fig_p.add_trace(go.Scatter(x=p_years, y=oil_vals, mode='lines+markers', name='Oil/Cond. (MMbbl)'))
            
            fig_p.update_layout(
                title="Annual Production Profile",
                xaxis_title="Year",
                yaxis_title="Volume",
                legend_title="Product"
            )
            st.plotly_chart(fig_p, use_container_width=True)
            
            st.divider()
            
            # --- 5. Case Management ---
            st.header("📁 Case Management")
            case_name = st.text_input("Enter Case Name (Used for both Dev & Prod)", value="Questor Case")
            
            if st.button("💾 Save Questor Case (Dev & Prod)"):
                if not st.session_state.current_project:
                    st.error("⚠️ No active project! Please create or select a project in the Sidebar first.")
                else:
                    # 1. Save Development Case
                    cost_case = {
                        "dev_obj": q_dev,
                        "summary": {
                            "total_capex": q_dev.total_capex,
                            "total_opex": q_dev.total_opex,
                            "total_abex": q_dev.total_abex
                        }
                    }
                    st.session_state.development_cases[case_name] = cost_case
                    
                    # 2. Save Production Case
                    # We need to ensure drilling_plan is at least an empty dict to avoid errors
                    prod_case = {
                        "profiles": {
                            "gas": q_dev.annual_gas_production,
                            "oil": q_dev.annual_oil_production,
                            "drilling_plan": getattr(q_dev, 'yearly_drilling_schedule', {})
                        }
                    }
                    st.session_state.production_cases[case_name] = prod_case
                    
                    save_project(st.session_state.current_project)
                    st.success(f"Case '{case_name}' saved to Dev & Production project '{st.session_state.current_project}'!")
                    
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.info("Ensure the sheet name is correct and the file matches the expected QUE$TOR format.")

else:
    st.info("💡 Please upload a QUE$TOR Excel file to get started.")
