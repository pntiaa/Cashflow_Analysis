import streamlit as st
import pandas as pd
import os
import tempfile
from development import QuestorDevelopmentCost
from utils import ensure_state_init, save_project, render_project_sidebar
from plotting import plot_dev_cost_profile, plot_dev_prod_profile
import plotly.graph_objects as go

st.set_page_config(page_title="Development Cost from QUE$TOR data", layout="wide")

st.title("📂 Development Cost from QUE$TOR")

st.space(size="large")

# --- Initialize Session State & Sidebar ---
ensure_state_init()
render_project_sidebar()

# st.write("### Development Parameters")
# st.caption("Development Parameters")
st.subheader("🏗️ Development Cost from QUE$TOR")

dev_start_year = st.number_input("Development Start Year", value=2024)

# --- 1. File Upload Section ---


col_1, col_2 = st.columns(2)
with col_1.container(border=True, height="stretch"):
    # st.write("### Upload QUE$TOR Excel File")
    uploaded_file = st.file_uploader("Choose a QUE$TOR Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Option for sheet name - Automatically detect sheet names
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        
        # Set default index if "Economic Detail" exists
        default_idx = sheet_names.index("Economic Detail") if "Economic Detail" in sheet_names else 0
        sheet_name = st.selectbox("Select Excel Sheet", options=sheet_names, index=default_idx)
    else:
        st.info("💡 Please upload a QUE$TOR Excel file to get started.")

with col_2.container(border=True, height="stretch"):
    # st.write("### Data Preview")
    st.caption("Data Preview")
    if uploaded_file is not None:
        try:
            # Preview data
            df_preview = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
            st.dataframe(df_preview.head(50))
        
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.info("Ensure the sheet name is correct and the file matches the expected QUE$TOR format.")

st.space(size="small")

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

# --- 2. Calculation Section ---
if st.button("Apply Parameters & Calculate", type="primary"):
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

# --- 3. Visualization ---

if "last_questor_obj" in st.session_state:
    q_dev = st.session_state.last_questor_obj

    st.subheader("📊 Annual Cost Profile")
    st.plotly_chart(plot_dev_cost_profile(q_dev), width='stretch')
    
    st.subheader("📈 Annual Production Profile")
    st.plotly_chart(plot_dev_prod_profile(q_dev), width='stretch')
    
    st.divider()
                
# --- 5. Case Management ---
st.header("📁 Case Management")
case_name = st.text_input("Enter Case Name (Used for both Dev & Prod)", value="Questor Case")

if st.button("💾 Save Questor Case (Dev & Prod)"):
    if not st.session_state.current_project:
        st.error("⚠️ No active project! Please create or select a project in the Sidebar first.")
    else:
        # 1. Save Development Case & Production Case
        # We need to ensure drilling_plan is at least an empty dict to avoid errors
        dev_case = {
            "dev_obj": q_dev, 
            "cost_summary": {
                "total_capex": q_dev.total_capex,
                "total_opex": q_dev.total_opex,
                "total_abex": q_dev.total_abex
            },
            "profiles": {
                "gas": q_dev.annual_gas_production,
                "oil": q_dev.annual_oil_production,
                "drilling_plan": getattr(q_dev, 'yearly_drilling_schedule', {})
            },
        }
        st.session_state.development_cases[case_name] = dev_case

        save_project(st.session_state.current_project)
        st.success(f"Case '{case_name}' saved to Dev & Production project '{st.session_state.current_project}'!")
