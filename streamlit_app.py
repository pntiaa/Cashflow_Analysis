import streamlit as st
from utils import ensure_state_init, list_projects, load_project, save_project

# --- Page Configuration ---
st.set_page_config(
    page_title="Cashflow Analysis App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize Session State ---
ensure_state_init()

# --- Sidebar: Project Management ---
with st.sidebar:
    st.title("📁 Project Management")
    
    project_option = st.radio("Action", ["Select Existing", "Create New"])

    if project_option == "Create New":
        new_project_name = st.text_input("New Project Name")
        if st.button("➕ Create Project"):
            if new_project_name:
                # Initialize empty state for new project
                st.session_state.current_project = new_project_name
                st.session_state.production_cases = {}
                st.session_state.development_cases = {}
                st.session_state.price_cases = {}
                save_project(new_project_name)
                st.success(f"Project '{new_project_name}' created!")
                st.rerun()
            else:
                st.error("Please enter a name.")

    else:
        existing_projects = list_projects()
        if existing_projects:
            # Determine current index
            try:
                current_idx = existing_projects.index(st.session_state.current_project) if st.session_state.current_project in existing_projects else 0
            except ValueError:
                current_idx = 0
                
            selected_project = st.selectbox("Select Project", existing_projects, index=current_idx)
            
            if st.button("📂 Load Project"):
                load_project(selected_project)
                st.success(f"Loaded '{selected_project}'")
                st.rerun()
        else:
            st.info("No projects found. Create one!")

    if st.session_state.current_project:
        st.markdown(f"---")
        st.markdown(f"**Current Project:** `{st.session_state.current_project}`")
    else:
        st.warning("⚠️ No project active. Data will NOT be saved.")

# --- Main Page UI ---
st.title("💰 Cashflow Analysis App")

st.markdown("""
Welcome to the **Cashflow Analysis App**. This tool allows you to perform end-to-end economic evaluations of oil and gas projects.

### Workflow:
1.  **Production**: Define your type curves and generate field production profiles. Save multiple cases for sensitivity analysis.
2.  **Development**: Create development cost scenarios based on your drilling plans.
3.  **Price Deck**: Set up your oil and gas price forecasts and inflation expectations.
4.  **Cash Flow**: Combine your saved cases to calculate NPV, IRR, and overall project economics.

**Get started by selecting or creating a Project in the sidebar.**
""")

with st.sidebar:
    st.divider()
    if st.checkbox("Show Debug Session State"):
        st.write(st.session_state)
