import streamlit as st
from utils import ensure_state_init, render_project_sidebar

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
render_project_sidebar()

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
