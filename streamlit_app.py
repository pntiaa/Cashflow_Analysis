import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Cashflow Analysis App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize Session State ---
def init_session_state():
    # Production Cases
    if "production_cases" not in st.session_state:
        st.session_state.production_cases = {}
    
    # Development Cases
    if "development_cases" not in st.session_state:
        st.session_state.development_cases = {}
        
    # Price Deck Cases
    if "price_cases" not in st.session_state:
        st.session_state.price_cases = {}

# Run initialization
init_session_state()

# --- Main Page UI ---
st.title("💰 Cashflow Analysis App")

st.markdown("""
Welcome to the **Cashflow Analysis App**. This tool allows you to perform end-to-end economic evaluations of oil and gas projects.

### Workflow:
1.  **Production**: Define your type curves and generate field production profiles. Save multiple cases for sensitivity analysis.
2.  **Development**: Create development cost scenarios based on your drilling plans.
3.  **Price Deck**: Set up your oil and gas price forecasts and inflation expectations.
4.  **Cash Flow**: Combine your saved cases to calculate NPV, IRR, and overall project economics.

**Get started by selecting "1 Production" from the sidebar.**
""")

with st.sidebar:
    st.info("Navigate through the stages using the sidebar above.")
    
    if st.checkbox("Show Debug Session State"):
        st.write(st.session_state)
