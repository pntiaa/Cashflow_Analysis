import streamlit as st
import pandas as pd
import numpy as np
from production import YearlyProductionProfile
import plotly.express as px
import plotly.graph_objects as go

import math

st.set_page_config(page_title="Production Profile", layout="wide")

st.title("🛢️ Production Profile & Type Curve")

# --- Initialize Session State ---
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'tc_data' not in st.session_state:
    st.session_state.tc_data = None
if 'prod_data' not in st.session_state:
    st.session_state.prod_data = None
if 'drilling_plan_results' not in st.session_state:
    st.session_state.drilling_plan_results = None

tab_p1, tab_p2 = st.tabs(["📈 Type Curve", "🏭 Production Profile"])

with tab_p1:
    col_tc_in1, col_tc_in2 = st.columns(2)
    with col_tc_in1:
        qi_mmcfd = st.number_input("Initial Rate (MMcf/d)", min_value=1.0, value=40.0, key="qi_input")
    with col_tc_in2:
        well_eur_bcf = st.number_input("Well EUR (BCF)", min_value=1.0, value=60.0, key="well_eur_input")
    
    # Duration is needed for TC generation, using the one from tab2 or default
    tc_duration = st.session_state.get("prod_dur_input", 30)
    
    if st.button("🚀 Generate & Display Type Curve", use_container_width=True):
        profile = YearlyProductionProfile(production_duration=int(tc_duration))
        profile.generate_type_curve_from_exponential(
            qi_mmcfd=qi_mmcfd,
            EUR_target_mmcf=well_eur_bcf * 1000,
            T_years=int(tc_duration)
        )
        st.session_state.profile = profile
        st.session_state.tc_data = pd.DataFrame({
            'Year': range(1, len(profile.yearly_type_rate) + 1),
            'Annual Rate (MMcf/y)': profile.yearly_type_rate,
            'Cumulative Production (MMcf)': profile.yearly_type_cum
        })
        st.success("Type Curve Generated Sucessfully!")

    if st.session_state.tc_data is not None:
        tc_df = st.session_state.tc_data
        col_tc_g1, col_tc_g2 = st.columns(2)
        with col_tc_g1:
            fig1 = px.line(tc_df, x='Year', y='Annual Rate (MMcf/y)', title="Annual Rate vs. Years")
            st.plotly_chart(fig1, use_container_width=True)
        with col_tc_g2:
            fig2 = px.line(tc_df, x='Year', y='Cumulative Production (MMcf)', title="Cumulative Production vs. Years")
            st.plotly_chart(fig2, use_container_width=True)

with tab_p2:
    # Reactive Well Calculation
    giip_val = st.session_state.get("giip_input", 4980.0)
    eur_val = st.session_state.get("well_eur_input", 60.0)
    total_wells = math.ceil(giip_val / eur_val)
    st.info(f"🔢 **Estimated Total Wells: {total_wells}** (based on {giip_val:,.1f} BCF Reserves / {eur_val:,.1f} BCF Well EUR)")

    col_pp_in1, col_pp_in2 = st.columns(2)
    with col_pp_in1:
        giip_bcf = st.number_input("Gas Reserves (BCF)", min_value=1.0, value=4980.0, step=100.0, key="giip_input")
        oiip_mmbbl = st.number_input("Oil Reserves (MMbbl)", min_value=0.0, value=329.0, step=10.0, key="oiip_input")
    with col_pp_in2:
        prod_duration = st.number_input("Production Duration (Years)", min_value=1, value=30, key="prod_dur_input")
        drilling_rate = st.number_input("Drilling Rate (Wells/Year)", min_value=1, value=12, key="drilling_rate_input")
        max_prod_rate = st.number_input("Max Field Rate (MMcf/y)", min_value=0, value=250_000, key="max_rate_input")

    if st.button("🚀 Generate Field Production Profile", use_container_width=True):
        if st.session_state.profile is None:
            st.error("⚠️ Please generate a **Type Curve** in the first tab before simulating the field profile.")
        else:
            profile = st.session_state.profile
            # Sync duration if changed
            profile.production_duration = int(prod_duration)
            
            # Use current widget-based reactive well count
            wells_to_drill = math.ceil(giip_bcf / st.session_state.well_eur_input)
            
            drilling_plan = profile.make_drilling_plan(total_wells_number=wells_to_drill, drilling_rate=drilling_rate)
            gas_profile = profile.make_production_profile_yearly(peak_production_annual=max_prod_rate if max_prod_rate > 0 else None)
            
            cgr = (oiip_mmbbl / giip_bcf) * 1000
            oil_profile = {year: gas * cgr / 1000 for year, gas in gas_profile.items()}
            
            st.session_state.prod_data = pd.DataFrame({
                'Year': list(gas_profile.keys()),
                'Gas Production (BCF/y)': list(gas_profile.values()),
                'Oil Production (MMbbl/y)': list(oil_profile.values())
            })
            st.session_state.drilling_plan_results = drilling_plan
            st.session_state.current_cgr = cgr
            st.success("Field Production Profile Generated Successfully!")

    if st.session_state.prod_data is not None:
        col_res1, col_res2 = st.columns([2, 1])
        with col_res1:
            st.subheader("Field Production & Drilling")
            
            # Create subplots with secondary y-axis
            from plotly.subplots import make_subplots
            fig_prod = make_subplots(specs=[[{"secondary_y": True}]])

            # Add Gas Production Bar Chart
            fig_prod.add_trace(
                go.Bar(
                    x=st.session_state.prod_data['Year'],
                    y=st.session_state.prod_data['Gas Production (BCF/y)'],
                    name="Gas Production (BCF/y)",
                    marker_color="lightblue",
                    opacity=0.7
                ),
                secondary_y=False,
            )

            # Add Drilling Plan Scatter Plot
            drill_years = list(st.session_state.drilling_plan_results.keys())
            drill_wells = list(st.session_state.drilling_plan_results.values())
            
            fig_prod.add_trace(
                go.Scatter(
                    x=drill_years,
                    y=drill_wells,
                    name="Wells Drilled",
                    mode="markers+lines+text",
                    text=drill_wells,
                    textposition="top center",
                    marker=dict(size=10, color="orange", symbol="diamond"),
                    line=dict(width=2, color="orange", dash="dot")
                ),
                secondary_y=True,
            )

            # Set axes titles
            fig_prod.update_layout(
                title_text="Annual Field Gas Production & Drilling Count",
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_prod.update_xaxes(title_text="Year")
            fig_prod.update_yaxes(title_text="Gas Production (BCF/y)", secondary_y=False)
            fig_prod.update_yaxes(title_text="Number of Wells", secondary_y=True, rangemode='tozero')

            st.plotly_chart(fig_prod, use_container_width=True)
        with col_res2:
            st.subheader("Drilling Plan")
            drill_df = pd.DataFrame(list(st.session_state.drilling_plan_results.items()), columns=['Rel Year', 'Wells'])
            st.dataframe(drill_df, hide_index=True, use_container_width=True)

# --- Case Management & Persistence ---
st.divider()
st.subheader("📁 Case Management")
case_name = st.text_input("Enter Case Name", value="Base Case")

if st.button("💾 Save Final Case"):
    if st.session_state.prod_data is None:
        st.error("Please generate the production profile before saving.")
    else:
        case_data = {
            "params": {
                "giip_bcf": st.session_state.giip_input,
                "oiip_mmbbl": st.session_state.oiip_input,
                "well_eur_bcf": st.session_state.well_eur_input,
                "drilling_rate": st.session_state.drilling_rate_input,
                "max_prod_rate": st.session_state.max_rate_input
            },
            "profiles": {
                "gas": dict(zip(st.session_state.prod_data['Year'], st.session_state.prod_data['Gas Production (BCF/y)'])),
                "oil": dict(zip(st.session_state.prod_data['Year'], st.session_state.prod_data['Oil Production (MMbbl/y)'])),
                "drilling_plan": st.session_state.drilling_plan_results
            }
        }
        st.session_state.production_cases[case_name] = case_data
        st.success(f"Case '{case_name}' saved successfully!")

# --- Export ---
if st.session_state.prod_data is not None:
    csv = st.session_state.prod_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Production Profile CSV",
        data=csv,
        file_name=f'production_profile_{case_name}.csv',
        mime='text/csv',
    )


# --- Case Management ---
st.divider()
st.subheader("📁 Case Management")
case_name = st.text_input("Enter Case Name", value="Base Case")

if st.button("💾 Save Production Case"):
    case_data = {
        "params": {
            "giip_bcf": giip_bcf,
            "oiip_mmbbl": oiip_mmbbl,
            "well_eur_bcf": well_eur_bcf,
            "drilling_rate": drilling_rate,
            "max_prod_rate": max_prod_rate
        },
        "profiles": {
            "gas": gas_profile,
            "oil": oil_profile,
            "drilling_plan": drilling_plan
        },
        "summary": {
            "total_wells": total_wells,
            "cgr": cgr
        }
    }
    st.session_state.production_cases[case_name] = case_data
    st.success(f"Case '{case_name}' saved successfully!")

if st.session_state.production_cases:
    st.write("Saved Cases:", list(st.session_state.production_cases.keys()))

# --- Export ---
csv = prod_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Production Profile CSV",
    data=csv,
    file_name=f'production_profile_{case_name}.csv',
    mime='text/csv',
)
