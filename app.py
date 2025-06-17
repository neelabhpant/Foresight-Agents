# app.py
import streamlit as st
import pandas as pd
from crew.main import WalmartForecastingCrew
import os

st.set_page_config(page_title="Walmart Sales Forecaster", layout="wide")

st.title("🤖 Walmart Sales Forecasting Agent")
st.markdown("""
Welcome! This tool uses a crew of AI agents to generate a 52-week sales forecast for any Walmart store.
Select a store ID from the dropdown below and click the button to generate your report.
""")

@st.cache_data
def load_store_ids():
    """Loads the dataset once and caches the list of unique store IDs."""
    try:
        df = pd.read_excel('data/Walmart_Sales.xlsx')
        return sorted(df['Store'].unique())
    except FileNotFoundError:
        st.error("Error: 'data/Walmart_Sales.xlsx' not found. Please make sure the data file is in the correct directory.")
        return []

store_ids = load_store_ids()

if not store_ids:
    st.warning("Could not load store IDs. Please check the data file.")
else:
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_store = st.selectbox(
            '**Select a Store ID**',
            options=store_ids,
            index=0
        )

        if st.button('🚀 Generate Forecast Report'):
            if selected_store:
                with st.spinner(f"🤖 The agent crew is analyzing Store {selected_store}. This may take a minute..."):
                    try:
                        forecasting_crew = WalmartForecastingCrew(store_id=selected_store)
                        report = forecasting_crew.run()
                        st.session_state.report = report
                        st.session_state.plot_path = f'outputs/store_{selected_store}_forecast_plot.png'
                        st.session_state.run_complete = True
                    except Exception as e:
                        st.error(f"An error occurred while running the crew: {e}")
                        st.session_state.run_complete = False
            else:
                st.warning("Please select a store ID.")

    if 'run_complete' in st.session_state and st.session_state.run_complete:
        with col2:
            st.markdown("### 📊 Forecast Report")
            st.markdown(st.session_state.report)

            plot_path = st.session_state.get('plot_path')
            if plot_path and os.path.exists(plot_path):
                st.markdown("### 📈 Forecast Plot")
                st.image(plot_path, caption=f"52-Week Sales Forecast for Store {selected_store}")
            else:
                st.warning("Forecast plot image not found.")