import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from FERidge_classes2 import ConstrainedFEModel, ConstrainedFEScenarioSimulator

TARGETS = ["Baseload_Price_EUR_MWh", "PV_Captured_Price_EUR_MWh"]

NUM_VARS = [
    "Carbon_price_EU_ETS_EUR_tonne", 
    "Gas_price_PSV_EUR_MWh",
    "Electricity_demand_TWh",
    "PV_Capacity_GW",
    "Wind_Capacity_GW",
    "Hydro_Generation_TWh",
    "Battery_storage_Capacity_GW",
]

# Model coefficients (fallback values if model coefficients are not accessible)
MODEL_COEFFICIENTS = {
    "Baseload_Price_EUR_MWh": {
        "Carbon_price_EU_ETS_EUR_tonne": 0.313380,#0.323380,
        "Gas_price_PSV_EUR_MWh": 1.399184,#1.411840,
        "Electricity_demand_TWh": 0.365280,#0.371280,
        "PV_Capacity_GW": -0.604740,#-0.634740,
        "Wind_Capacity_GW": -0.571800,
        "Hydro_Generation_TWh": -0.569845,#-0.549845,
        "Battery_storage_Capacity_GW": -0.277740,#-0.257740

    },
    "PV_Captured_Price_EUR_MWh": {
        "Carbon_price_EU_ETS_EUR_tonne": 0.295430,
        "Gas_price_PSV_EUR_MWh": 0.622685,
        "Electricity_demand_TWh": 0.321280,#0.351280,
        "PV_Capacity_GW": -0.758200,
        "Wind_Capacity_GW": -0.553750,#0.593750,
        "Hydro_Generation_TWh":-0.305300,
        "Battery_storage_Capacity_GW": 0.615700
    }
}
CAPACITY_FACTOR_HYDRO = pd.read_csv("capacity_factor_hydro.csv")

#path = os.getcwd() + "\\"

st.set_page_config(
    page_title = "Long Term Energy Price Scenarios",
    page_icon = "⚡",
    layout = "wide"
    )

st.title("Long Term Energy Price Scenarios ⚡")
st.markdown("---")

@st.cache_data

def load_model_and_data(scenario,data_plot):
    try:
        baseload_model_file = f'baseload_{scenario.lower()}_model_9.pkl'
        pv_model_file = f'pv_{scenario.lower()}_model_9.pkl'
        data_file = f'processed_{scenario.lower()}_scenario_data_6.csv'
        with open(baseload_model_file, 'rb') as f:
            baseload_model = pickle.load(f)
        with open(pv_model_file, 'rb') as f:
            pv_model = pickle.load(f)
        data = pd.read_csv(data_file)
        data["Year"] = data["Year"].astype(int)
        data = data.loc[data['Year'] >= 2026].reset_index(drop=True)
        data["Release"] = data["Release"].astype(str)
        simulator_baseload = ConstrainedFEScenarioSimulator(
            fitted_model=baseload_model,
            training_data=data,
            baseline_data=data_plot
            )
        simulator_pv = ConstrainedFEScenarioSimulator(
            fitted_model=pv_model,
            training_data=data,
            baseline_data=data_plot
            )
        return simulator_baseload, simulator_pv, baseload_model, pv_model, data
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None, None, None, None

data_plot = pd.read_csv("data_baseline_scenario_3.csv")
# Load hydro capacity factors for calculations
try:
    hydro_capacity_factors = pd.read_csv("capacity_factor_hydro.csv")
except:
    st.error("Missing capacity_factor_hydro.csv file needed for hydro calculations.")
    st.stop()

scenario = "Central"
simulator_baseload, simulator_pv, baseload_model, pv_model, data = load_model_and_data(scenario,data_plot)
if simulator_baseload is None or data is None or simulator_pv is None:
    st.error("Failed to load model or data. Please check if the files exist.")
    st.stop()

def calculate_hydro_generation(new_hydro_capacity, capacity_factor):
    """
    Calculate hydro generation based on capacity and capacity factor
    
    Formula:
    - new_generation_hydro = capacity_user_hydro * capacity_factor * 8760/1000
    
    Logic: If hydro capacity changes -> calculate corresponding hydro generation
    """
    new_generation_hydro = new_hydro_capacity * capacity_factor * 8760 / 1000
    return new_generation_hydro

#data_plot = simulator_baseload.baseline_data.copy()
#data_plot = pd.merge(data_plot, simulator_baseload.baseline_capacity, on='Year', how='left')
# Hydro_Generation_TWh is already present in baseline_data
#data_plot.to_csv("data_plot.csv", index=False)
variable_info = {
    'PV_Capacity_GW': {'icon': '☀️', 'unit': 'GW', 'description': 'Photovoltaic Capacity', 'color': '#FF6B35'},
    'Wind_Capacity_GW': {'icon': '💨', 'unit': 'GW', 'description': 'Wind Capacity', 'color': '#4ECDC4'},
    'Battery_storage_Capacity_GW': {'icon': '🔋', 'unit': 'GW', 'description': 'Battery Storage Capacity', 'color': '#22C55E'},
    'Hydro_Generation_TWh': {'icon': '🌊', 'unit': 'TWh', 'description': 'Hydro Generation', 'color': '#0891B2'},
    'Electricity_demand_TWh': {'icon': '⚡', 'unit': 'TWh', 'description': 'Electricity Demand', 'color': '#FFAA00'},
    'Gas_price_PSV_EUR_MWh': {'icon': '🔥', 'unit': '€/MWh', 'description': 'Gas Price PSV', 'color': '#96CEB4'},
    'Carbon_price_EU_ETS_EUR_tonne': {'icon': '🌿', 'unit': '€/tonne CO₂', 'description': 'EUA Price', 'color': '#92400E'}
}



all_variables = list(variable_info.keys())
years_data_all = data_plot[(data_plot['Year'] >= 2026) & (data_plot['Year'] <= 2040)].copy()

if 'edited_data_all' not in st.session_state:
    st.session_state.edited_data_all = {}
    for var in all_variables:
        if var in years_data_all.columns:
            var_data = years_data_all.groupby('Year')[var].mean().reset_index()
            st.session_state.edited_data_all[var] = var_data.copy()
    # Also initialize Hydro_Capacity_GW for the editor (not in variable_info but needed for sidebar)
    if 'Hydro_Capacity_GW' in years_data_all.columns:
        hydro_cap_data = years_data_all.groupby('Year')['Hydro_Capacity_GW'].mean().reset_index()
        st.session_state.edited_data_all['Hydro_Capacity_GW'] = hydro_cap_data.copy()

# Track which years have been modified
if 'modified_years_set' not in st.session_state:
    st.session_state.modified_years_set = set()

# Track which specific variables have been modified for each year
if 'modified_variables_by_year' not in st.session_state:
    st.session_state.modified_variables_by_year = {}  # {year: {var1, var2, ...}}

st.subheader("⚡ Energy Price Drivers & Market Fundamentals")
st.markdown("    **Interactive scenario modeling** - Modify key market variables and analyze their impact on energy prices in the Italian market.")
st.markdown("Edit the table on the right: double click on the cell you want to modify and write the new value. Changes are saved automatically")

tabs = st.tabs([f"{variable_info[var]['icon']} {variable_info[var]['description']}" for var in all_variables])
for i, var in enumerate(all_variables):
    with tabs[i]:
        if var in st.session_state.edited_data_all and var in years_data_all.columns:
            years_data = years_data_all.groupby('Year')[var].mean().reset_index()
            edited_data = st.session_state.edited_data_all[var]
            col1, col2 = st.columns([3, 1.5])
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=edited_data['Year'],
                    y=[round(val, 2) for val in edited_data[var]],
                    mode='lines+markers',
                    name=f'Modified {var}',
                    line=dict(color=variable_info[var]['color'], width=3),
                    marker=dict(size=12, color=variable_info[var]['color'], line=dict(color='white', width=2)),
                    hovertemplate='<b>Year:</b> %{x}<br><b>Modified Data:</b> %{y:.2f}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=years_data['Year'],
                    y=[round(val, 2) for val in years_data[var]],
                    mode='lines+markers',
                    name='Original',
                    line=dict(color='lightgray', width=2, dash='dash'),
                    marker=dict(size=8, color='lightgray'),
                    hovertemplate='<b>Year:</b> %{x}<br><b>Baseline:</b> %{y:.2f}<extra></extra>'
                ))
                fig.update_layout(
                    title=f'{variable_info[var]["description"]}',
                    xaxis_title='Year',
                    yaxis_title=f'{variable_info[var]["description"]} ({variable_info[var]["unit"]})',
                    hovermode='x unified',
                    height=500,
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='lightgray', gridwidth=0.5, range=[2025.5, 2040.5]),
                    yaxis=dict(gridcolor='lightgray', gridwidth=0.5)
                )
                st.plotly_chart(fig, use_container_width=True, key=f"plot_{var}")
            with col2:
                    st.markdown("### 🎛️ Change Scenario")
                    # Prepare column name with description and unit
                    var_desc = f"{variable_info[var]['description']}\n[{variable_info[var]['unit']}]"
                    
                    # Special handling for Hydro Generation tab - show capacity editor instead
                    if var == 'Hydro_Generation_TWh':
                        # Initialize hydro capacity factors in session state if not present
                        if 'hydro_capacity_factors_edited' not in st.session_state:
                            st.session_state.hydro_capacity_factors_edited = hydro_capacity_factors.copy()
                        
                        # Get Hydro_Capacity_GW data for the editor
                        if 'Hydro_Capacity_GW' not in st.session_state.edited_data_all:
                            hydro_cap_data = years_data_all.groupby('Year')['Hydro_Capacity_GW'].mean().reset_index()
                            st.session_state.edited_data_all['Hydro_Capacity_GW'] = hydro_cap_data.copy()
                        
                        editable_df = st.session_state.edited_data_all['Hydro_Capacity_GW'][['Year', 'Hydro_Capacity_GW']].copy()
                        
                        # Add capacity factor column to the dataframe
                        capacity_factors_for_years = []
                        for year in editable_df['Year']:
                            cf_row = st.session_state.hydro_capacity_factors_edited[st.session_state.hydro_capacity_factors_edited['Year'] == year]
                            if not cf_row.empty:
                                capacity_factors_for_years.append(float(f"{cf_row['Capacity_factor_hydro'].iloc[0]:.3f}"))
                            else:
                                capacity_factors_for_years.append(0.175)  # Default capacity factor
                        
                        editable_df['Capacity_Factor'] = capacity_factors_for_years
                        
                        updated_df = st.data_editor(
                            editable_df,
                            num_rows="static",
                            use_container_width=True,
                            column_config={
                                "Year": st.column_config.Column("Year", disabled=True, width="small"),
                                "Hydro_Capacity_GW": st.column_config.NumberColumn("Capacity [GW]", required=True, min_value=0, max_value=1000, step=0.01, format="%.2f"),
                                "Capacity_Factor": st.column_config.NumberColumn("Capacity Factor", required=True, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
                            },
                            key=f"data_editor_{var}",
                            hide_index=True
                        )
                        st.markdown("💡 **Tip**: Modify capacity and capacity factor. Generation is calculated automatically.")
                        
                        # Handle changes to capacity or capacity factor
                        if not updated_df[['Year', 'Hydro_Capacity_GW']].equals(editable_df[['Year', 'Hydro_Capacity_GW']]) or \
                           not all(updated_df['Capacity_Factor'] == editable_df['Capacity_Factor']):
                            for idx, (_, row) in enumerate(updated_df.iterrows()):
                                year = row['Year']
                                old_capacity = editable_df.iloc[idx]['Hydro_Capacity_GW']
                                new_capacity = row['Hydro_Capacity_GW']
                                old_cf = editable_df.iloc[idx]['Capacity_Factor']
                                new_cf = row['Capacity_Factor']
                                
                                capacity_changed = abs(new_capacity - old_capacity) > 1e-6
                                cf_changed = abs(new_cf - old_cf) > 1e-6
                                
                                if capacity_changed or cf_changed:
                                    # Persist capacity factor change
                                    if cf_changed:
                                        st.session_state.hydro_capacity_factors_edited.loc[
                                            st.session_state.hydro_capacity_factors_edited['Year'] == year,
                                            'Capacity_factor_hydro'
                                        ] = new_cf
                                    
                                    # Update capacity
                                    if capacity_changed:
                                        capacity_mask = st.session_state.edited_data_all['Hydro_Capacity_GW']['Year'] == year
                                        st.session_state.edited_data_all['Hydro_Capacity_GW'].loc[capacity_mask, 'Hydro_Capacity_GW'] = new_capacity
                                    
                                    # Calculate and store new hydro generation
                                    new_generation = calculate_hydro_generation(new_capacity, new_cf)
                                    generation_mask = st.session_state.edited_data_all['Hydro_Generation_TWh']['Year'] == year
                                    st.session_state.edited_data_all['Hydro_Generation_TWh'].loc[generation_mask, 'Hydro_Generation_TWh'] = new_generation
                                    
                                    # Track modification for scenario computation
                                    st.session_state.modified_years_set.add(year)
                                    if year not in st.session_state.modified_variables_by_year:
                                        st.session_state.modified_variables_by_year[year] = set()
                                    st.session_state.modified_variables_by_year[year].add('Hydro_Generation_TWh')
                            
                            st.rerun()
                    else:
                        # Standard data editor for other variables
                        editable_df = edited_data[['Year', var]].copy()
                        
                        updated_df = st.data_editor(
                            editable_df,
                            num_rows="static",
                            use_container_width=True,
                            column_config={
                                "Year": st.column_config.Column("Year", disabled=True),
                                var: st.column_config.Column(f"{variable_info[var]['description']} [{variable_info[var]['unit']}]", required=True)
                            },
                            key=f"data_editor_{var}"
                        )
                        # Update session state if changes are made
                        if not updated_df.equals(editable_df):
                            # Find which years were modified
                            for idx, (_, row) in enumerate(updated_df.iterrows()):
                                year = row['Year']
                                old_value = editable_df.iloc[idx][var]
                                new_value = row[var]
                                value_changed = abs(new_value - old_value) > 1e-6
                                
                                if value_changed:
                                    st.session_state.modified_years_set.add(year)
                                    if year not in st.session_state.modified_variables_by_year:
                                        st.session_state.modified_variables_by_year[year] = set()
                                    st.session_state.modified_variables_by_year[year].add(var)
                            
                            st.session_state.edited_data_all[var] = updated_df.copy()
                            st.rerun()
                    
                    if st.button("↩️ Reset to Original", key=f"reset_{var}"):
                        # Reset this variable
                        original_data = years_data.copy()
                        st.session_state.edited_data_all[var] = original_data
                        
                        # Special handling for Hydro Generation reset: also reset capacity and capacity factors
                        if var == 'Hydro_Generation_TWh':
                            # Reset hydro capacity to original values
                            original_capacity = years_data_all.groupby('Year')['Hydro_Capacity_GW'].mean().reset_index()
                            st.session_state.edited_data_all['Hydro_Capacity_GW'] = original_capacity.copy()
                            
                            # Reset capacity factors to original values
                            st.session_state.hydro_capacity_factors_edited = hydro_capacity_factors.copy()
                        
                        # Remove this variable from all years in the tracking
                        years_to_clean = []
                        for year in st.session_state.modified_variables_by_year:
                            if var in st.session_state.modified_variables_by_year[year]:
                                st.session_state.modified_variables_by_year[year].discard(var)
                                # If no variables left for this year, mark for removal
                                if len(st.session_state.modified_variables_by_year[year]) == 0:
                                    years_to_clean.append(year)
                        
                        # Remove years with no modified variables
                        for year in years_to_clean:
                            del st.session_state.modified_variables_by_year[year]
                            st.session_state.modified_years_set.discard(year)
                        
                        st.rerun()
        else:
            st.warning(f"Variable {var} not found in data")

st.divider()

# Global reset button
col_reset_left, col_reset_center, col_reset_right = st.columns([1, 2, 1])
with col_reset_center:
    if st.button("🔄 Reset All Variables to Baseline Scenario", type="secondary", use_container_width=True, key="global_reset"):
        # Reset all variables to original values
        for var in all_variables:
            if var in years_data_all.columns:
                original_data = years_data_all.groupby('Year')[var].mean().reset_index()
                st.session_state.edited_data_all[var] = original_data.copy()
        
        # Also reset Hydro_Capacity_GW (used in the editor but not in variable_info)
        if 'Hydro_Capacity_GW' in years_data_all.columns:
            original_capacity = years_data_all.groupby('Year')['Hydro_Capacity_GW'].mean().reset_index()
            st.session_state.edited_data_all['Hydro_Capacity_GW'] = original_capacity.copy()
        
        # Reset hydro capacity factors to original values
        st.session_state.hydro_capacity_factors_edited = hydro_capacity_factors.copy()
        
        # Clear all modification tracking
        st.session_state.modified_years_set = set()
        st.session_state.modified_variables_by_year = {}
        
        st.success("✅ All variables have been reset to baseline values!")
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)  # Add some space

# CSV Upload Section
st.markdown("### 📁 Upload Custom Scenario CSV")
col_upload_left, col_upload_center, col_upload_right = st.columns([1, 2, 1])

with col_upload_center:
    # # Create template CSV for download
    # template_data = []
    # for year in range(2026, 2041):
    #     row = {'Year': year}
    #     for var in all_variables:
    #         if var in years_data_all.columns:
    #             baseline_value = years_data_all[years_data_all['Year'] == year][var].mean()
    #             row[var] = round(baseline_value, 2)
    #     template_data.append(row)
    
    template_df = pd.DataFrame(data_plot)
    template_df = template_df[['Year'] + all_variables]

    template_csv = template_df.to_csv(index=False)
    
    # Download template button
    st.download_button(
        label="📥 Download CSV Template",
        data=template_csv,
        file_name="scenario_template.csv",
        mime="text/csv",
        help="Download a template CSV with baseline values that you can modify"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your custom scenario CSV file",
        type=['csv'],
        help="Upload a CSV file with your custom scenario values. Must contain 'Year' column and variable columns.",
        key="csv_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded CSV
            uploaded_df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['Year'] + [var for var in all_variables if var in years_data_all.columns]
            missing_cols = [col for col in required_cols if col not in uploaded_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                # Validate years
                valid_years = set(range(2026, 2041))
                uploaded_years = set(uploaded_df['Year'].unique())
                
                if not uploaded_years.issubset(valid_years):
                    invalid_years = uploaded_years - valid_years
                    st.warning(f"⚠️ Invalid years found: {sorted(invalid_years)}. Only years 2026-2040 will be used.")
                
                # Filter to valid years
                uploaded_df = uploaded_df[uploaded_df['Year'].isin(valid_years)]
                
                if len(uploaded_df) == 0:
                    st.error("❌ No valid years (2026-2040) found in the uploaded file.")
                else:
                    # Show preview of uploaded data
                    st.success(f"✅ Successfully loaded {len(uploaded_df)} rows from CSV!")
                    with st.expander("Preview uploaded data"):
                        st.dataframe(uploaded_df, use_container_width=True)
                    
                    # Apply uploaded data to session state
                    if st.button("🔄 Apply CSV Data", type="secondary", use_container_width=True, key="apply_csv"):
                        years_modified = set()
                        
                        for _, row in uploaded_df.iterrows():
                            year = row['Year']
                            
                            for var in all_variables:
                                if var in uploaded_df.columns and var in st.session_state.edited_data_all:
                                    # Get baseline value for comparison
                                    baseline_value = years_data_all[years_data_all['Year'] == year][var].mean()
                                    uploaded_value = row[var]
                                    
                                    # Update the value in session state
                                    mask = st.session_state.edited_data_all[var]['Year'] == year
                                    st.session_state.edited_data_all[var].loc[mask, var] = uploaded_value
                                    
                                    # Track modifications if different from baseline
                                    if abs(uploaded_value - baseline_value) > 1e-6:
                                        years_modified.add(year)
                                        if year not in st.session_state.modified_variables_by_year:
                                            st.session_state.modified_variables_by_year[year] = set()
                                        st.session_state.modified_variables_by_year[year].add(var)
                                        
                                        # Special handling for Hydro Capacity
                                        if var == 'Hydro_Capacity_GW':
                                            # Get capacity factor for this year
                                            capacity_factor_row = hydro_capacity_factors[hydro_capacity_factors['Year'] == year]
                                            if not capacity_factor_row.empty:
                                                capacity_factor = capacity_factor_row['Capacity_factor_hydro'].iloc[0]
                                                
                                                # Get baseline values
                                                baseline_hydro_generation = years_data_all[years_data_all['Year'] == year]['Hydro_Generation_TWh'].iloc[0]
                                                
                                                # Calculate new hydro generation
                                                new_generation = calculate_hydro_generation(uploaded_value, capacity_factor)
                                                
                                                # Update hydro generation in session state
                                                if 'Hydro_Generation_TWh' not in st.session_state.edited_data_all:
                                                    generation_data = years_data_all.groupby('Year')['Hydro_Generation_TWh'].mean().reset_index()
                                                    st.session_state.edited_data_all['Hydro_Generation_TWh'] = generation_data.copy()
                                                
                                                generation_mask = st.session_state.edited_data_all['Hydro_Generation_TWh']['Year'] == year
                                                st.session_state.edited_data_all['Hydro_Generation_TWh'].loc[generation_mask, 'Hydro_Generation_TWh'] = new_generation
                                                
                                                # Mark generation as modified
                                                years_modified.add(year)
                                                if year not in st.session_state.modified_variables_by_year:
                                                    st.session_state.modified_variables_by_year[year] = set()
                                                st.session_state.modified_variables_by_year[year].add('Hydro_Generation_TWh')
                        
                        # Update modified years set
                        st.session_state.modified_years_set.update(years_modified)
                        
                        st.success(f"✅ CSV data applied successfully! Modified {len(years_modified)} years.")
                        st.rerun()
                        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

st.markdown("<br>", unsafe_allow_html=True)  # Add some space
compute_button = st.button("🚀 Compute Scenario", type="primary", use_container_width=True)

if compute_button:
    modified_data_plot = data_plot.copy()
    total_modifications = 0
    for var in all_variables:
        if var in st.session_state.edited_data_all:
            # Skip Hydro_Capacity_GW as it's not a direct model input
            # Its effect is captured via the derived Hydro_Generation_TWh input
            if var == 'Hydro_Capacity_GW':
                continue
                
            for _, row in st.session_state.edited_data_all[var].iterrows():
                mask = modified_data_plot['Year'] == row['Year']
                if mask.any():
                    modified_data_plot.loc[mask, var] = row[var]
                    total_modifications += 1
    st.session_state.modified_data_plot_all = modified_data_plot
    st.markdown("<h2 style='text-align: center; font-size: 2.2em;'>🎯 Results</h2>", unsafe_allow_html=True)
    years = list(range(2026, 2041))
    baseline_baseload = [round(data_plot[data_plot['Year'] == y]['Baseload_Price_EUR_MWh'].values[0], 2) for y in years]
    baseline_pv = [round(data_plot[data_plot['Year'] == y]['PV_Captured_Price_EUR_MWh'].values[0], 2) for y in years]
    #[simulator_baseload.baseline_data[simulator_baseload.baseline_data['Year'] == y]['Baseload_Price_EUR_MWh'].values[0] for y in years]
    #baseline_pv = [simulator_pv.baseline_data[simulator_pv.baseline_data['Year'] == y]['PV_Captured_Price_EUR_MWh'].values[0] for y in years]
    scenario_baseload = baseline_baseload.copy()
    scenario_pv = baseline_pv.copy()
    ci_base_lower = [None] * len(years)
    ci_base_upper = [None] * len(years)
    ci_pv_lower = [None] * len(years)
    ci_pv_upper = [None] * len(years)
    for i, y in enumerate(years):
        # Decide whether to recompute this year based on tracked modifications
        modified = y in st.session_state.modified_years_set
        if modified:
            # Create scenario input using only NUM_VARS (model variables)
            scenario_input = {}
            for var in NUM_VARS:
                if var in st.session_state.edited_data_all:
                    scenario_input[var] = st.session_state.edited_data_all[var][st.session_state.edited_data_all[var]['Year'] == y][var].values[0]
                else:
                    # Use baseline value if not modified
                    scenario_input[var] = years_data_all[years_data_all['Year'] == y][var].mean()
            
            res_base = simulator_baseload.simulate_scenario({y: scenario_input})[y]
            scenario_baseload[i] = round(res_base['scenario']['predicted_corrected'], 2)
            ci_base = res_base['results']['confidence_interval']
            ci_base_lower[i] = round(ci_base['lower_bound'], 2)
            ci_base_upper[i] = round(ci_base['upper_bound'], 2)
            res_pv = simulator_pv.simulate_scenario({y: scenario_input})[y]
            scenario_pv[i] = round(res_pv['scenario']['predicted_corrected'], 2)
            ci_pv = res_pv['results']['confidence_interval']
            ci_pv_lower[i] = round(ci_pv['lower_bound'], 2)
            ci_pv_upper[i] = round(ci_pv['upper_bound'], 2)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align: center; margin-bottom: -10px;'>Baseload Price (€/MWh)</h3>", unsafe_allow_html=True)
        fig_base = go.Figure()
        fig_base.add_trace(go.Scatter(
            x=years,
            y=scenario_baseload,
            mode='lines+markers',
            name='Scenario',
            line=dict(color='#1f77b4'),
            marker=dict(size=10, color='#1f77b4'),
            hovertemplate='<b>Year:</b> %{x}<br><b>Scenario:</b> %{y:.2f}<extra></extra>'
        ))
        fig_base.add_trace(go.Scatter(
            x=years,
            y=baseline_baseload,
            mode='lines+markers',
            name='Baseline',
            line=dict(color='lightgray', dash='dash'),
            marker=dict(size=8, color='lightgray'),
            hovertemplate='<b>Year:</b> %{x}<br><b>Baseline:</b> %{y:.2f}<extra></extra>'
        ))
        fig_base.update_layout(xaxis_title='Year', yaxis_title='Baseload Price (€/MWh)', height=500, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig_base)
    with col2:
        st.markdown("<h3 style='text-align: center; margin-bottom: -10px;'>PV Captured Price (€/MWh)</h3>", unsafe_allow_html=True)
        
        # Calculate cannibalization for scenario values using correct formula: 1 - pv/baseload
        cannibalization_scenario = [round((1 - pv/base)*100, 2) if base > 0 else 0 for pv, base in zip(scenario_pv, scenario_baseload)]
        
        fig_pv = go.Figure()
        
        # Add scenario line first (bottom layer)
        fig_pv.add_trace(go.Scatter(
            x=years,
            y=scenario_pv,
            mode='lines+markers',
            name='Scenario',
            line=dict(color='#e67e22'),
            marker=dict(size=10, color='#e67e22'),
            hovertemplate='<b>Year:</b> %{x}<br><b>Scenario:</b> %{y:.2f} €/MWh<br><b>Cannibalization:</b> %{customdata:.1f}%<extra></extra>',
            customdata=cannibalization_scenario
        ))
        
        # Add baseline line second (on top of scenario line)
        fig_pv.add_trace(go.Scatter(
            x=years,
            y=baseline_pv,
            mode='lines+markers',
            name='Baseline',
            line=dict(color='lightgray', dash='dash'),
            marker=dict(size=8, color='lightgray'),
            hovertemplate='<b>Year:</b> %{x}<br><b>Baseline:</b> %{y:.2f} €/MWh<extra></extra>'
        ))
        
        # Add invisible trace just for legend to show cannibalization info
        fig_pv.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name='% = Cannibalization',
            marker=dict(size=8, color='#e67e22', symbol='square'),
            #showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add cannibalization labels as annotations (top layer - always on top)
        annotations = []
        for i, (year, pv_price, cann) in enumerate(zip(years, scenario_pv, cannibalization_scenario)):
            annotations.append(
                dict(
                    x=year + 0.6,
                    y=pv_price + 1.9,  # Small offset above the point
                    text=f"{cann:.1f}",
                    showarrow=False,
                    font=dict(size=10, color='#e67e22', family='Arial Bold')
                    # Add semi-transparent white background for readability over lines
                    #bgcolor="rgba(255, 255, 255, 0.8)",
                    #bordercolor="rgba(230, 126, 34, 0.3)",
                    #borderwidth=1,
                    #borderpad=2
                )
            )
               
        fig_pv.update_layout(
            xaxis_title='Year', 
            yaxis_title='PV Captured Price (€/MWh)', 
            height=500, 
            showlegend=True, 
            hovermode='x unified',
            annotations=annotations
        )
        st.plotly_chart(fig_pv)
    st.markdown("<h4 style='text-align: center;'>📊 Scenario Summary & Impacts</h4>", unsafe_allow_html=True)
    
    # Get years that have been modified (tracked when user makes changes)
    modified_years = sorted(list(st.session_state.modified_years_set))
    
    # Only show years that have been modified
    if not modified_years:
        st.info("🔄 No modifications made yet. Edit variables in the tabs above to see scenario impacts.")
    else:        
        # Show metrics only for modified years
        for i, y in enumerate(years):
            if y not in modified_years:
                continue
            baseline_val = baseline_baseload[i]
            scenario_val = scenario_baseload[i]
            abs_change = scenario_val - baseline_val
            pct_change = (abs_change / baseline_val * 100) if baseline_val != 0 else 0
            baseline_val_pv = baseline_pv[i]
            scenario_val_pv = scenario_pv[i]
            abs_change_pv = scenario_val_pv - baseline_val_pv
            pct_change_pv = (abs_change_pv / baseline_val_pv * 100) if baseline_val_pv != 0 else 0
            ci_base = (ci_base_lower[i], ci_base_upper[i]) if ci_base_lower[i] is not None and ci_base_upper[i] is not None else None
            ci_pv = (ci_pv_lower[i], ci_pv_upper[i]) if ci_pv_lower[i] is not None and ci_pv_upper[i] is not None else None
            cols = st.columns(2)
            with cols[0]:
                st.metric(
                    label=f"Baseload {y}",
                    value=f"{scenario_val:.2f} €/MWh",
                    delta=f"{abs_change:+.2f} €/MWh ({pct_change:+.1f}%)"
                )
                if ci_base:
                    st.markdown(f"<div style='background-color: #eaf6ff; padding: 4px 8px; border-radius: 10px; text-align: center; margin-top: 5px; font-size: 1.1em;'>95% CI: [{ci_base[0]:.2f}, {ci_base[1]:.2f}]</div>", unsafe_allow_html=True)
            with cols[1]:
                st.metric(
                    label=f"PV Captured {y}",
                    value=f"{scenario_val_pv:.2f} €/MWh",
                    delta=f"{abs_change_pv:+.2f} €/MWh ({pct_change_pv:+.1f}%)"
                )
                if ci_pv:
                    st.markdown(f"<div style='background-color: #fff6e9; padding: 4px 8px; border-radius: 10px; text-align: center; margin-top: 5px; font-size: 1.1em;'>95% CI: [{ci_pv[0]:.2f}, {ci_pv[1]:.2f}]</div>", unsafe_allow_html=True)
            
            # Show impact analysis for this year
            baseload_coefficients = MODEL_COEFFICIENTS["Baseload_Price_EUR_MWh"]
            pv_coefficients = MODEL_COEFFICIENTS["PV_Captured_Price_EUR_MWh"]
            impact_data = []
            # Only show variables that were actually modified for this specific year
            variables_modified_this_year = st.session_state.modified_variables_by_year.get(y, set())
            
            for var in variables_modified_this_year:
                if var in st.session_state.edited_data_all and var in years_data_all.columns:
                    if var not in variable_info:
                        continue  # Skip variables not in variable_info (e.g., internally calculated ones)
                    # Get original value
                    original_data = years_data_all.groupby('Year')[var].mean().reset_index()
                    orig_val = original_data[original_data['Year'] == y][var].iloc[0]
                    mod_val = st.session_state.edited_data_all[var][st.session_state.edited_data_all[var]['Year'] == y][var].iloc[0]
                    input_change = mod_val - orig_val
                    # We know this variable was modified, so calculate impact
                    if abs(input_change) > 1e-6:
                        base_impact = baseload_coefficients.get(var, 0) * input_change
                        pv_impact = pv_coefficients.get(var, 0) * input_change
                        impact_data.append({
                            'Variable': variable_info[var]['description'] + " [" + variable_info[var]['unit'] + "]",
                            'Input Change': f"{input_change:+.2f}",
                            'Baseload Impact (€/MWh)': f"{base_impact:+.2f}",
                            'PV Captured Impact (€/MWh)': f"{pv_impact:+.2f}"
                        })
            
            if impact_data:  # Only show table if there are impacts to display
                impact_df = pd.DataFrame(impact_data)
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(impact_df, use_container_width=True, hide_index=True)
    
    # 📁 DOWNLOAD RESULTS SECTION
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📁 Download Results")
    
    # Create comprehensive results dataframe with clear scenario separation
    results_data = []
    
    # Calculate cannibalization for baseline
    cannibalization_baseline = [round((1 - pv/base)*100, 2) if base > 0 else 0 for pv, base in zip(baseline_pv, baseline_baseload)]
    
    # Define column structure with clear separation (using EUR instead of € for better CSV compatibility)
    columns_structure = {
        'Year': 'Year',
        'Modified_Year': 'Modified',
        # BASELINE SCENARIO columns
        'Baseline_Baseload_Price_EUR_MWh': 'Baseload Price (EUR/MWh)',
        'Baseline_PV_Captured_Price_EUR_MWh': 'PV Captured Price (EUR/MWh)', 
        'Baseline_Cannibalization_Percent': 'Cannibalization (%)',
        # NEW SCENARIO columns
        'Scenario_Baseload_Price_EUR_MWh': 'Baseload Price (EUR/MWh)',
        'Scenario_PV_Captured_Price_EUR_MWh': 'PV Captured Price (EUR/MWh)',
        'Scenario_Cannibalization_Percent': 'Cannibalization (%)',
        # # CHANGES columns
        # 'Change_Baseload_Price_EUR_MWh': 'Delta Baseload (EUR/MWh)',
        # 'Change_PV_Captured_Price_EUR_MWh': 'Delta PV Captured (EUR/MWh)',
        # 'Change_Cannibalization_Percent': 'Delta Cannibalization (%)'
    }
    
    for i, year in enumerate(years):
        row = {
            'Year': year,
            'Modified_Year': 'Yes' if year in modified_years else 'No',
            # BASELINE SCENARIO
            'Baseline_Baseload_Price_EUR_MWh': baseline_baseload[i],
            'Baseline_PV_Captured_Price_EUR_MWh': baseline_pv[i],
            'Baseline_Cannibalization_Percent': cannibalization_baseline[i],
            # NEW SCENARIO
            'Scenario_Baseload_Price_EUR_MWh': scenario_baseload[i],
            'Scenario_PV_Captured_Price_EUR_MWh': scenario_pv[i],
            'Scenario_Cannibalization_Percent': cannibalization_scenario[i],
            # # CHANGES
            # 'Change_Baseload_Price_EUR_MWh': round(scenario_baseload[i] - baseline_baseload[i], 2),
            # 'Change_PV_Captured_Price_EUR_MWh': round(scenario_pv[i] - baseline_pv[i], 2),
            # 'Change_Cannibalization_Percent': round(cannibalization_scenario[i] - cannibalization_baseline[i], 2)
        }
        
        # # Add confidence intervals if available
        # if ci_base_lower[i] is not None and ci_base_upper[i] is not None:
        #     row['Baseline_CI_Lower'] = ci_base_lower[i]
        #     row['Baseline_CI_Upper'] = ci_base_upper[i]
        #     row['Scenario_CI_Lower'] = ci_base_lower[i]  # Use same for now
        #     row['Scenario_CI_Upper'] = ci_base_upper[i]
        
        # if ci_pv_lower[i] is not None and ci_pv_upper[i] is not None:
        #     row['Baseline_PV_CI_Lower'] = ci_pv_lower[i]
        #     row['Baseline_PV_CI_Upper'] = ci_pv_upper[i]
        #     row['Scenario_PV_CI_Lower'] = ci_pv_lower[i]
        #     row['Scenario_PV_CI_Upper'] = ci_pv_upper[i]
        
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    
    # Create CSV with custom header to distinguish scenarios
    import io
    import datetime
    
    # Create the CSV content with scenario headers
    csv_buffer = io.StringIO()
    
    # Write scenario identification headers
    scenario_headers = [
        "",                    # Year column
        "",                    # Modified column  
        "BASELINE SCENARIO",   # Baseline Baseload
        "",                    # Baseline PV
        "",                    # Baseline Cannibalization
        "NEW SCENARIO",        # Scenario Baseload
        "",                    # Scenario PV  
        ""                     # Change Cannibalization
    ]
    
    # # Add CI scenario headers if confidence intervals are present
    # if any(x is not None for x in ci_base_lower):
    #     scenario_headers.extend(["BASELINE CI", "", "NEW SCENARIO CI", ""])
    # if any(x is not None for x in ci_pv_lower):
    #     scenario_headers.extend(["BASELINE PV CI", "", "NEW SCENARIO PV CI", ""])
    
    # Write the scenario header row
    csv_buffer.write(",".join(scenario_headers) + "\n")
    
    # Rename columns using the columns_structure mapping for display
    results_df_display = results_df.rename(columns=columns_structure)
    
    # Write the actual dataframe with renamed columns (without index, with column headers)
    results_df_display.to_csv(csv_buffer, index=False, header=True)
    
    # Get CSV content and encode it properly for special characters
    csv_content = csv_buffer.getvalue()
    # Add BOM for better Excel compatibility with special characters
    csv_results = '\ufeff' + csv_content
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"long_term_scenario_results_{timestamp}.csv"
    
    st.download_button(
        label="📥 Download Results CSV",
        data=csv_results,
        file_name=filename,
        mime="text/csv",
        help="Download complete results with baseline, scenario values, and cannibalization data",
        use_container_width=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)







