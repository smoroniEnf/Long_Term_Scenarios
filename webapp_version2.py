import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from FERidge_classes import ConstrainedFEModel, ConstrainedFEScenarioSimulator

TARGETS = ["Baseload_Price_EUR_MWh", "PV_Captured_Price_EUR_MWh"]

NUM_VARS = [
    "Carbon_price_EU_ETS_EUR_tonne",
    "Gas_price_PSV_EUR_MWh",
    "Electricity_demand_TWh",
    "PV_Generation_TWh",
    "Wind_Generation_TWh",
    "Hydro_Generation_TWh",
    "Battery_storage_Capacity_GW",
]

# Additional capacity variables that can be modified in the UI
CAPACITY_VARS = [
    "PV_Capacity_GW",
    "Wind_Capacity_GW", 
    "Hydro_Capacity_GW",
    "Battery_storage_Capacity_GW"
]
CAPACITY_HYDRO = [
    21.5,
    21.511,
    21.540,
    21.561,
    21.579,
    21.584,
    21.589,
    21.590,
    21.616,
    21.634,
    21.650,
    21.656,
    21.685,
    21.710,
    21.717
 ]

# Model coefficients (fallback values if model coefficients are not accessible)
MODEL_COEFFICIENTS = {
    "Baseload_Price_EUR_MWh": {
        "Carbon_price_EU_ETS_EUR_tonne": 0.313388,
        "Gas_price_PSV_EUR_MWh": 1.37184,
        "Electricity_demand_TWh": 0.29128,
        "PV_Generation_TWh": -0.574711,
        "Wind_Generation_TWh": -0.372100,
        "Hydro_Generation_TWh": -0.336500,
        "Battery_storage_Capacity_GW": -0.253251
    },
    "PV_Captured_Price_EUR_MWh": {
        "Carbon_price_EU_ETS_EUR_tonne": 0.300174,
        "Gas_price_PSV_EUR_MWh": 0.495745,
        "Electricity_demand_TWh": 0.335653,
        "PV_Generation_TWh": -0.880342,
        "Wind_Generation_TWh": -0.264800,
        "Hydro_Generation_TWh": -0.16429,
        "Battery_storage_Capacity_GW": 0.695700
    }
}

# path = os.getcwd() + "\\Enfinity Global\\Energy Management - Documents\\General\\05_Market Intelligence\\Data Team\\Long_Term_Scenarios\\"

YIELDS_FACTORS = pd.read_csv("yields_for_capacities.csv")



# Configure page
st.set_page_config(
    page_title = "Long Term Energy Price Scenarios",
    page_icon = "⚡",
    layout = "wide"
    )

# Title
st.title("Long Term Energy Price Scenarios ⚡")
st.markdown("---")

# Load models and data function
@st.cache_data
def load_model_and_data(scenario):
    "Load the scenario data  and the trained model"

    try:

        baseload_model_file = f'baseload_{scenario.lower()}_model_2.pkl'
        pv_model_file = f'pv_{scenario.lower()}_model_2.pkl'
        data_file = f'processed_{scenario.lower()}_scenario_data_2.csv'

        # Load models

        with open(baseload_model_file, 'rb') as f:
            baseload_model = pickle.load(f)
        with open(pv_model_file, 'rb') as f:
            pv_model = pickle.load(f)

        # Load data
        data = pd.read_csv(data_file)
        data["Year"] = data["Year"].astype(int)
        data = data.loc[data['Year'] >= 2026].reset_index(drop=True)

        data["Release"] = data["Release"].astype(str)

        # Initialize simulator
        simulator_baseload = ConstrainedFEScenarioSimulator(
            fitted_model=baseload_model,
            training_data=data
            )
        simulator_pv = ConstrainedFEScenarioSimulator(
            fitted_model=pv_model,
            training_data=data
            )
        simulator_baseload.baseline_capacity['Hydro_Capacity_GW'] = CAPACITY_HYDRO
        simulator_pv.baseline_capacity['Hydro_Capacity_GW'] = CAPACITY_HYDRO

        return simulator_baseload, simulator_pv, baseload_model, pv_model, data
    
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None, None, None, None

# Use central scenario by default
scenario = "Central"

# Load model and data
simulator_baseload, simulator_pv, baseload_model, pv_model, data = load_model_and_data(scenario)

if simulator_baseload is None or data is None or simulator_pv is None:
    st.error("Failed to load model or data. Please check if the files exist.")
    st.stop()

# Year Selection in main page
st.header("🎯 Year Selection")
# Year selection
available_years = sorted(data['Year'].unique())
selected_year = st.selectbox(
    "Select Year for Analysis",
    available_years,
    help="Choose the year for prediction",
    index=0
)
st.markdown("---")

# Main interface - Year Selection moved to main page

baseline_df = simulator_baseload.baseline_data
baseline_df['PV_Captured_Price_EUR_MWh'] = simulator_pv.baseline_data['PV_Captured_Price_EUR_MWh']

# Get baseline data for selected year
baseline_row = baseline_df[baseline_df['Year'] == selected_year].iloc[0]

# Get capacity data if available
baseline_capacity_row = None
if hasattr(simulator_baseload, 'baseline_capacity') and not simulator_baseload.baseline_capacity.empty:
    capacity_data = simulator_baseload.baseline_capacity[simulator_baseload.baseline_capacity['Year'] == selected_year]
    if not capacity_data.empty:
        baseline_capacity_row = capacity_data.iloc[0]

feature_cols = NUM_VARS

# Input section
st.header(f"🔧 Values for Year {int(selected_year)}")

# Function to format variable names for display
def format_variable_name(col):
    """Format variable names according to specific requirements"""
    
    # Capacity variables: remove "Capacity" and add [GW]
    if "Capacity_GW" in col:
        if col == "PV_Capacity_GW":
            return "PV [GW]"
        elif col == "Wind_Capacity_GW":
            return "Wind [GW]"
        elif col == "Hydro_Capacity_GW":
            return "Hydro [GW]"
        elif col == "Battery_storage_Capacity_GW":
            return "Battery Storage [GW]"
    
    # Price variables: specific formatting
    elif "Carbon_price_EU_ETS" in col:
        return "EUA Price [€/tonne CO₂]"
    elif "Gas_price_PSV" in col:
        return "PSV Gas Price [€/MWh]"
    
    # Demand variables: add [TWh]
    elif "Electricity_demand_TWh" in col:
        return "Electricity Demand [TWh]"
    
    # Generation variables: add [TWh] 
    elif "Generation_TWh" in col:
        if col == "PV_Generation_TWh":
            return "PV Generation [TWh]"
        elif col == "Wind_Generation_TWh":
            return "Wind Generation [TWh]"
        elif col == "Hydro_Generation_TWh":
            return "Hydro Generation [TWh]"
    
    # Fallback: replace underscores with spaces
    return col.replace('_', ' ')

# Create input fields for each regressor
modified_values = {}

# Group regressors by category for better organization
# Get available capacity columns from simulator
available_capacity_cols = []
if baseline_capacity_row is not None:
    available_capacity_cols = [col for col in CAPACITY_VARS if col in baseline_capacity_row.index] + ['Battery_storage_Capacity_GW']

price_cols = [col for col in NUM_VARS if ('price' in col or 'Price' in col) and col in baseline_row.index]
demand_cols = [col for col in NUM_VARS if ('demand' in col or 'Demand' in col) and col in baseline_row.index]

# Display regressors by category
categories = [
    ("🔋 Capacity Variables", available_capacity_cols),
    ("💰 Price Variables", price_cols),
    ("📈 Demand Variable", demand_cols)
]

for category_name, cols in categories:
        if cols:
            st.subheader(category_name)
            
            # Create columns for better layout
            num_cols = min(2, len(cols))
            sub_cols = st.columns(num_cols)
            
            for i, col in enumerate(cols):
                # Determine the data source for baseline values
                if col in available_capacity_cols and baseline_capacity_row is not None and col != 'Battery_storage_Capacity_GW':
                    # Use capacity data
                    baseline_value = baseline_capacity_row[col]
                    data_source = "capacity"
                elif col in baseline_row.index:
                    # Use baseline data
                    baseline_value = baseline_row[col]
                    data_source = "baseline"
                else:
                    continue  # Skip if column not available
                
                with sub_cols[i % num_cols]:
                    # Format the variable name
                    formatted_label = format_variable_name(col)
                    
                    # Create input field with larger label
                    st.markdown(f"<div style='font-size: 18px; font-weight: 500; margin-bottom: 5px; color: #1f77b4;'>{formatted_label}</div>", unsafe_allow_html=True)
                    
                    new_value = st.number_input(
                        "",  # Empty label since we use custom HTML above
                        value=float(baseline_value),
                        step=0.1 if abs(baseline_value) < 10 else 1.0,
                        format="%.2f",
                        key=f"input_{col}_{selected_year}",
                        label_visibility="collapsed"
                    )
                    
                    modified_values[col] = new_value
                    
                    # Show change if modified
                    if abs(new_value - baseline_value) > 1e-6:
                        change_pct = ((new_value - baseline_value) / baseline_value) * 100 if baseline_value != 0 else np.inf
                        st.caption(f"📊 Change: {change_pct:+.1f}%")

# Add compute scenario button
st.divider()
compute_button = st.button("🚀 Compute Scenario", type="primary", use_container_width=True)

# Results section - only show when button is pressed
if compute_button:
    st.header("🎯 Results")
    
    # Show baseline values with centered title
    st.markdown("<h3 style='text-align: center;'>📊 Baseline Scenario</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    
    cols = st.columns(len(TARGETS))
    for i, target in enumerate(TARGETS):
        baseline_target = baseline_row[target]
        with cols[i]:
            st.metric(
                label=f"{target.replace('_', ' ')}",
                value=f"{baseline_target:.2f} €/MWh"
            )
    
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add more spacing between sections
    
    # Check if any values were modified
    # Check both baseline and capacity values
    any_modified = False
    baseline_cols = [col for col in NUM_VARS if col in modified_values and col in baseline_row.index]
    capacity_cols = [col for col in available_capacity_cols if col in modified_values and col != 'Battery_storage_Capacity_GW']

    
    # Check baseline modifications
    for col in baseline_cols:
        if abs(modified_values[col] - baseline_row[col]) > 1e-6:
            any_modified = True
            break
    
    # Check capacity modifications  
    if not any_modified and baseline_capacity_row is not None:
        for col in capacity_cols:
            if abs(modified_values[col] - baseline_capacity_row[col]) > 1e-6:
                any_modified = True
                break

    if any_modified:
        
        # Centered title for predicted values
        st.markdown("<h3 style='text-align: center;'>🔮 New Scenario</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        
        # Create scenario dictionary for simulator
        changed_variables = {}
        
        # Add changed baseline variables
        for col in baseline_cols:
            if abs(modified_values[col] - baseline_row[col]) > 1e-6:
                changed_variables[col] = modified_values[col]
        
        # Add changed capacity variables
        if baseline_capacity_row is not None:
            for col in capacity_cols:
                if abs(modified_values[col] - baseline_capacity_row[col]) > 1e-6:
                    changed_variables[col] = modified_values[col]
        
        # Use simulator's simulate_scenario function
        try:
            scenario_dict = {selected_year: changed_variables}
            results_baseload = simulator_baseload.simulate_scenario(scenario_dict)
            results_pv = simulator_pv.simulate_scenario(scenario_dict)

            # Display predictions in columns
            cols = st.columns(len(TARGETS))
            for i, (target, results) in enumerate(zip(TARGETS, [results_baseload, results_pv])):
                # Extract results for the selected year
                year_result = results[selected_year]
                
                prediction = year_result['scenario']['predicted_corrected']
                baseline_y_true = year_result['baseline']['actual_value']
                absolute_change = year_result['results']['absolute_change']
                percentage_change = year_result['results']['percentage_change']
                confidence_interval = year_result['results']['confidence_interval']
                
                # Display prediction
                with cols[i]:
                    st.metric(
                        label=f"New {target.replace('_', ' ')}",
                        value=f"{prediction:.2f} €/MWh",
                        delta=f"{absolute_change:+.2f} €/MWh ({percentage_change:+.1f}%)"
                    )
                    
                    # Display confidence interval with better styling
                    if confidence_interval and len(confidence_interval) == 2:
                        lower_bound = confidence_interval['lower_bound'] if isinstance(confidence_interval, list) else confidence_interval.get('lower_bound', 0)
                        upper_bound = confidence_interval['upper_bound'] if isinstance(confidence_interval, list) else confidence_interval.get('upper_bound', 0)
                        
                        st.markdown(f"<div style='background-color: #f0f2f6; padding: 4px 8px; border-radius: 10px; text-align: center; margin-top: 5px; font-size: 1.5em;'>95% CI: [{lower_bound:.1f}, {upper_bound:.1f}]</div>", unsafe_allow_html=True)
                        

            try:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                st.markdown("<h3 style='text-align: center;'>Comprehensive Impact Analysis</h3>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing

                # Get coefficients for both targets
                baseload_coefficients = MODEL_COEFFICIENTS["Baseload_Price_EUR_MWh"].copy()
                pv_coefficients = MODEL_COEFFICIENTS["PV_Captured_Price_EUR_MWh"].copy()
                                
                # Calculate impacts for all changed variables
                unified_impact_data = []
                total_baseload_impact = 0
                total_pv_impact = 0
                
                for col in baseline_cols + capacity_cols:
                    if col in changed_variables:
                        # Calculate the change in input
                        if col in baseline_cols:
                            baseline_val = baseline_row[col]

                        else:
                            baseline_val = baseline_capacity_row[col] if baseline_capacity_row is not None else 0
                        
                        new_val = changed_variables[col]
                        input_change = new_val - baseline_val
                        
                        # Calculate impacts for both targets using correct coefficients
                        baseload_impact = 0
                        pv_impact = 0
                        
                        # Use appropriate coefficient (capacity or generation)
                        if col in baseload_coefficients:
                            baseload_impact = baseload_coefficients[col] * input_change
                            total_baseload_impact += baseload_impact
                        else: 
                            prefix = col.split('_')[0]  # Get the prefix ('PV', 'Wind', 'Hydro')
                            gen_var = f"{prefix}_Generation_TWh"
                            if gen_var in baseload_coefficients:
                                baseload_impact = baseload_coefficients[gen_var] * YIELDS_FACTORS.loc[YIELDS_FACTORS['Year'] == selected_year, prefix].iloc[0] * input_change 
                                total_baseload_impact += baseload_impact
                        
                        if col in pv_coefficients:
                            pv_impact = pv_coefficients[col] * input_change
                            total_pv_impact += pv_impact
                        else:                         
                            prefix = col.split('_')[0]  # Get the prefix ('PV', 'Wind', 'Hydro')
                            gen_var = f"{prefix}_Generation_TWh"
                            if gen_var in pv_coefficients:
                                pv_impact = pv_coefficients[gen_var] * YIELDS_FACTORS.loc[YIELDS_FACTORS['Year'] == selected_year, prefix].iloc[0] * input_change 
                                total_pv_impact += pv_impact

                        
                        # Add capacity factor info for capacity variables
                        variable_display = format_variable_name(col)
                        
                        # Format impacts with direction indicators
                        baseload_formatted = f"📈 {baseload_impact:+.2f}" if baseload_impact > 0 else f"📉 {baseload_impact:+.2f}" if baseload_impact < 0 else "➖ 0.00"
                        pv_formatted = f"📈 {pv_impact:+.2f}" if pv_impact > 0 else f"📉 {pv_impact:+.2f}" if pv_impact < 0 else "➖ 0.00"
                        
                        unified_impact_data.append({
                            'Variable': variable_display,
                            'Input Change': f"{input_change:+.2f}",
                            'Baseload Impact (€/MWh)': baseload_formatted,
                            'PV Captured Impact (€/MWh)': pv_formatted,
                            'Total Magnitude': abs(baseload_impact) + abs(pv_impact)  # For sorting
                        })
                
                if unified_impact_data:
                    # Sort by total magnitude (descending)
                    unified_impact_data_sorted = sorted(unified_impact_data, key=lambda x: x['Total Magnitude'], reverse=True)
                    
                    # Remove the sorting column
                    for item in unified_impact_data_sorted:
                        del item['Total Magnitude']
                    
                    # Display the unified table
                    impact_df = pd.DataFrame(unified_impact_data_sorted)
                    st.dataframe(impact_df, use_container_width=True, hide_index=True)                   
                        
                else:
                    st.info("No variable impacts to analyze.")
                    
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")
                st.write("Debug info:", str(e))            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.write("Debug info:", str(e))
    else:

        st.info("ℹ️ No changes detected. Modify some input values to see predictions.")
