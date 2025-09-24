# Constrained FE Ridge (Release & Year)

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, List
import pickle
import subprocess
import sys
try:
    import cvxpy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cvxpy"])
    import cvxpy

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


@dataclass
class ConstrainedFEModel:
    target: str
    beta: np.ndarray                 # coefficients for NUM_VARS (p,)
    alpha: np.ndarray                # Release FE (r,) matching D_cols
    gamma: np.ndarray                # Year FE (t,) matching U_cols
    D_cols: List[str]                # dummy column names for Release (drop_first=True)
    U_cols: List[str]                # dummy column names for Year (drop_first=True)
    release_levels: List[str]        # sorted distinct releases
    year_levels: List[str]           # sorted distinct years as str
    baseline_release: str            # baseline (dropped) release
    baseline_year: str               # baseline (dropped) year


def _build_design(df: pd.DataFrame):
    # Categorical levels (sorted for reproducibility)
    release_levels = sorted(df["Release"].astype(str).unique().tolist())
    year_levels = sorted(df["Year"].astype(str).unique().tolist())

    # Dummies with drop_first=True => drop the first (baseline) in sorted order
    D = pd.get_dummies(df["Release"].astype(str), drop_first=True)
    U = pd.get_dummies(df["Year"].astype(str), drop_first=True)

    X = df[NUM_VARS].astype(float).values
    D_cols = D.columns.tolist()
    U_cols = U.columns.tolist()

    baseline_release = release_levels[0]  # dropped column
    baseline_year = year_levels[0]        # dropped column

    return X, D.values, U.values, D_cols, U_cols, release_levels, year_levels, baseline_release, baseline_year


def _objective_and_constraints(y, X, D, U, signs_dict, lb_dict, ub_dict, ridge_lambda, weighted_ridge):
    n = len(y)
    p = X.shape[1]
    r = D.shape[1]
    t = U.shape[1]

    alpha = cp.Variable(r)
    gamma = cp.Variable(t)
    beta  = cp.Variable(p)

    yhat = D @ alpha + U @ gamma + X @ beta

    if weighted_ridge:
        w = np.ones(p)
        for j, name in enumerate(NUM_VARS):
            if name in weighted_ridge:
                w[j] = float(weighted_ridge[name])
        obj = cp.sum_squares(y - yhat) + ridge_lambda * cp.sum_squares(cp.multiply(w, beta))
    else:
        obj = cp.sum_squares(y - yhat) + ridge_lambda * cp.sum_squares(beta)

    cons = []
    # Sign & magnitude constraints on beta
    for j, name in enumerate(NUM_VARS):
        s = signs_dict.get(name, 0)
        if s == +1:
            cons.append(beta[j] >= 0)
        elif s == -1:
            cons.append(beta[j] <= 0)
        if lb_dict and (name in lb_dict):
            cons.append(beta[j] >= float(lb_dict[name]))
        if ub_dict and (name in ub_dict):
            cons.append(beta[j] <= float(ub_dict[name]))

    return alpha, gamma, beta, yhat, obj, cons


def fit_constrained_model(df: pd.DataFrame, target: str, cfg: dict) -> ConstrainedFEModel:
    # Build design
    X, D, U, D_cols, U_cols, release_levels, year_levels, base_rel, base_year = _build_design(df)

    y = df[target].astype(float).values
    alpha, gamma, beta, yhat, obj, cons = _objective_and_constraints(
        y, X, D, U,
        signs_dict=cfg["signs"],
        lb_dict=cfg.get("lb_bounds", {}),
        ub_dict=cfg.get("ub_bounds", {}),
        ridge_lambda=cfg.get("ridge_lambda", 0.5),
        weighted_ridge=cfg.get("weighted_ridge", None),
    )

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000, verbose=False)

    # Extract params
    alpha_v = np.array(alpha.value).reshape(-1)
    gamma_v = np.array(gamma.value).reshape(-1)
    beta_v  = np.array(beta.value).reshape(-1)
    yhat_v  = np.array(yhat.value).reshape(-1)
    # Metrics
    resid = y - yhat_v
    mse = float(np.mean(resid**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(resid)))
    r2 = float(1.0 - np.sum(resid**2) / np.sum((y - y.mean())**2))

    print(f"\n=== Constrained FE Ridge — {target} ===")
    print(f"Status: {prob.status}  Objective: {prob.value:.2f}")
    print(f"R2: {r2:.3f}   RMSE: {rmse:.2f}   MAE: {mae:.2f}")

    coef_tbl = pd.DataFrame({
        "feature": NUM_VARS,
        "coef_constrained_(€/MWh per unit)": beta_v
    })
    #display(coef_tbl)

    return ConstrainedFEModel(
        target=target,
        beta=beta_v,
        alpha=alpha_v,
        gamma=gamma_v,
        D_cols=D_cols,
        U_cols=U_cols,
        release_levels=release_levels,
        year_levels=year_levels,
        baseline_release=base_rel,
        baseline_year=base_year,
    )


def _design_for_new_rows(model: ConstrainedFEModel, new_df: pd.DataFrame):
    """
    Build (X_new, D_new, U_new) aligned with a fitted model.
    Unseen categories (new Release/Year not in training) fall back to baseline (all dummy zeros).
    """
    # X
    X_new = new_df[NUM_VARS].astype(float).values

    # D (Release)
    D_new = np.zeros((len(new_df), len(model.D_cols)))
    # Create columns for known dummies, fill if present:
    rel_series = new_df["Release"].astype(str)
    for j, col in enumerate(model.D_cols):
        # col format e.g., 'Apr25' if get_dummies used that name
        # Build indicator where new release equals this dummy's category
        # Pandas get_dummies names are exactly the category names for simple series
        D_new[:, j] = (rel_series.values == col).astype(float)

    # U (Year)
    U_new = np.zeros((len(new_df), len(model.U_cols)))
    year_series = new_df["Year"].astype(str)
    for j, col in enumerate(model.U_cols):
        U_new[:, j] = (year_series.values == col).astype(float)

    # If a category was unseen → all zeros in that block => baseline category
    return X_new, D_new, U_new


def predict_df(model: ConstrainedFEModel, new_df: pd.DataFrame) -> np.ndarray:
    """
    Predict y for new_df with columns: NUM_VARS + Release + Year
    """
    for col in NUM_VARS + ["Release", "Year"]:
        if col not in new_df.columns:
            raise ValueError(f"Missing column '{col}' in new_df.")

    X_new, D_new, U_new = _design_for_new_rows(model, new_df)
    yhat = D_new @ model.alpha + U_new @ model.gamma + X_new @ model.beta
    return yhat


def what_if(model: ConstrainedFEModel, df_full: pd.DataFrame, year: int, deltas: Dict[str, float],
            baseline_method: str = "latest") -> Dict:
    """
    Baseline row for selected year (latest release or mean), apply deltas to NUM_VARS, predict baseline & scenario.
    """
    sub = df_full[df_full["Year"] == year].copy()
    if sub.empty:
        raise ValueError(f"No data for Year={year}")

    if baseline_method == "latest":
        rel = sub["Release"].astype(str).max()
        base_row = sub[sub["Release"] == rel].iloc[0]
    elif baseline_method.startswith("mean"):
        # mean over releases for that year; use latest release label to anchor dummies
        rel = sub["Release"].astype(str).max()
        base_vals = sub[NUM_VARS].mean()
        base_row = pd.Series({**base_vals.to_dict(), "Release": rel, "Year": year})
    else:
        raise ValueError("Unknown baseline_method")

    base_df = pd.DataFrame([base_row])[NUM_VARS + ["Release", "Year"]]
    base_pred = float(predict_df(model, base_df)[0])

    scen_row = base_row.copy()
    for k, v in deltas.items():
        if k not in NUM_VARS:
            raise ValueError(f"Unknown feature '{k}' in deltas.")
        scen_row[k] = float(scen_row[k]) + v
    scen_df = pd.DataFrame([scen_row])[NUM_VARS + ["Release", "Year"]]
    scen_pred = float(predict_df(model, scen_df)[0])

    # One-at-a-time components
    components = {}
    for k, v in deltas.items():
        tmp = base_row.copy()
        tmp[k] = float(tmp[k]) + v
        tmp_df = pd.DataFrame([tmp])[NUM_VARS + ["Release", "Year"]]
        components[k] = float(predict_df(model, tmp_df)[0]) - base_pred

    return {
        "year": year,
        "baseline_release": str(base_row["Release"]),
        "baseline": base_pred,
        "scenario": scen_pred,
        "delta": scen_pred - base_pred,
        "components": components
    }




class ConstrainedFEScenarioSimulator:
    """
    Scenario Simulator for Constrained FE Models with Residual Correction
    
    This simulator implements a residual correction approach adapted for ConstrainedFEModel:
    1. Baseline scenario = mean of last 4 releases for each year
    2. Calculate residuals = actual_values - baseline_predictions
    3. For scenarios: scenario_predictions + residuals = corrected_predictions
    
    This ensures that:
    - The baseline shows true values (corrected predictions)
    - Scenario variations are applied relative to this corrected baseline
    - Model bias is automatically corrected
    """
    
    def __init__(self, fitted_model: ConstrainedFEModel, training_data: pd.DataFrame, baseline_release=None):
        """
        Initialize the simulator
        
        Parameters:
        -----------
        fitted_model : ConstrainedFEModel
            Fitted constrained FE model
        training_data : DataFrame
            Training dataset with Release column
        baseline_release : str, optional
            Specific release to use as baseline (if None, uses mean of last 4 releases)
        """
        self.model = fitted_model
        self.data = training_data.copy()
        self.baseline_release = baseline_release
        
        self.y_col = fitted_model.target
        self.selected_features = NUM_VARS  # Use the global NUM_VARS
        
        # Create baseline and calculate residuals
        self._create_baseline()
        self._calculate_residuals()
        
        print(f"=== CONSTRAINED FE SCENARIO SIMULATOR ===")
        print(f"Target variable: {self.y_col}")
        print(f"Features: {self.selected_features}")

    def _create_baseline(self):
        """Create baseline scenario from mean of latest releases"""

        if self.baseline_release is not None:
            self.baseline_data = self.data[self.data['Release'] == self.baseline_release].copy()
            self.baseline_years = self.baseline_data['Year'].values
            if self.baseline_data.empty:
                raise ValueError(f"No data found for baseline release: {self.baseline_release}")
        else:
            unique_releases = sorted(self.data['Release'].unique(), key=lambda x: pd.to_datetime(x, format='%b%y'))
            last_releases = unique_releases
            self.baseline_data = self.data[self.data['Release'].isin(last_releases)].copy()
            self.baseline_data = self.baseline_data[self.selected_features + [self.y_col, 'Year']].groupby('Year').mean().reset_index()
            self.baseline_years = self.baseline_data['Year'].values
            
        # Baseline capacity table (for yield calculations)
        capacity_cols = ['Year', 'PV_Capacity_GW', 'Wind_Capacity_GW', 'Hydro_Capacity_GW']
        available_capacity_cols = [col for col in capacity_cols if col in self.data.columns]
        if available_capacity_cols:
            self.baseline_capacity = self.data[available_capacity_cols].groupby('Year').mean().reset_index()
        else:
            self.baseline_capacity = pd.DataFrame({'Year': self.baseline_years})

        # Compute yields if capacity data is available
        self.yields_coefficients = self.compute_yields_coefficients()

        # Lookup for fast access
        self.baseline_lookup = {}
        for i, year in enumerate(self.baseline_years):
            self.baseline_lookup[year] = {
                'actual_value': self.baseline_data[self.y_col].values[i],
                'features': self.baseline_data[self.selected_features].iloc[i].to_dict(),
                'data_idx': i,
                'full_row': self.baseline_data.iloc[i]
            }
   
    def _calculate_residuals(self):
        """Calculate residuals between actual and predicted values on baseline"""
        print(f"\nCalculating residuals for baseline correction...")
        
        # Create prediction dataframe with Release and Year
        # Use the latest release for predictions
        unique_releases = sorted(self.data['Release'].unique(), key=lambda x: pd.to_datetime(x, format='%b%y'))
        latest_release = unique_releases[-1]
        print(latest_release)
        
        baseline_for_prediction = self.baseline_data[self.selected_features + ['Year']].copy()
        baseline_for_prediction['Release'] = latest_release
        
        # Make predictions on baseline using the constrained FE model
        baseline_predictions = predict_df(self.model, baseline_for_prediction)
        actual_values = self.baseline_data[self.y_col].values
        
        # Calculate residuals for each year
        self.residuals = actual_values - baseline_predictions
        
        # Store residuals by year
        self.year_residuals = {}
        for i, year in enumerate(self.baseline_data['Year']):
            self.year_residuals[year] = self.residuals[i]
            self.baseline_lookup[year]['residual'] = self.residuals[i]
            self.baseline_lookup[year]['predicted_value'] = baseline_predictions[i]
    
    def compute_yields_coefficients(self):
        """Compute yield coefficients (Generation/Capacity ratios)"""
        yields = pd.DataFrame()
        yields['Year'] = self.baseline_data['Year']
        
        for feature in ['PV_Generation_TWh', 'Wind_Generation_TWh', 'Hydro_Generation_TWh']:
            capacity_col = feature.replace('Generation_TWh', 'Capacity_GW')
            if (capacity_col in self.baseline_capacity.columns and 
                feature in self.baseline_data.columns):
                name = capacity_col.split('_')[0]
                capacity_data = self.baseline_capacity.set_index('Year')[capacity_col]
                generation_data = self.baseline_data.set_index('Year')[feature]
                yields[name] = (generation_data / capacity_data).values
        return yields
    
    def _calculate_confidence_interval(self, prediction, residual_variance=None, confidence=0.90):

        """Calculate confidence interval for a prediction"""
        from scipy.stats import norm
        
        # If no residual variance provided, calculate from baseline residuals
        if residual_variance is None:
            residual_variance = np.var(self.residuals)
        
        # Calculate confidence interval
        z_score = norm.ppf(1 - (1 - confidence) / 2)
        margin_of_error = z_score * np.sqrt(residual_variance)
        
        ci_interval = {
            'lower_bound': float(prediction - margin_of_error),
            'upper_bound': float(prediction + margin_of_error)
        }
        
        return ci_interval
        
    def simulate_scenario(self, scenario_dict, use_year_specific_residuals=True):
        """
        Simulate scenario with residual correction and capacity handling
        
        Parameters:
        -----------
        scenario_dict : dict
            {year: {variable: value, ...}, ...}
            Can include both generation variables (TWh) and capacity variables (GW)
            Capacity changes will automatically update corresponding generation
        use_year_specific_residuals : bool
            If True, use year-specific residuals; if False, use mean residual
            
        Returns:
        --------
        dict : Simulation results by year
        """
        print(f"\n{'='*60}")
        print("CONSTRAINED FE SCENARIO SIMULATION WITH RESIDUAL CORRECTION")
        print(f"{'='*60}")

        # Use the latest release for predictions
        unique_releases = sorted(self.data['Release'].unique(), key=lambda x: pd.to_datetime(x, format='%b%y'))
        latest_release = unique_releases[-1]

        print(f"\n=== SCENARIO ===")
        results = {}
        
        for year, variable_changes in scenario_dict.items():
            if year not in self.baseline_lookup:
                print(f"⚠️  Warning: Year {year} not in baseline data. Skipping.")
                continue
            
            print(f"\n--- YEAR {int(year)} ---")
            
            # Get baseline data for this year
            baseline_info = self.baseline_lookup[year]
            baseline_row = baseline_info['features'].copy()
            baseline_actual = baseline_info['actual_value']
            baseline_residual = baseline_info['residual']
            baseline_predicted = baseline_info['predicted_value']
            
            # Create scenario data
            scenario_features = baseline_info['features'].copy()
            changes_applied = {}
            capacity_changes = {}
            
            # First pass: Handle capacity changes and convert to generation
            for var, value in variable_changes.items():
                if var.endswith('_Capacity_GW'):
                    # This is a capacity variable
                    capacity_changes[var] = value
                    
                    # Only convert PV, Wind, Hydro capacity to generation (not Battery)
                    if var == 'PV_Capacity_GW':
                        gen_var = 'PV_Generation_TWh'
                        yield_col = 'PV'
                    elif var == 'Wind_Capacity_GW':
                        gen_var = 'Wind_Generation_TWh'
                        yield_col = 'Wind'
                    elif var == 'Hydro_Capacity_GW':
                        gen_var = 'Hydro_Generation_TWh'
                        yield_col = 'Hydro'
                    elif var == 'Battery_storage_Capacity_GW':
                        # Battery is already a model variable, no yield conversion needed
                        continue
                    else:
                        continue
                    
                    # Calculate new generation based on new capacity and yield
                    if (gen_var in self.selected_features and 
                        not self.yields_coefficients.empty and 
                        yield_col in self.yields_coefficients.columns):
                        
                        year_yields = self.yields_coefficients[self.yields_coefficients['Year'] == year]
                        if not year_yields.empty:
                            yield_factor = year_yields[yield_col].iloc[0]
                            new_generation = value * yield_factor
                            
                            # Get baseline capacity for comparison
                            baseline_capacity = None
                            if hasattr(self, 'baseline_capacity') and var in self.baseline_capacity.columns:
                                baseline_cap_data = self.baseline_capacity[self.baseline_capacity['Year'] == year]
                                if not baseline_cap_data.empty:
                                    baseline_capacity = baseline_cap_data[var].iloc[0]
                            
                            print(f"  🔧 Capacity Change: {var}")
                            if baseline_capacity is not None:
                                cap_change_pct = ((value - baseline_capacity) / baseline_capacity) * 100
                                print(f"     Capacity: {baseline_capacity:.2f} GW → {value:.2f} GW ({cap_change_pct:+.1f}%)")
                            else:
                                print(f"     New Capacity: {value:.2f} GW")
                                
                            print(f"     Yield Factor: {yield_factor:.3f} TWh/GW")
                            print(f"     Computed {gen_var}: {scenario_features[gen_var]:.2f} TWh → {new_generation:.2f} TWh")
                            
                            # Update the generation in scenario
                            old_generation = scenario_features[gen_var]
                            scenario_features[gen_var] = new_generation
                            gen_change_pct = ((new_generation - old_generation) / old_generation) * 100 if old_generation != 0 else np.nan
                            changes_applied[gen_var] = gen_change_pct
                            changes_applied[var] = cap_change_pct if baseline_capacity is not None else np.nan
                        else:
                            print(f"⚠️  Warning: No yield data found for {yield_col} in year {year}")
                    else:
                        print(f"⚠️  Warning: Cannot compute generation for {var} - missing yield data or generation variable not in model")

            # Second pass: Handle direct variable changes
            for var, value in variable_changes.items():
                if not var.endswith('_Capacity_GW') or var == 'Battery_storage_Capacity_GW':
                    # Handle non-capacity variables OR Battery_storage_Capacity_GW (which is a direct model variable)
                    if var in self.selected_features:
                        old_value = baseline_row[var]
                        scenario_features[var] = value
                        change_pct = ((value - old_value) / old_value) * 100 if old_value != 0 else np.nan
                        changes_applied[var] = change_pct
                    else:
                        print(f"⚠️  Warning: Variable '{var}' not in model features. Skipping.")
            
            # Make prediction on scenario
            scenario_df = pd.DataFrame([{**scenario_features, 'Release': latest_release, 'Year': year}])
            scenario_prediction_raw = predict_df(self.model, scenario_df)[0]
            
            # Apply residual correction
            scenario_prediction_corrected = scenario_prediction_raw + baseline_residual
            
            # Calculate changes
            absolute_change = scenario_prediction_corrected - baseline_actual
            percentage_change = (absolute_change / baseline_actual) * 100 if baseline_actual != 0 else 0
            
            # Calculate confidence interval for the corrected prediction
            residual_variance = 3.855 if self.y_col == "Baseload_Price_EUR_MWh" else 2.761
            ci_interval = self._calculate_confidence_interval(scenario_prediction_corrected, residual_variance)
            
            # Store results
            results[year] = {
                'baseline': {

                    'actual_value': baseline_actual,
                    'predicted_value': baseline_predicted,
                    'corrected_value': baseline_actual,
                    'residual': baseline_residual,
                    'features': baseline_row
                },
                'scenario': {
                    'predicted_raw': scenario_prediction_raw,
                    'predicted_corrected': scenario_prediction_corrected,
                    'features': scenario_features,
                    'changes_applied': changes_applied,
                    'capacity_changes': capacity_changes
                },
                'results': {
                    'absolute_change': absolute_change,
                    'percentage_change': percentage_change,
                    'confidence_interval': ci_interval
                }
            }
            
            # Print results summary
            print(f"\n  📊 CHANGES SUMMARY:")
            
            # Print renewable capacity changes (if any) - exclude Battery
            renewable_capacity_changes = {k: v for k, v in capacity_changes.items() 
                                        if k in ['PV_Capacity_GW', 'Wind_Capacity_GW', 'Hydro_Capacity_GW']}
            if renewable_capacity_changes:
                print(f"     Renewable Capacity Changes:")
                for cap_var, cap_val in renewable_capacity_changes.items():
                    if cap_var in changes_applied:
                        change_pct = changes_applied[cap_var]
                        if not np.isnan(change_pct):
                            print(f"       {cap_var}: {change_pct:+.1f}% → {cap_val:.2f} GW")
                        else:
                            print(f"       {cap_var}: → {cap_val:.2f} GW (new)")
            
            # Print feature changes (including Battery_storage_Capacity_GW as direct model variable)
            print(f"     Generation/Model Variable Changes:")
            for var in self.selected_features:
                old_val = baseline_row[var]
                new_val = scenario_features[var]
                if var in changes_applied:
                    change_pct = changes_applied[var]
                    if not np.isnan(change_pct):
                        print(f"       {var}: {float(old_val):.2f} → {float(new_val):.2f} ({change_pct:+0.1f}%) 🚩")
                    else:
                        print(f"       {var}: {float(old_val):.2f} → {float(new_val):.2f} (new) 🚩")
                else:
                    print(f"       {var}: {float(old_val):.2f} (unchanged)")
            
            # Print target variable result
            print(f"\n  🎯 PRICE IMPACT:")
            print(f"     {self.y_col}: {float(baseline_actual):.2f} → {float(scenario_prediction_corrected):.2f} EUR/MWh")
            print(f"     Change: {float(absolute_change):+0.2f} EUR/MWh ({float(percentage_change):+0.2f}%)")
            print(f"     95% CI: [{ci_interval['lower_bound']:.2f}, {ci_interval['upper_bound']:.2f}]")
        
        return results
    
    def plot_scenario_comparison(self, scenario_results, figsize=(15, 10)):
        """
        Plot comparison between baseline and scenario results
        
        Parameters:
        -----------
        scenario_results : dict
            Results from simulate_scenario()
        """
        import matplotlib.pyplot as plt
        
        years = sorted(scenario_results.keys())
        
        # Extract data for plotting
        baseline_values = [scenario_results[year]['baseline']['actual_value'] for year in years]
        scenario_values = [scenario_results[year]['scenario']['predicted_corrected'] for year in years]
        changes = [scenario_results[year]['results']['absolute_change'] for year in years]
        pct_changes = [scenario_results[year]['results']['percentage_change'] for year in years]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Constrained FE Scenario Analysis Results', fontsize=16)
        
        # 1. Baseline vs Scenario values
        axes[0,0].plot(years, baseline_values, 'o-', label='Baseline (Actual)', linewidth=2, markersize=8)
        axes[0,0].plot(years, scenario_values, 's-', label='Scenario (Corrected)', linewidth=2, markersize=8)
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel(f'{self.y_col}')
        axes[0,0].set_title('Baseline vs Scenario Values')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Absolute changes
        colors = ['red' if x < 0 else 'green' for x in changes]
        axes[0,1].bar(years, changes, color=colors, alpha=0.7)
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Absolute Change (EUR/MWh)')
        axes[0,1].set_title('Absolute Changes by Year')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Percentage changes
        colors = ['red' if x < 0 else 'green' for x in pct_changes]
        axes[1,0].bar(years, pct_changes, color=colors, alpha=0.7)
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Percentage Change (%)')
        axes[1,0].set_title('Percentage Changes by Year')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Confidence intervals
        lower_bounds = [scenario_results[year]['results']['confidence_interval']['lower_bound'] for year in years]
        upper_bounds = [scenario_results[year]['results']['confidence_interval']['upper_bound'] for year in years]
        
        axes[1,1].fill_between(years, lower_bounds, upper_bounds, alpha=0.3, label='95% Confidence Interval')
        axes[1,1].plot(years, scenario_values, 'o-', label='Scenario Prediction', linewidth=2, markersize=8)
        axes[1,1].set_xlabel('Year')
        axes[1,1].set_ylabel(f'{self.y_col}')
        axes[1,1].set_title('Scenario Predictions with Confidence Intervals')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()

        plt.show()
