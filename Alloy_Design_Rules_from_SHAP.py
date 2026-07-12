"""
Alloy Design Rules Analysis Based on SHAP Dependence Plots
This script analyzes SHAP values to derive logical alloy design rules and selection criteria.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

# Column renaming dictionary
column_mapping = {
    'NI ': 'Ni',
    'CR': 'Cr',
    'CO': 'Co',
    'MO': 'Mo',
    'FE': 'Fe',
    'AL': 'Al',
    'TI': 'Ti',
    'HF': 'Hf',
    'NB': 'Nb',
    'TA': 'Ta',
    'W': 'W',
    'V': 'V',
    'ZR': 'Zr',
    'CU': 'Cu',
    'MN': 'Mn',
    'C': 'C',
    'Stress (Mpa)': 'σ',
    'Temperature': 'T',
    'Lattice Misfit': 'δ',
    'Amount of Gamma Prime': 'Xγ′',
    'ln(Effective Diffusivity of Gamma Phase)': 'Dγ',
    'target': 'target'
}


def create_preprocessing_pipeline():
    """Creates a preprocessing pipeline with StandardScaler"""
    return Pipeline([
        ('scaler', StandardScaler())
    ])


def preprocess_data(df, is_test=False, training_stats=None):
    """Preprocesses input data, handling NaN values with zero."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if df[numeric_columns].isna().any().any():
        print(f"NaN values in {df[numeric_columns].columns[df[numeric_columns].isna().any()].tolist()}:")
        print(df[numeric_columns].isna().sum())

    if not is_test:
        training_stats = {
            'numeric_columns': list(numeric_columns),
            'feature_names': df.columns.tolist()
        }

    for col in numeric_columns:
        df[col] = df[col].fillna(0)

    if df.isna().any().any():
        raise ValueError(f"Unexpected NaN values in columns: {df.columns[df.isna().any()].tolist()}")

    return (df, training_stats) if not is_test else df


def analyze_shap_dependence_for_design_rules(gbr_model, x_scaled, x_unscaled_df, y_predicted, feature_names):
    """
    Analyze SHAP dependence patterns to derive alloy design rules.
    
    Returns:
    - design_rules: Dictionary with design rules for each feature
    - optimal_ranges: Optimal value ranges for each feature
    """
    # Compute SHAP values - use a sample if dataset is too large
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(gbr_model)
    
    # If dataset is too large, sample it for SHAP computation
    max_samples = 1000
    if len(x_scaled) > max_samples:
        print(f"Dataset too large ({len(x_scaled)} samples). Sampling {max_samples} instances for SHAP computation...")
        sample_indices = np.random.choice(len(x_scaled), max_samples, replace=False)
        x_scaled_sample = x_scaled[sample_indices]
        x_unscaled_sample = x_unscaled_df.iloc[sample_indices]
        y_predicted_sample = y_predicted[sample_indices]
    else:
        x_scaled_sample = x_scaled
        x_unscaled_sample = x_unscaled_df
        y_predicted_sample = y_predicted
        sample_indices = np.arange(len(x_scaled))
    
    try:
        shap_values = explainer.shap_values(x_scaled_sample)
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        print("Trying alternative approach...")
        shap_values = explainer(x_scaled_sample).values
    
    # Ensure shap_values is 2D
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(-1, 1)
    
    design_rules = {}
    optimal_ranges = {}
    feature_importance = {}
    
    print("\n" + "="*80)
    print("ALLOY DESIGN RULES ANALYSIS BASED ON SHAP DEPENDENCE PLOTS")
    print("="*80)
    
    for feature_idx, feature in enumerate(feature_names):
        feature_values = x_unscaled_sample[feature].values
        
        # Handle different SHAP output formats
        if len(shap_values.shape) == 2:
            feature_shap_values = shap_values[:, feature_idx]
        else:
            feature_shap_values = shap_values
        
        # Analyze relationships
        # 1. Positive SHAP values = increases creep rupture life
        # 2. Negative SHAP values = decreases creep rupture life
        
        # Find optimal ranges (where SHAP values are most positive)
        positive_shap_mask = feature_shap_values > 0
        negative_shap_mask = feature_shap_values < 0
        
        if np.sum(positive_shap_mask) > 0:
            optimal_low = np.percentile(feature_values[positive_shap_mask], 25)
            optimal_high = np.percentile(feature_values[positive_shap_mask], 75)
            optimal_median = np.median(feature_values[positive_shap_mask])
        else:
            optimal_low = optimal_high = optimal_median = np.median(feature_values)
        
        # Find detrimental ranges (where SHAP values are most negative)
        if np.sum(negative_shap_mask) > 0:
            detrimental_low = np.percentile(feature_values[negative_shap_mask], 25)
            detrimental_high = np.percentile(feature_values[negative_shap_mask], 75)
        else:
            detrimental_low = detrimental_high = np.median(feature_values)
        
        # Calculate correlation between feature values and SHAP values
        correlation = np.corrcoef(feature_values, feature_shap_values)[0, 1]
        
        # Calculate mean absolute SHAP value (importance)
        mean_abs_shap = np.abs(feature_shap_values).mean()
        
        # Determine trend
        if correlation > 0.3:
            trend = "INCREASING (Higher values → Higher creep rupture life)"
            recommendation = f"Increase {feature} to improve creep rupture life"
        elif correlation < -0.3:
            trend = "DECREASING (Higher values → Lower creep rupture life)"
            recommendation = f"Decrease {feature} to improve creep rupture life"
        else:
            trend = "NON-LINEAR (Complex relationship)"
            recommendation = f"Optimize {feature} within range {optimal_low:.2f}-{optimal_high:.2f}"
        
        # Store results
        design_rules[feature] = {
            'trend': trend,
            'correlation': correlation,
            'mean_abs_shap': mean_abs_shap,
            'recommendation': recommendation,
            'optimal_range': (optimal_low, optimal_high),
            'optimal_median': optimal_median,
            'detrimental_range': (detrimental_low, detrimental_high)
        }
        
        optimal_ranges[feature] = {
            'optimal_low': optimal_low,
            'optimal_high': optimal_high,
            'optimal_median': optimal_median,
            'current_range': (feature_values.min(), feature_values.max())
        }
        
        feature_importance[feature] = mean_abs_shap
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Print analysis results
    print("\nFEATURE IMPORTANCE RANKING (by mean absolute SHAP value):")
    print("-" * 80)
    for rank, (feature, importance) in enumerate(sorted_features, 1):
        print(f"{rank:2d}. {feature:15s}: {importance:.4f}")
    
    print("\n" + "="*80)
    print("DETAILED DESIGN RULES FOR EACH ELEMENT:")
    print("="*80)
    
    for feature in feature_names:
        rule = design_rules[feature]
        print(f"\n{feature}:")
        print(f"  Trend: {rule['trend']}")
        print(f"  Correlation (Feature vs SHAP): {rule['correlation']:.3f}")
        print(f"  Mean Absolute SHAP: {rule['mean_abs_shap']:.4f}")
        print(f"  Optimal Range: {rule['optimal_range'][0]:.2f} - {rule['optimal_range'][1]:.2f}")
        print(f"  Optimal Median: {rule['optimal_median']:.2f}")
        print(f"  Recommendation: {rule['recommendation']}")
    
    return design_rules, optimal_ranges, sorted_features


def generate_design_rules_summary(design_rules, optimal_ranges, sorted_features):
    """Generate a comprehensive summary of alloy design rules."""
    
    print("\n" + "="*80)
    print("ALLOY DESIGN RULES SUMMARY")
    print("="*80)
    
    # Categorize elements
    chemical_elements = ['Ni', 'Cr', 'Co', 'Mo', 'Fe', 'Al', 'Ti', 'Hf', 'Nb', 'Ta', 'W', 'V', 'Zr', 'Cu', 'Mn', 'C']
    process_conditions = ['σ', 'T']
    calphad_descriptors = ['δ', 'Xγ′', 'Dγ']
    
    print("\n1. CHEMICAL COMPOSITION RULES:")
    print("-" * 80)
    for feature in chemical_elements:
        if feature in design_rules:
            rule = design_rules[feature]
            opt_range = optimal_ranges[feature]
            print(f"   {feature:3s}: Target range {opt_range['optimal_low']:6.2f} - {opt_range['optimal_high']:6.2f} "
                  f"(Optimal: {opt_range['optimal_median']:6.2f})")
            print(f"          {rule['recommendation']}")
    
    print("\n2. PROCESS CONDITION RULES:")
    print("-" * 80)
    for feature in process_conditions:
        if feature in design_rules:
            rule = design_rules[feature]
            opt_range = optimal_ranges[feature]
            print(f"   {feature:3s}: Target range {opt_range['optimal_low']:6.2f} - {opt_range['optimal_high']:6.2f} "
                  f"(Optimal: {opt_range['optimal_median']:6.2f})")
            print(f"          {rule['recommendation']}")
    
    print("\n3. CALPHAD DESCRIPTOR RULES:")
    print("-" * 80)
    for feature in calphad_descriptors:
        if feature in design_rules:
            rule = design_rules[feature]
            opt_range = optimal_ranges[feature]
            print(f"   {feature:3s}: Target range {opt_range['optimal_low']:6.2f} - {opt_range['optimal_high']:6.2f} "
                  f"(Optimal: {opt_range['optimal_median']:6.2f})")
            print(f"          {rule['recommendation']}")
    
    print("\n4. SELECTION CRITERIA:")
    print("-" * 80)
    print("   Based on SHAP analysis, the following criteria should be used for alloy design:")
    print("   a) Prioritize features with highest mean absolute SHAP values")
    print("   b) Optimize features with strong positive correlations (increase these)")
    print("   c) Minimize features with strong negative correlations (decrease these)")
    print("   d) Fine-tune features with non-linear relationships within optimal ranges")
    print("   e) Consider interactions between features (use SHAP interaction values)")
    
    # Top 5 most important features
    print("\n5. TOP 5 MOST CRITICAL FEATURES FOR ALLOY DESIGN:")
    print("-" * 80)
    for rank, (feature, importance) in enumerate(sorted_features[:5], 1):
        rule = design_rules[feature]
        opt_range = optimal_ranges[feature]
        print(f"   {rank}. {feature:15s} (Importance: {importance:.4f})")
        print(f"      Optimal Range: {opt_range['optimal_low']:.2f} - {opt_range['optimal_high']:.2f}")
        print(f"      {rule['recommendation']}")


def main():
    # Load best parameters for GBR
    print("Loading GBR parameters...")
    try:
        with open('fold2_best_params_using CALPHAD descriptors.json', 'r') as f:
            gbr_params = json.load(f)
    except FileNotFoundError:
        print("Error: 'fold2_best_params_using CALPHAD descriptors.json' not found.")
        exit(1)

    # Load and rename columns
    print("Reading data...")
    try:
        train_data = pd.read_csv('fold2_train_data_using CALPHAD descriptors.csv')
        test_data = pd.read_csv('fold2_test_data_using CALPHAD descriptors.csv')
    except FileNotFoundError:
        print("Error: One or both CSV files not found.")
        exit(1)

    # Rename columns
    train_data.rename(columns=column_mapping, inplace=True)
    test_data.rename(columns=column_mapping, inplace=True)

    # Preprocess data
    print("Preprocessing data...")
    train_data, training_stats = preprocess_data(train_data, is_test=False)
    test_data = preprocess_data(test_data, is_test=True, training_stats=training_stats)

    # Define features and target
    target_name = 'target'
    feature_names = ['Ni', 'Cr', 'Co', 'Mo', 'Fe', 'Al', 'Ti', 'Hf', 'Nb', 'Ta', 'W', 'V', 'Zr', 'Cu', 'Mn', 'C', 'σ',
                     'T', 'δ', 'Xγ′', 'Dγ']

    X_train = train_data[feature_names]
    y_train = train_data[target_name]
    X_test = test_data[feature_names]
    y_test = test_data[target_name]

    # Create and fit preprocessing pipeline
    preprocess_pipeline = create_preprocessing_pipeline()
    preprocess_pipeline.fit(X_train)

    # Transform data
    X_scaled_train = preprocess_pipeline.transform(X_train)
    X_scaled_test = preprocess_pipeline.transform(X_test)

    # Train GradientBoostingRegressor
    print("Training GradientBoostingRegressor model...")
    gbr_model = GradientBoostingRegressor(**gbr_params, random_state=random_seed)
    gbr_model.fit(X_scaled_train, y_train)

    # Combine training and testing data for comprehensive analysis
    print("\nCombining training and testing data for comprehensive SHAP analysis...")
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = pd.concat([y_train, y_test], ignore_index=True)
    X_scaled_combined = preprocess_pipeline.transform(X_combined)
    y_predicted_combined = gbr_model.predict(X_scaled_combined)
    
    print(f"Combined dataset size: {len(X_combined)} instances")

    # Analyze SHAP dependence for design rules
    design_rules, optimal_ranges, sorted_features = analyze_shap_dependence_for_design_rules(
        gbr_model, X_scaled_combined, X_combined, y_predicted_combined, feature_names
    )
    
    # Generate summary
    generate_design_rules_summary(design_rules, optimal_ranges, sorted_features)
    
    # Save results to CSV
    print("\n" + "="*80)
    print("SAVING RESULTS TO CSV FILES")
    print("="*80)
    
    # Save design rules
    rules_df = pd.DataFrame({
        'Feature': feature_names,
        'Trend': [design_rules[f]['trend'] for f in feature_names],
        'Correlation': [design_rules[f]['correlation'] for f in feature_names],
        'Mean_Abs_SHAP': [design_rules[f]['mean_abs_shap'] for f in feature_names],
        'Optimal_Low': [optimal_ranges[f]['optimal_low'] for f in feature_names],
        'Optimal_High': [optimal_ranges[f]['optimal_high'] for f in feature_names],
        'Optimal_Median': [optimal_ranges[f]['optimal_median'] for f in feature_names],
        'Recommendation': [design_rules[f]['recommendation'] for f in feature_names]
    })
    rules_df = rules_df.sort_values('Mean_Abs_SHAP', ascending=False)
    rules_df.to_csv('alloy_design_rules_from_shap.csv', index=False)
    print("Design rules saved to: alloy_design_rules_from_shap.csv")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
