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

# Set global plot styling to match publication standards
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 26,
    'font.weight': 'bold',
    'axes.titlesize': 28,
    'axes.labelsize': 26,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 26
})

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
    """
    Preprocesses input data, handling NaN values with zero.
    """
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

    print(f"Processed {len(df.columns)} features: {df.columns.tolist()}")
    return (df, training_stats) if not is_test else df

def shap_summary_plot(gbr_model, x_scaled, x_df, feature_names):
    """
    Generate SHAP summary plot for the entire dataset.
    """
    explainer = shap.TreeExplainer(gbr_model)
    shap_values = explainer.shap_values(x_scaled)

    shap_min = shap_values.min()
    shap_max = shap_values.max()
    x_range = max(abs(shap_min), abs(shap_max)) * 1.1

    plt.figure(figsize=(10, 10))
    shap.summary_plot(
        shap_values,
        x_df,
        feature_names=feature_names,
        plot_type="dot",
        show=False
    )
    plt.title('SHAP summary plot of all features using GB on full dataset',
              pad=20, color='black', fontweight='bold', fontsize=28)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=26, labelcolor='black')
    ax.set_xlabel('SHAP Value (Impact on Model)', fontsize=26, color='black', fontweight='bold')
    ax.set_ylabel('Features', fontsize=26, color='black', fontweight='bold')
    ax.set_xlim(-x_range, x_range)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    plt.tight_layout()
    plt.savefig('shap_summary_full_dataset.tif', dpi=1000, bbox_inches='tight', format='tif')
    plt.savefig('shap_summary_full_dataset.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()

    return shap_values

def feature_prioritization(shap_values, x_df, feature_names):
    """
    Generate feature prioritization plot based on mean absolute SHAP values.
    """
    shap_explanation = shap.Explanation(
        values=shap_values,
        feature_names=feature_names,
        data=x_df
    )

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_shap = mean_abs_shap.sum()
    shap_percentages = (mean_abs_shap / total_shap) * 100

    plt.figure(figsize=(10, 10))
    shap.plots.bar(
        shap_explanation,
        max_display=len(feature_names),
        show=False
    )
    plt.title('SHAP feature prioritization using GB on full dataset',
              pad=20, color='black', fontweight='bold', fontsize=28)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=26, labelcolor='black')
    ax.set_xlabel('Mean(|SHAP Value|) (%)', fontsize=26, color='black', fontweight='bold')
    ax.set_ylabel('Features', fontsize=26, color='black', fontweight='bold')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([f'{x:.1f}' for x in (ax.get_xticks() / total_shap * 100)])
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    plt.tight_layout()
    plt.savefig('feature_prioritization_full_dataset.tif', dpi=1000, bbox_inches='tight', format='tif')
    plt.savefig('feature_prioritization_full_dataset.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()

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
        print("error: One or both CSV files not found.")
        exit(1)

    # Combine train and test data
    full_data = pd.concat([train_data, test_data], ignore_index=True)
    print(f"Total number of instances in full dataset: {len(full_data)}")

    # Rename columns
    full_data.rename(columns=column_mapping, inplace=True)

    # Preprocess data
    print("Preprocessing data...")
    full_data, training_stats = preprocess_data(full_data, is_test=False)

    # Define features and target
    target_name = 'target'
    feature_names = ['Ni', 'Cr', 'Co', 'Mo', 'Fe', 'Al', 'Ti', 'Hf', 'Nb', 'Ta', 'W', 'V', 'Zr', 'Cu', 'Mn', 'C', 'σ',
                     'T', 'δ', 'Xγ′', 'Dγ']

    print(f'Features: {feature_names}')
    print(f'Target: {target_name}')

    if target_name not in full_data.columns:
        print(f"Error: Target column '{target_name}' not found in data.")
        exit(1)
    missing_features = [col for col in feature_names if col not in full_data.columns]
    if missing_features:
        print(f"Error: Feature columns {missing_features} not found in data.")
        exit(1)

    X_full = full_data[feature_names]
    y_full = full_data[target_name]

    # Create and fit preprocessing pipeline
    preprocess_pipeline = create_preprocessing_pipeline()
    preprocess_pipeline.fit(X_full)

    # Transform data
    X_scaled_full = preprocess_pipeline.transform(X_full)

    # Train GradientBoostingRegressor
    print("Training GradientBoostingRegressor model...")
    gbr_model = GradientBoostingRegressor(**gbr_params, random_state=random_seed)
    gbr_model.fit(X_scaled_full, y_full)

    # SHAP analysis
    X_full_df = pd.DataFrame(X_scaled_full, columns=feature_names)

    print("\nGenerating SHAP summary plot for full dataset...")
    shap_values = shap_summary_plot(gbr_model, X_scaled_full, X_full_df, feature_names)

    print("\nGenerating feature prioritization plot for full dataset...")
    feature_prioritization(shap_values, X_full_df, feature_names)

    print("\nAnalysis complete! All visualizations saved.")

if __name__ == "__main__":
    main()