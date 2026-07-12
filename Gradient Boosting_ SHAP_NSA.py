import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import shap
import itertools

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


def horizontal_explanations(gbr_model, x_scaled, X_test, feature_names, first_test_instance_idx=653):
    """
    Generate SHAP-based waterfall plot for a specific test instance, displaying original feature values.
    """
    if first_test_instance_idx >= len(X_test):
        raise ValueError(
            f"Test instance index {first_test_instance_idx} is out of bounds. Maximum index is {len(X_test) - 1}.")

    explainer = shap.TreeExplainer(gbr_model)
    shap_values = explainer(x_scaled)

    sample_ind = first_test_instance_idx
    instance_to_explain_original = X_test.iloc[sample_ind:sample_ind + 1]

    shap_explanation = shap.Explanation(
        values=shap_values.values[sample_ind],
        base_values=shap_values.base_values[sample_ind],
        data=instance_to_explain_original.values[0],
        feature_names=feature_names
    )

    plt.figure(figsize=(12, 20))
    shap.plots.waterfall(
        shap_explanation,
        max_display=len(feature_names),
        show=False
    )

    ax = plt.gca()
    for text in ax.texts:
        text_content = text.get_text()
        if "f(x)" in text_content:
            current_pos = text.get_position()
            text.set_position((current_pos[0], current_pos[1] + 0.05))
            text.set_fontsize(22)

    ax.tick_params(axis='both', labelsize=28, labelcolor='black')
    ax.set_ylabel('Features', fontsize=35, color='black', fontweight='bold')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'shap_waterfall_test_instance_{first_test_instance_idx}.tif', dpi=1000, bbox_inches='tight',
                format='tif')
    plt.savefig(f'shap_waterfall_test_instance_{first_test_instance_idx}.png', dpi=300, bbox_inches='tight',
                format='png')
    plt.close()

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values.values[sample_ind],
        'Feature Value': instance_to_explain_original.values[0]
    })
    shap_df['Absolute SHAP'] = shap_df['SHAP Value'].abs()
    shap_df = shap_df.sort_values('Absolute SHAP', ascending=False)

    print(f"\nSHAP Explanation Table for Test Instance {first_test_instance_idx} (Sorted by Impact):")
    print(shap_df[['Feature', 'Feature Value', 'SHAP Value']].to_string(index=False))

    return shap_df, shap_values.values[sample_ind], explainer


def shap_summary_plot(gbr_model, x_scaled, x_df, feature_names):
    """
    Generate SHAP summary plot for the test dataset.
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
    plt.title('SHAP summary plot of all features using GB on test data',
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
    plt.savefig('shap_summary_test.tif', dpi=1000, bbox_inches='tight', format='tif')
    plt.savefig('shap_summary_test.png', dpi=300, bbox_inches='tight', format='png')
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
    plt.title('SHAP feature prioritization using GB based on test data',
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
    plt.savefig('feature_prioritization.tif', dpi=1000, bbox_inches='tight', format='tif')
    plt.savefig('feature_prioritization.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()


def compute_sensitivity(model, X_scaled, feature_names, scaler, X_unscaled_df, instance_idx=None):
    """
    Compute individual (S_j) and pairwise (S_ij) sensitivities by perturbing features
    in the scaled space, with perturbation sizes equivalent to 10% of the original range.
    """
    # Compute 10% of the original feature ranges
    feature_ranges = X_unscaled_df.max() - X_unscaled_df.min()
    delta_x_unscaled = 0.1 * feature_ranges

    # Convert unscaled perturbations to scaled space
    delta_x_scaled = np.zeros(len(feature_names))
    for j, feature in enumerate(feature_names):
        if scaler.scale_[j] != 0:  # Avoid division by zero
            delta_x_scaled[j] = delta_x_unscaled[feature] / scaler.scale_[j]
        else:
            delta_x_scaled[j] = 0  # If scale is 0, no perturbation in scaled space

    n_features = len(feature_names)
    S_j = np.zeros(n_features)
    S_ij = np.zeros((n_features, n_features))

    if instance_idx is not None:
        x_scaled = X_scaled[instance_idx].reshape(1, -1)
        f_x = model.predict(x_scaled)[0]

        # Individual sensitivities (S_j)
        for j in range(n_features):
            x_perturbed = x_scaled.copy()
            x_perturbed[0, j] += delta_x_scaled[j]
            f_perturbed = model.predict(x_perturbed)[0]
            S_j[j] = (f_perturbed - f_x) / delta_x_scaled[j] if delta_x_scaled[j] != 0 else 0

        # Pairwise sensitivities (S_ij)
        for i, j in itertools.combinations(range(n_features), 2):
            x_perturbed = x_scaled.copy()
            x_perturbed[0, i] += delta_x_scaled[i]
            x_perturbed[0, j] += delta_x_scaled[j]
            f_perturbed = model.predict(x_perturbed)[0]
            denominator = np.sqrt(delta_x_scaled[i] ** 2 + delta_x_scaled[j] ** 2)
            S_ij[i, j] = (f_perturbed - f_x) / denominator if denominator != 0 else 0
            S_ij[j, i] = S_ij[i, j]
    else:
        n_instances = X_scaled.shape[0]
        S_j_instances = np.zeros((n_instances, n_features))
        S_ij_instances = np.zeros((n_instances, n_features, n_features))

        for idx in range(n_instances):
            x_scaled = X_scaled[idx].reshape(1, -1)
            f_x = model.predict(x_scaled)[0]

            for j in range(n_features):
                x_perturbed = x_scaled.copy()
                x_perturbed[0, j] += delta_x_scaled[j]
                f_perturbed = model.predict(x_perturbed)[0]
                S_j_instances[idx, j] = (f_perturbed - f_x) / delta_x_scaled[j] if delta_x_scaled[j] != 0 else 0

            for i, j in itertools.combinations(range(n_features), 2):
                x_perturbed = x_scaled.copy()
                x_perturbed[0, i] += delta_x_scaled[i]
                x_perturbed[0, j] += delta_x_scaled[j]
                f_perturbed = model.predict(x_perturbed)[0]
                denominator = np.sqrt(delta_x_scaled[i] ** 2 + delta_x_scaled[j] ** 2)
                S_ij_instances[idx, i, j] = (f_perturbed - f_x) / denominator if denominator != 0 else 0
                S_ij_instances[idx, j, i] = S_ij_instances[idx, i, j]

        S_j = np.mean(S_j_instances, axis=0)
        S_ij = np.mean(S_ij_instances, axis=0)

    return S_j, S_ij


def plot_sensitivity_heatmap(S_matrix, feature_names, title, filename_prefix):
    """
    Plot and save sensitivity heatmap as TIFF and PNG, using original feature names.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        S_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap = 'RdBu'
        #cmap='Blues',
        center=0,
        annot=False,
        cbar_kws={'label': 'Sensitivity'}
    )
    plt.title(title, pad=20, color='black', fontweight='bold', fontsize=28)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=26, labelcolor='black')
    ax.set_xlabel('Features', fontsize=26, color='black', fontweight='bold')
    ax.set_ylabel('Features', fontsize=26, color='black', fontweight='bold')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    plt.tight_layout()

    plt.savefig(f'{filename_prefix}.tif', dpi=1000, bbox_inches='tight', format='tif')
    plt.savefig(f'{filename_prefix}.png', dpi=300, bbox_inches='tight', format='png')
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
        print("Error: One or both CSV files not found.")
        exit(1)

    print(f"Number of test instances: {len(test_data)}")

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

    print(f'Features: {feature_names}')
    print(f'Target: {target_name}')

    if target_name not in train_data.columns:
        print(f"Error: Target column '{target_name}' not found in training data.")
        exit(1)
    missing_features = [col for col in feature_names if col not in train_data.columns]
    if missing_features:
        print(f"Error: Feature columns {missing_features} not found in training data.")
        exit(1)

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

    # Evaluate GBR model
    y_pred_gbr = gbr_model.predict(X_scaled_test)
    mse_gbr = mean_squared_error(y_test, y_pred_gbr)
    r2_gbr = r2_score(y_test, y_pred_gbr)
    print(f"GBR Mean Squared Error: {mse_gbr:.4f}")
    print(f"GBR R² Score: {r2_gbr:.4f}")

    # SHAP analysis
    X_test_df = pd.DataFrame(X_scaled_test, columns=feature_names)

    print("\nGenerating SHAP explanation for test instance 653...")
    shap_df, instance_shap_values, explainer = horizontal_explanations(
        gbr_model, X_scaled_test, X_test, feature_names, first_test_instance_idx=653
    )

    print("\nGenerating SHAP summary plot...")
    shap_values = shap_summary_plot(gbr_model, X_scaled_test, X_test_df, feature_names)

    print("\nGenerating feature prioritization plot...")
    feature_prioritization(shap_values, X_test_df, feature_names)

    # Sensitivity analysis for GBR
    print("\nComputing sensitivities for test instance 653...")
    if 653 >= len(X_scaled_test):
        print(f"Error: Test instance 653 is out of bounds. Maximum index is {len(X_scaled_test) - 1}.")
        exit(1)
    S_j_653, S_ij_653 = compute_sensitivity(
        gbr_model, X_scaled_test, feature_names, preprocess_pipeline.named_steps['scaler'], X_test, instance_idx=653
    )

    print("\nComputing global sensitivities across all test data...")
    S_j_global, S_ij_global = compute_sensitivity(
        gbr_model, X_scaled_test, feature_names, preprocess_pipeline.named_steps['scaler'], X_test
    )

    # Plot sensitivity heatmaps
    print("\nGenerating heatmap for test instance 653...")
    plot_sensitivity_heatmap(
        S_ij_653,
        feature_names,
        'Sensitivity matrix using GB for sample test instance',
        'sensitivity_heatmap_instance_653'
    )

    print("\nGenerating global sensitivity heatmap...")
    plot_sensitivity_heatmap(
        S_ij_global,
        feature_names,
        'Global sensitivity matrix using GB across all the test data',
        'global_sensitivity_heatmap'
    )

    # Save sensitivity results
    S_j_df_653 = pd.DataFrame({
        'Feature': feature_names,
        'Sensitivity (S_j)': S_j_653
    })
    S_j_df_global = pd.DataFrame({
        'Feature': feature_names,
        'Sensitivity (S_j)': S_j_global
    })

    print("\nIndividual Sensitivities for Test Instance 653 (GBR):")
    print(S_j_df_653.to_string(index=False))

    print("\nGlobal Individual Sensitivities (GBR):")
    print(S_j_df_global.to_string(index=False))

    S_j_df_653.to_csv('sensitivity_instance_653_gbr.csv', index=False)
    S_j_df_global.to_csv('global_sensitivity_gbr.csv', index=False)
    pd.DataFrame(S_ij_653, columns=feature_names, index=feature_names).to_csv('sensitivity_matrix_instance_653_gbr.csv')
    pd.DataFrame(S_ij_global, columns=feature_names, index=feature_names).to_csv('global_sensitivity_matrix_gbr.csv')

    print("\nAnalysis complete! All visualizations and data files saved.")


if __name__ == "__main__":
    main()