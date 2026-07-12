import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    'NI': 'Ni',
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
    'ln(Time to rupture(H))': 'Creep life'
}

def preprocess_data(df):
    """
    Preprocesses input data, handling NaN values with zero and renaming columns.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Rename columns according to column_mapping
    df.rename(columns=column_mapping, inplace=True)

    # Select numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if df[numeric_columns].isna().any().any():
        print(f"NaN values in {df[numeric_columns].columns[df[numeric_columns].isna().any()].tolist()}:")
        print(df[numeric_columns].isna().sum())

    # Replace NaN with zero
    for col in numeric_columns:
        df[col] = df[col].fillna(0)

    if df.isna().any().any():
        raise ValueError(f"Unexpected NaN values in columns: {df.columns[df.isna().any()].tolist()}")

    print(f"Processed {len(df.columns)} features: {df.columns.tolist()}")
    return df

def plot_correlation_heatmap(df, feature_names, title, filename_prefix):
    """
    Generate and save Pearson correlation heatmap as TIFF and PNG.
    """
    # Compute Pearson correlation matrix
    corr_matrix = df[feature_names].corr(method='pearson')

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='RdBu',
        center=0,
        annot=False,
        cbar_kws={'label': 'Pearson correlation'}
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

    # Save plots
    plt.savefig(f'{filename_prefix}.tif', dpi=1000, bbox_inches='tight', format='tif')
    plt.savefig(f'{filename_prefix}.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()

def main():
    # Load data
    print("Reading data...")
    try:
        data = pd.read_csv('ML_data_creep_project.csv')
    except FileNotFoundError:
        print("Error: 'ML_data_creep_project.csv' not found.")
        exit(1)

    # Preprocess data
    print("Preprocessing data...")
    data = preprocess_data(data)

    # Define features for correlation
    feature_names = ['Ni', 'Cr', 'Co', 'Mo', 'Fe', 'Al', 'Ti', 'Hf', 'Nb', 'Ta', 'W', 'V', 'Zr', 'Cu', 'Mn', 'C', 'σ',
                     'T', 'δ', 'Xγ′', 'Dγ', 'Creep life']

    # Verify all features exist
    missing_features = [col for col in feature_names if col not in data.columns]
    if missing_features:
        print(f"Error: Feature columns {missing_features} not found in data.")
        exit(1)

    # Generate correlation heatmap
    print("\nGenerating Pearson correlation heatmap...")
    plot_correlation_heatmap(
        data,
        feature_names,
        'Pearson correlation matrix',
        'pearson_correlation_heatmap'
    )

    print("\nAnalysis complete! Heatmap saved as 'pearson_correlation_heatmap.tif' and 'pearson_correlation_heatmap.png'.")

if __name__ == "__main__":
    main()