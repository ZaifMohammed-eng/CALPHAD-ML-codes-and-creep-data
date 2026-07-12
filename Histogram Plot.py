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
    'axes.titlesize': 26,
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
    'ln(Time to rupture(H))': 'ln(Creep life)(H)'
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


def plot_histogram_grid(df, feature_names, filename_prefix):
    """
    Generate and save a grid of histograms for the specified features.
    """
    # Number of features
    n_features = len(feature_names)

    # Determine grid size (e.g., 4 rows x 6 columns to accommodate 22 features)
    n_rows = 4
    n_cols = 6

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 15), constrained_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    # Plot histogram for each feature
    for idx, feature in enumerate(feature_names):
        if idx < len(axes):
            sns.histplot(
                data=df,
                x=feature,
                ax=axes[idx],
                color='purple',  # Match the color in the provided plot
                bins=30  # Adjust bins for better visualization, can be tuned
            )
            axes[idx].set_title(feature, fontsize=26, color='black', fontweight='bold')
            axes[idx].set_xlabel('Value', fontsize=26, color='black', fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontsize=26, color='black', fontweight='bold')
            axes[idx].tick_params(axis='both', labelsize=26, labelcolor='black')
            for spine in axes[idx].spines.values():
                spine.set_linewidth(2.5)
                spine.set_color('black')

    # Turn off any unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    # Save the plot
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

    # Define features for histograms
    feature_names = ['Ni', 'Cr', 'Co', 'Mo', 'Fe', 'Al', 'Ti', 'Hf', 'Nb', 'Ta', 'W', 'V', 'Zr', 'Cu', 'Mn', 'C', 'σ',
                     'T', 'δ', 'Xγ′', 'Dγ', 'ln(Creep life)(H)']

    # Verify all features exist
    missing_features = [col for col in feature_names if col not in data.columns]
    if missing_features:
        print(f"Error: Feature columns {missing_features} not found in data.")
        exit(1)

    # Generate histogram grid
    print("\nGenerating histogram grid...")
    plot_histogram_grid(
        data,
        feature_names,
        'histogram_grid'
    )

    print("\nAnalysis complete! Histogram grid saved as 'histogram_grid.tif' and 'histogram_grid.png'.")


if __name__ == "__main__":
    main()