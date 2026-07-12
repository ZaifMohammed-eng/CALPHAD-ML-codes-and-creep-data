import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import json

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)


def create_preprocessing_pipeline():
    """Creates a preprocessing pipeline with StandardScaler."""
    return Pipeline([('scaler', StandardScaler())])


def preprocess_data(df, is_test=False, training_stats=None):
    """
    Preprocesses input data, handling NaN values with zero.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with alloy features.
    - is_test (bool): If True, use training_stats for consistent preprocessing.
    - training_stats (dict): Stats from training data (e.g., numeric columns).

    Returns:
    - If not is_test: (preprocessed_df, training_stats)
    - If is_test: preprocessed_df
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Identify numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Log NaN prevalence
    if df[numeric_columns].isna().any().any():
        print(f"NaN values in {df[numeric_columns].columns[df[numeric_columns].isna().any()].tolist()}:")
        print(df[numeric_columns].isna().sum())

    # Initialize training_stats if not testing
    if not is_test:
        training_stats = {
            'numeric_columns': list(numeric_columns),
            'feature_names': df.columns.tolist()
        }

    # Replace NaN with zero
    for col in numeric_columns:
        df[col].fillna(0, inplace=True)

    # Verify no NaN values remain
    if df.isna().any().any():
        raise ValueError(f"Unexpected NaN values in columns: {df.columns[df.isna().any()].tolist()}")

    print(f"Processed {len(df.columns)} features: {df.columns.tolist()}")
    return (df, training_stats) if not is_test else df

def plot_predictions(y_true, y_pred, title, save_path):
    """Creates prediction vs actual plot with R2 score for research publication."""
    plt.figure(figsize=(12,12))

      # Set font to Times New Roman for publication standards
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 26,
        'font.weight': 'bold',
        'axes.titlesize': 28,
        'font.weight': 'bold',
        'axes.labelsize': 26,
        'font.weight': 'bold',
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'legend.fontsize': 26
    })

    # Scatter plot of predictions
    plt.scatter(y_true, y_pred, alpha=0.9, label='Data points', color='blue', s=50)

    # Perfect prediction line
    plt.plot([-5,40],[-5,40], 'r--', label = 'Perfect prediction', linewidth = 3)

    # Calculate and display R2 score in top left
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'),
             verticalalignment='top', horizontalalignment='left',fontsize = 26 , fontfamily = 'Times New Roman', fontweight = 'bold', color = 'black')

    plt.xlim(-5, 40)
    plt.ylim(-5, 40)

    # Set custom tick marks to explicitly show -5 and 40
    plt.xticks([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40], color = 'black', fontweight = 'bold')
    plt.yticks([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40], color = 'black', fontweight = 'bold')

    # Set labels and title
    plt.xlabel('ln(Measured creep rupture life) (h)', color = 'black', fontweight = 'bold')
    plt.ylabel('ln(Predicted creep rupture life) (h)', color = 'black', fontweight = 'bold')
    plt.title(title, pad = 20, color = 'black', fontweight = 'bold')

    # Add legend in bottom right
    plt.legend(loc='lower right', labelcolor = 'black', prop = {'weight': 'bold', 'size': 26})

    # Save figure with high quality
    plt.savefig(save_path, dpi = 1000,  bbox_inches='tight', format='tif')
    plt.close()


def optimize_hyperparameters(x_train, y_train):
    """
    Optimizes SVR hyperparameters using Optuna with stratified k-fold cross validation.
    """
    # Convert inputs to arrays
    x_train_array = x_train.values if isinstance(x_train, pd.DataFrame) else x_train
    y_train_array = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

    # Create bins for stratification
    n_bins = min(10, len(np.unique(y_train_array)))
    try:
        y_binned = pd.qcut(y_train_array.ravel(), q=n_bins, labels=False, duplicates='drop')
    except ValueError as e:
        print(f"Warning: qcut failed with error {str(e)}. Using equal bins instead.")
        y_binned = pd.cut(y_train_array.ravel(), bins=n_bins, labels=False, include_lowest=True)

    def objective(trial):
        pipeline = Pipeline([
            ('preprocessor', create_preprocessing_pipeline()),
            ('svm', SVR(
                kernel='rbf',
                C=trial.suggest_float("C", 1e1, 1e4, log=True),
                epsilon=trial.suggest_float("epsilon", 1e-2, 1.0, log=True),
                gamma=trial.suggest_float("gamma", 1e-3, 1e0, log=True),
                tol=trial.suggest_float("tol", 1e-4, 1e-2, log=True),
                max_iter=100000,
                cache_size=1000
            ))
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        cv_scores = []

        for train_idx, val_idx in skf.split(x_train_array, y_binned):
            X_fold_train, X_fold_val = x_train_array[train_idx], x_train_array[val_idx]
            y_fold_train, y_fold_val = y_train_array[train_idx], y_train_array[val_idx]

            try:
                pipeline.fit(X_fold_train, y_fold_train)
                y_pred = pipeline.predict(X_fold_val)
                fold_mse = mean_squared_error(y_fold_val, y_pred)
                cv_scores.append(fold_mse)
            except Exception as e:
                print(f"Warning: Fold failed with error {str(e)}")
                return float('inf')

        return np.mean(cv_scores) if cv_scores else float('inf')

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    return study.best_params, study.best_value


def run_analysis(x, y, suffix):
    """
    Runs the complete analysis pipeline with SVR using KFold for train-test splitting.
    Plots predictions for the fold with R², MAE, and RMSE closest to the mean performance across folds.
    """
    # Initialize KFold for 80/20 train-test splits
    kf = KFold(n_splits=3, shuffle=True, random_state=random_seed)
    fold_results = []
    first_fold_best_params = None

    print(f"\nRunning KFold analysis for {suffix}...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(x), 1):
        print(f"\nProcessing Fold {fold}...")
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Preprocess data
        x_train_preprocessed, training_stats = preprocess_data(x_train, is_test=False)
        x_test_preprocessed = preprocess_data(x_test, is_test=True, training_stats=training_stats)

        # Optimize hyperparameters on training set
        print(f"Optimizing hyperparameters for Fold {fold}...")
        best_params, best_score = optimize_hyperparameters(x_train_preprocessed, y_train)

        print(f"Best hyperparameters for Fold {fold}:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        # Store best parameters from first fold for final model
        if fold == 1:
            first_fold_best_params = best_params

        # Train model on training set
        fold_pipeline = Pipeline([
            ('preprocessor', create_preprocessing_pipeline()),
            ('svm', SVR(**best_params))
        ])

        try:
            fold_pipeline.fit(x_train_preprocessed, y_train)
            y_train_pred = fold_pipeline.predict(x_train_preprocessed)
            y_test_pred = fold_pipeline.predict(x_test_preprocessed)
        except Exception as e:
            print(f"Error in Fold {fold} model fitting: {str(e)}")
            continue

        # Compute metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        }

        test_metrics = {
            'mse': mean_squared_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred)
        }

        fold_results.append({
            'fold': fold,
            'best_params': best_params,
            'best_score': best_score,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'y_train': y_train,
            'y_train_pred': y_train_pred,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        })

    # Aggregate metrics across folds
    if fold_results:
        aggregated_metrics = {
            'train_metrics': {
                'mse_mean': np.mean([r['train_metrics']['mse'] for r in fold_results]),
                'mse_std': np.std([r['train_metrics']['mse'] for r in fold_results]),
                'rmse_mean': np.mean([r['train_metrics']['rmse'] for r in fold_results]),
                'rmse_std': np.std([r['train_metrics']['rmse'] for r in fold_results]),
                'mae_mean': np.mean([r['train_metrics']['mae'] for r in fold_results]),
                'mae_std': np.std([r['train_metrics']['mae'] for r in fold_results]),
                'r2_mean': np.mean([r['train_metrics']['r2'] for r in fold_results]),
                'r2_std': np.std([r['train_metrics']['r2'] for r in fold_results])
            },
            'test_metrics': {
                'mse_mean': np.mean([r['test_metrics']['mse'] for r in fold_results]),
                'mse_std': np.std([r['test_metrics']['mse'] for r in fold_results]),
                'rmse_mean': np.mean([r['test_metrics']['rmse'] for r in fold_results]),
                'rmse_std': np.std([r['test_metrics']['rmse'] for r in fold_results]),
                'mae_mean': np.mean([r['test_metrics']['mae'] for r in fold_results]),
                'mae_std': np.std([r['test_metrics']['mae'] for r in fold_results]),
                'r2_mean': np.mean([r['test_metrics']['r2'] for r in fold_results]),
                'r2_std': np.std([r['test_metrics']['r2'] for r in fold_results])
            }
        }

        print(f"\nAggregated Metrics for {suffix}:")
        print("Training Set:")
        print(f"  MSE: {aggregated_metrics['train_metrics']['mse_mean']:.4f} ± {aggregated_metrics['train_metrics']['mse_std']:.4f}")
        print(f"  RMSE: {aggregated_metrics['train_metrics']['rmse_mean']:.4f} ± {aggregated_metrics['train_metrics']['rmse_std']:.4f}")
        print(f"  MAE: {aggregated_metrics['train_metrics']['mae_mean']:.4f} ± {aggregated_metrics['train_metrics']['mae_std']:.4f}")
        print(f"  R2: {aggregated_metrics['train_metrics']['r2_mean']:.4f} ± {aggregated_metrics['train_metrics']['r2_std']:.4f}")
        print("Test Set:")
        print(f"  MSE: {aggregated_metrics['test_metrics']['mse_mean']:.4f} ± {aggregated_metrics['test_metrics']['mse_std']:.4f}")
        print(f"  RMSE: {aggregated_metrics['test_metrics']['rmse_mean']:.4f} ± {aggregated_metrics['test_metrics']['rmse_std']:.4f}")
        print(f"  MAE: {aggregated_metrics['test_metrics']['mae_mean']:.4f} ± {aggregated_metrics['test_metrics']['mae_std']:.4f}")
        print(f"  R2: {aggregated_metrics['test_metrics']['r2_mean']:.4f} ± {aggregated_metrics['test_metrics']['r2_std']:.4f}")

        # Find fold with R², MAE, and RMSE closest to mean performance
        mean_r2 = aggregated_metrics['test_metrics']['r2_mean']
        std_r2 = aggregated_metrics['test_metrics']['r2_std'] or 1  # Avoid division by zero
        mean_mae = aggregated_metrics['test_metrics']['mae_mean']
        std_mae = aggregated_metrics['test_metrics']['mae_std'] or 1
        mean_rmse = aggregated_metrics['test_metrics']['rmse_mean']
        std_rmse = aggregated_metrics['test_metrics']['rmse_std'] or 1

        def composite_score(fold):
            r2_diff = abs(fold['test_metrics']['r2'] - mean_r2) / std_r2
            mae_diff = abs(fold['test_metrics']['mae'] - mean_mae) / std_mae
            rmse_diff = abs(fold['test_metrics']['rmse'] - mean_rmse) / std_rmse
            return r2_diff + mae_diff + rmse_diff

        closest_fold = min(fold_results, key=composite_score)
        closest_fold_num = closest_fold['fold']
        print(f"\nFold {closest_fold_num} has R², MAE, and RMSE closest to mean performance:")
        print(f"  Test R²: {closest_fold['test_metrics']['r2']:.4f} (mean: {mean_r2:.4f})")
        print(f"  Test MAE: {closest_fold['test_metrics']['mae']:.4f} (mean: {mean_mae:.4f})")
        print(f"  Test RMSE: {closest_fold['test_metrics']['rmse']:.4f} (mean: {mean_rmse:.4f})")

        # Generate plots for the fold with performance closest to average
        plot_predictions(
            closest_fold['y_train'], closest_fold['y_train_pred'],
            f'Training set predictions from SVR model ({suffix})',
            f'SVR_train_predictions_{suffix}_fold{closest_fold_num}.tif'
        )
        plot_predictions(
            closest_fold['y_test'], closest_fold['y_test_pred'],
            f'Test set predictions from SVR model ({suffix})',
            f'SVR_test_predictions_{suffix}_fold{closest_fold_num}.tif'
        )
    else:
        print(f"Error: KFold analysis failed for {suffix}.")
        return None

    # Train final model on full dataset using first fold's best parameters with specified settings
    final_pipeline = Pipeline([
        ('preprocessor', create_preprocessing_pipeline()),
        ('svm', SVR(**{**first_fold_best_params, 'max_iter': 1000000, 'cache_size': 1000}))
    ])

    x_preprocessed, training_stats = preprocess_data(x, is_test=False)
    try:
        final_pipeline.fit(x_preprocessed, y)
    except Exception as e:
        print(f"Error in final model fitting: {str(e)}")
        return None

    return {
        'fold_results': fold_results,
        'aggregated_metrics': aggregated_metrics,
        'final_pipeline': final_pipeline,
        'feature_names': training_stats['feature_names']
    }


if __name__ == "__main__":
    # Read and prepare data
    try:
        Data = pd.read_csv('ML_data_creep_project.csv', header=0)
    except FileNotFoundError:
        print("Error: 'ML_data_creep_project.csv' not found.")
        exit(1)

    # Feature names
    feature_names = Data.columns[1:-2].tolist()

    # Original feature set (with CALPHAD descriptors)
    x_original = Data.iloc[:, 1:-2]
    y = Data.iloc[:, -2]

    # New feature set (without CALPHAD descriptors)
    all_cols = range(Data.shape[1])
    exclude_cols = {0, 19, 20, 21, Data.shape[1] - 1, Data.shape[1] - 2}
    selected_cols = [col for col in all_cols if col not in exclude_cols]
    x_new = Data.iloc[:, selected_cols]

    # Run analysis for both feature sets
    results_original = run_analysis(x_original, y, "using CALPHAD descriptors")
    results_new = run_analysis(x_new, y, "excluding CALPHAD descriptors")

    if results_original is None or results_new is None:
        print("Error: Analysis failed for one or both feature sets.")
        exit(1)

    # Save combined results (exclude non-serializable objects)
    combined_results = {
        'with_descriptors': {
            'aggregated_metrics': results_original['aggregated_metrics'],
            'fold_results': [
                {
                    'fold': r['fold'],
                    'best_params': r['best_params'],
                    'best_score': r['best_score'],
                    'train_metrics': r['train_metrics'],
                    'test_metrics': r['test_metrics']
                } for r in results_original['fold_results']
            ]
        },
        'without_descriptors': {
            'aggregated_metrics': results_new['aggregated_metrics'],
            'fold_results': [
                {
                    'fold': r['fold'],
                    'best_params': r['best_params'],
                    'best_score': r['best_score'],
                    'train_metrics': r['train_metrics'],
                    'test_metrics': r['test_metrics']
                } for r in results_new['fold_results']
            ]
        }
    }

    with open('SVR_model_results_comparison.json', 'w') as f:
        json.dump(combined_results, f, indent=4)

    # Create comparative metrics plot
    metrics = ['mse', 'rmse', 'mae', 'r2']
    metric_labels = ['MSE', 'RMSE', 'MAE', 'R²']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        data = [
            ['With Descriptors', results_original['aggregated_metrics']['train_metrics'][f'{metric}_mean'], 'Train'],
            ['With Descriptors', results_original['aggregated_metrics']['test_metrics'][f'{metric}_mean'], 'Test'],
            ['Without Descriptors', results_new['aggregated_metrics']['train_metrics'][f'{metric}_mean'], 'Train'],
            ['Without Descriptors', results_new['aggregated_metrics']['test_metrics'][f'{metric}_mean'], 'Test']
        ]
        df_plot = pd.DataFrame(data, columns=['Model', 'Value', 'Set'])

        sns.barplot(x='Model', y='Value', hue='Set', data=df_plot, ax=axes[i])
        axes[i].set_title(f'{label} Comparison (Mean Across Folds)')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylabel(label)

    plt.tight_layout()
    plt.savefig('SVR_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()