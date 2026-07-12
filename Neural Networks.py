import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Masking, Dropout, Input, Activation
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

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

    # Replace NaN with zero using direct assignment
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Verify no NaN values remain
    if df.isna().any().any():
        raise ValueError(f"Unexpected NaN values in columns: {df.columns[df.isna().any()].tolist()}")

    print(f"Processed {len(df.columns)} features: {df.columns.tolist()}")
    return (df, training_stats) if not is_test else df

def create_regularizer(kernel_regularizer_type, reg_strength):
    """
    Creates the appropriate regularizer based on type and strength.
    """
    reg_strength = max(1e-6, float(reg_strength))
    if kernel_regularizer_type == 0:  # l1
        return regularizers.l1(reg_strength)
    elif kernel_regularizer_type == 1:  # l2
        return regularizers.l2(reg_strength)
    else:
        raise ValueError(f"Invalid regularizer type: {kernel_regularizer_type}")

def create_optimizer(optimizer_type, learning_rate):
    """
    Creates the specified optimizer with given learning rate.
    """
    if optimizer_type == 'adam':
        return optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        return optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

def build_model(input_dim, params):
    """
    Builds and returns a neural network model with specified parameters.
    """
    activation_funcs = ['relu', 'sigmoid', 'tanh', 'linear']
    regularizer = create_regularizer(params['kernel_regularizer'], params['reg_strength'])

    model = Sequential([
        Input(shape=(input_dim,)),
        Masking(mask_value=-1)
    ])

    # First layer
    model.add(Dense(int(params['neurons']), kernel_regularizer=regularizer))
    model.add(Activation(activation_funcs[int(params['Activation'])]))
    model.add(Dropout(params['dropout_rate']))

    # Additional layers
    for _ in range(int(params['layers'])):
        model.add(Dense(int(params['neurons']), kernel_regularizer=regularizer))
        model.add(Activation(activation_funcs[int(params['Activation'])]))
        model.add(Dropout(params['dropout_rate']))

    # Output layer
    model.add(Dense(1))

    optimizer = create_optimizer('adam', params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    return model

def optimize_hyperparameters(x_train, y_train):
    """
    Optimizes neural network hyperparameters using Optuna with K-fold CV.
    """
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
            'neurons': trial.suggest_int('neurons', 10, 1000),
            'dropout_rate': trial.suggest_float('dropout_rate', 0, 0.5),
            'layers': trial.suggest_int('layers', 0, 4),
            'patience': trial.suggest_int('patience', 5, 50),
            'batch_size': trial.suggest_int('batch_size', 16, 128),
            'Activation': trial.suggest_int('Activation', 0, 3),
            'kernel_regularizer': trial.suggest_int('kernel_regularizer', 0, 1),
            'reg_strength': trial.suggest_float('reg_strength', 1e-6, 1e-2, log=True)
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        cv_scores = []

        for train_idx, val_idx in kf.split(x_train):
            x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model = build_model(x_train.shape[1], params)
            early_stop = EarlyStopping(monitor='val_loss', patience=params['patience'])

            model.fit(x_train_fold, y_train_fold,
                      validation_data=(x_val_fold, y_val_fold),
                      batch_size=params['batch_size'],
                      epochs=3000,
                      callbacks=[early_stop],
                      verbose=0)

            score = model.evaluate(x_val_fold, y_val_fold, verbose=0)[0]
            cv_scores.append(score)

        return np.mean(cv_scores)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        study_name="creep_snn_kfold"
    )

    study.optimize(
        objective,
        n_trials=20,
        show_progress_bar=True
    )

    return study.best_params, study.best_value

def plot_predictions(y_true, y_pred, title, save_path):
    """Creates prediction vs actual plot with R2 score for research publication."""
    plt.figure(figsize=(12, 12))

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
    plt.plot([-5, 40], [-5, 40], 'r--', label='Perfect prediction', linewidth=3)

    # Calculate and display R2 score in top left
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'),
             verticalalignment='top', horizontalalignment='left', fontsize=26,
             fontfamily='Times New Roman', fontweight='bold', color='black')

    plt.xlim(-5, 40)
    plt.ylim(-5, 40)

    # Set custom tick marks to explicitly show -5 and 40
    plt.xticks([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40], color='black', fontweight='bold')
    plt.yticks([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40], color='black', fontweight='bold')

    # Set labels and title
    plt.xlabel('ln(Measured creep rupture life) (h)', color='black', fontweight='bold')
    plt.ylabel('ln(Predicted creep rupture life) (h)', color='black', fontweight='bold')
    plt.title(title, pad=20, color='black', fontweight='bold')

    # Add legend in bottom right
    plt.legend(loc='lower right', labelcolor='black', prop={'weight': 'bold', 'size': 26})

    # Save figure with high quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='tif')
    plt.close()

def run_analysis(x, y, suffix, feature_names=None):
    """
    Runs the complete analysis pipeline with ANN using KFold for train-test splitting.
    Plots predictions for the fold with R², MAE, and RMSE closest to the mean performance across folds.

    Parameters:
    - x: Input features (DataFrame or NumPy array).
    - y: Target variable (Series or NumPy array).
    - suffix: String suffix for output files and logging.
    - feature_names: List of feature names to use for DataFrame columns (optional).
    """
    # Convert to numpy arrays if they're pandas objects
    x = x.to_numpy() if isinstance(x, pd.DataFrame) else x
    y = y.to_numpy() if isinstance(y, pd.Series) else y

    # Reshape y if needed
    y = y.reshape(-1, 1) if len(y.shape) == 1 else y

    # Initialize KFold for 80/20 train-test splits
    kf = KFold(n_splits=3, shuffle=True, random_state=random_seed)
    fold_results = []
    first_fold_best_params = None
    first_fold_scaler_x = None
    first_fold_scaler_y = None

    logger.info(f"\nRunning KFold analysis for {suffix}...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(x), 1):
        logger.info(f"\nProcessing Fold {fold}...")
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Preprocess data, passing feature names
        x_train_df = pd.DataFrame(x_train, columns=feature_names)
        x_train_preprocessed, training_stats = preprocess_data(x_train_df, is_test=False)
        x_test_df = pd.DataFrame(x_test, columns=feature_names)
        x_test_preprocessed = preprocess_data(x_test_df, is_test=True, training_stats=training_stats)

        # Apply StandardScaler transformation for features
        scaler_x = StandardScaler()
        x_train_transformed = scaler_x.fit_transform(x_train_preprocessed)
        x_test_transformed = scaler_x.transform(x_test_preprocessed)

        # Apply StandardScaler transformation for target
        scaler_y = StandardScaler()
        y_train_transformed = scaler_y.fit_transform(y_train)
        y_test_transformed = scaler_y.transform(y_test)

        # Store scalers for first fold
        if fold == 1:
            first_fold_scaler_x = scaler_x
            first_fold_scaler_y = scaler_y

        # Optimize hyperparameters on training set
        logger.info(f"Optimizing hyperparameters for Fold {fold}...")
        best_params, best_score = optimize_hyperparameters(x_train_transformed, y_train_transformed)

        logger.info(f"Best hyperparameters for Fold {fold}:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")

        # Store best parameters from first fold for final model
        if fold == 1:
            first_fold_best_params = best_params

        # Train model on training set
        model = build_model(x_train_transformed.shape[1], best_params)
        early_stop = EarlyStopping(monitor='val_loss', patience=best_params['patience'])

        model.fit(x_train_transformed, y_train_transformed,
                  validation_split=0.2,
                  batch_size=best_params['batch_size'],
                  epochs=10000,
                  callbacks=[early_stop],
                  verbose=0)

        # Make predictions
        y_train_pred_transformed = model.predict(x_train_transformed, verbose=0)
        y_test_pred_transformed = model.predict(x_test_transformed, verbose=0)

        y_train_pred = scaler_y.inverse_transform(y_train_pred_transformed)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_transformed)

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

        logger.info(f"\nAggregated Metrics for {suffix}:")
        logger.info("Training Set:")
        logger.info(f"  MSE: {aggregated_metrics['train_metrics']['mse_mean']:.4f} ± {aggregated_metrics['train_metrics']['mse_std']:.4f}")
        logger.info(f"  RMSE: {aggregated_metrics['train_metrics']['rmse_mean']:.4f} ± {aggregated_metrics['train_metrics']['rmse_std']:.4f}")
        logger.info(f"  MAE: {aggregated_metrics['train_metrics']['mae_mean']:.4f} ± {aggregated_metrics['train_metrics']['mae_std']:.4f}")
        logger.info(f"  R2: {aggregated_metrics['train_metrics']['r2_mean']:.4f} ± {aggregated_metrics['train_metrics']['r2_std']:.4f}")
        logger.info("Test Set:")
        logger.info(f"  MSE: {aggregated_metrics['test_metrics']['mse_mean']:.4f} ± {aggregated_metrics['test_metrics']['mse_std']:.4f}")
        logger.info(f"  RMSE: {aggregated_metrics['test_metrics']['rmse_mean']:.4f} ± {aggregated_metrics['test_metrics']['rmse_std']:.4f}")
        logger.info(f"  MAE: {aggregated_metrics['test_metrics']['mae_mean']:.4f} ± {aggregated_metrics['test_metrics']['mae_std']:.4f}")
        logger.info(f"  R2: {aggregated_metrics['test_metrics']['r2_mean']:.4f} ± {aggregated_metrics['test_metrics']['r2_std']:.4f}")

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
        logger.info(f"\nFold {closest_fold_num} has R², MAE, and RMSE closest to mean performance:")
        logger.info(f"  Test R²: {closest_fold['test_metrics']['r2']:.4f} (mean: {mean_r2:.4f})")
        logger.info(f"  Test MAE: {closest_fold['test_metrics']['mae']:.4f} (mean: {mean_mae:.4f})")
        logger.info(f"  Test RMSE: {closest_fold['test_metrics']['rmse']:.4f} (mean: {mean_rmse:.4f})")

        # Generate plots for the fold with performance closest to average
        plot_predictions(
            closest_fold['y_train'].ravel(), closest_fold['y_train_pred'].ravel(),
            f'Training set predictions from ANN model ({suffix})',
            f'ANN_train_predictions_{suffix}_fold{closest_fold_num}.tif'
        )
        plot_predictions(
            closest_fold['y_test'].ravel(), closest_fold['y_test_pred'].ravel(),
            f'Test set predictions from ANN model ({suffix})',
            f'ANN_test_predictions_{suffix}_fold{closest_fold_num}.tif'
        )
    else:
        logger.error(f"KFold analysis failed for {suffix}.")
        return None

    # Train final model on full dataset using first fold's best parameters
    x_preprocessed, training_stats = preprocess_data(pd.DataFrame(x, columns=feature_names), is_test=False)
    x_transformed = first_fold_scaler_x.transform(x_preprocessed)
    y_transformed = first_fold_scaler_y.transform(y)

    final_model = build_model(x_transformed.shape[1], first_fold_best_params)
    early_stop = EarlyStopping(monitor='val_loss', patience=first_fold_best_params['patience'])

    try:
        final_model.fit(x_transformed, y_transformed,
                        validation_split=0.2,
                        batch_size=first_fold_best_params['batch_size'],
                        epochs=10000,
                        callbacks=[early_stop],
                        verbose=0)
    except Exception as e:
        logger.error(f"Error in final model fitting: {str(e)}")
        with open('partial_results.json', 'w') as f:
            json.dump({
                'fold_results': fold_results,
                'aggregated_metrics': aggregated_metrics
            }, f, indent=4)
        return None

    return {
        'fold_results': fold_results,
        'aggregated_metrics': aggregated_metrics,
        'final_pipeline': final_model,
        'feature_names': training_stats['feature_names']
    }

if __name__ == "__main__":
    # Read and prepare data
    try:
        Data = pd.read_csv('ML_data_creep_project.csv', header=0)
    except FileNotFoundError as e:
        logger.error(f"Could not find 'ML_data_creep_project.csv'. Ensure the file is in the correct directory. Details: {str(e)}")
        exit(1)
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file. Check file format. Details: {str(e)}")
        exit(1)

    # Validate input data
    if Data.shape[1] < 23:  # Ensure enough columns for index-based selection
        logger.error(f"Dataset has {Data.shape[1]} columns, expected at least 23.")
        exit(1)
    if Data.isna().any().any():
        logger.warning(f"Input data contains NaN values in columns: {Data.columns[Data.isna().any()].tolist()}")
    if not Data.select_dtypes(include=['float64', 'int64']).columns.tolist():
        logger.error("No numeric columns found in input data.")
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
    new_feature_names = Data.columns[selected_cols].tolist()

    # Run analysis for both feature sets
    results_original = run_analysis(x_original, y, "using CALPHAD descriptors", feature_names=feature_names)
    results_new = run_analysis(x_new, y, "excluding CALPHAD descriptors", feature_names=new_feature_names)

    if results_original is None or results_new is None:
        logger.error("Analysis failed for one or both feature sets.")
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

    with open('ANN_model_results_comparison.json', 'w') as f:
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
    plt.savefig('ANN_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("\nAnalysis Complete!")
    logger.info("Results have been saved to 'ANN_model_results_comparison.json'")
    logger.info("Comparison plots have been saved as PNG files")