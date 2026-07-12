import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set global font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'

# Data from the table
models = ['LR', 'ANN', 'RF', 'ET', 'SVR', 'KNN', 'XGB', 'LGBM', 'GB']
metrics = ['R2', 'MAE', 'RMSE']
folds = ['Fold1', 'Fold2', 'Fold3']

# Updated data for each metric across folds
r2_data = {
    'LR': [0.751, 0.731, 0.736],
    'ANN': [0.836, 0.822, 0.813],
    'RF': [0.925, 0.947, 0.925],
    'ET': [0.955, 0.903, 0.918],
    'SVR': [0.964, 0.959, 0.947],
    'KNN': [0.968, 0.965, 0.955],
    'XGB': [0.967, 0.966, 0.954],
    'LGBM': [0.975, 0.974, 0.965],
    'GB': [0.976, 0.976, 0.966]
}

mae_data = {
    'LR': [2.364, 2.227, 2.152],
    'ANN': [1.821, 1.726, 1.724],
    'RF': [1.293, 1.024, 1.129],
    'ET': [0.982, 1.359, 1.208],
    'SVR': [0.756, 0.587, 0.776],
    'KNN': [0.662, 0.657, 0.682],
    'XGB': [0.837, 0.802, 0.862],
    'LGBM': [0.632, 0.618, 0.673],
    'GB': [0.622, 0.597, 0.652]
}

rmse_data = {
    'LR': [3.132, 2.95, 2.75],
    'ANN': [2.542, 2.398, 2.315],
    'RF': [1.717, 1.3, 1.467],
    'ET': [1.331, 1.768, 1.532],
    'SVR': [1.187, 1.151, 1.229],
    'KNN': [1.114, 1.061, 1.128],
    'XGB': [1.129, 1.046, 1.137],
    'LGBM': [0.982, 0.9, 0.992],
    'GB': [0.953, 0.878, 0.975]
}

# Define colors for each fold
fold_colors = ['#0066cc', '#cc3311', '#009988']
marker_styles = ['o', 's', 'D']

# Create subplots for each metric with higher resolution
fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True, dpi=300)
fig.suptitle('Model performance across folds excluding CALPHAD descriptors', fontsize=34, fontweight='bold', y=0.92)

# Plot R²
for i, fold in enumerate(folds):
    r2_values = [r2_data[model][i] for model in models]
    axes[0].plot(models, r2_values, marker=marker_styles[i], markersize=10,
                 linestyle='-' if i == 0 else ('--' if i == 1 else '-.'),
                 color=fold_colors[i], linewidth=3.5, label=fold)

axes[0].set_title(r'$R^2$ Score', fontsize=30, fontweight='bold', pad=20)
axes[0].set_ylabel(r'$R^2$', fontsize=28, fontweight='bold', labelpad=15)
axes[0].set_ylim(0.72, 1.01)
axes[0].tick_params(axis='y', labelsize=26, labelcolor='black')
for label in axes[0].get_yticklabels():
    label.set_fontweight('bold')

# Plot MAE (lower is better)
for i, fold in enumerate(folds):
    mae_values = [mae_data[model][i] for model in models]
    axes[1].plot(models, mae_values, marker=marker_styles[i], markersize=10,
                 linestyle='-' if i == 0 else ('--' if i == 1 else '-.'),
                 color=fold_colors[i], linewidth=3.5, label=fold)

axes[1].set_title('Mean Absolute Error (MAE)', fontsize=30, fontweight='bold', pad=20)
axes[1].set_ylabel('MAE', fontsize=28, fontweight='bold', labelpad=15)
axes[1].set_ylim(0.5, 2.5)  # Adjusted for better visualization
axes[1].tick_params(axis='y', labelsize=26, labelcolor='black')
for label in axes[1].get_yticklabels():
    label.set_fontweight('bold')

# Plot RMSE (lower is better)
for i, fold in enumerate(folds):
    rmse_values = [rmse_data[model][i] for model in models]
    axes[2].plot(models, rmse_values, marker=marker_styles[i], markersize=10,
                 linestyle='-' if i == 0 else ('--' if i == 1 else '-.'),
                 color=fold_colors[i], linewidth=3.5, label=fold)

axes[2].set_title('Root Mean Squared Error (RMSE)', fontsize=30, fontweight='bold', pad=20)
axes[2].set_xlabel('Regression models', fontsize=28, fontweight='bold', labelpad=15)
axes[2].set_ylabel('RMSE', fontsize=28, fontweight='bold', labelpad=15)
axes[2].set_ylim(0.8, 3.2)  # Adjusted for better visualization
axes[2].tick_params(axis='y', labelsize=26, labelcolor='black')
for label in axes[2].get_yticklabels():
    label.set_fontweight('bold')

# Customize x-axis for all subplots
for ax in axes:
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=26, fontweight='bold', rotation=45)

    # Create enhanced legend
    legend = ax.legend(
        loc='best',
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        labelspacing=1.5,
        handlelength=3,
        handletextpad=1.0,
        borderpad=1.0,
        prop={'weight': 'bold', 'size': 20}
    )

    # Make the legend border more prominent
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_facecolor('#ffffff')

    # Enhance the subplot appearance
    ax.set_facecolor('#ffffff')

    # Make the axis borders more visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

# Adjust layout to prevent label cutoff and add extra padding
plt.tight_layout(rect=[0, 0, 1, 0.90], pad=3.0)

# Save the plot in multiple formats for different use cases
plt.savefig('Fold_excluding_CALPHAD_model_performance_comparison.tiff', format='tiff', dpi=300, bbox_inches='tight')