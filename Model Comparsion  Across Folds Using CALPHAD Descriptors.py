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

# Data for each metric across folds
r2_data = {
    'LR': [0.808, 0.794, 0.769], 'ANN': [0.863, 0.838, 0.882], 'RF': [0.939, 0.946, 0.916],
    'ET': [0.952, 0.955, 0.917], 'SVR': [0.964, 0.968, 0.948], 'KNN': [0.969, 0.969, 0.954],
    'XGB': [0.97, 0.972, 0.955], 'LGBM': [0.977, 0.976, 0.966], 'GB': [0.978, 0.977, 0.967]
}

mae_data = {
    'LR': [2.104, 2.054, 2.032], 'ANN': [1.74, 1.724, 1.268], 'RF': [1.111, 1.061, 1.199],
    'ET': [0.993, 0.936, 1.191], 'SVR': [0.708, 0.599, 0.764], 'KNN': [0.656, 0.634, 0.675],
    'XGB': [0.76, 0.7, 0.838], 'LGBM': [0.599, 0.585, 0.659], 'GB': [0.594, 0.557, 0.633]
}

rmse_data = {
    'LR': [2.749, 2.581, 2.572], 'ANN': [2.323, 2.288, 1.836], 'RF': [1.54, 1.322, 1.547],
    'ET': [1.367, 1.194, 1.539], 'SVR': [1.187, 1.102, 1.22], 'KNN': [1.099, 1, 1.139],
    'XGB': [1.079, 0.949, 1.134], 'LGBM': [0.933, 0.873, 0.978], 'GB': [0.929, 0.845, 0.976]
}

# Define colors for each fold
fold_colors = ['#0066cc', '#cc3311', '#009988']
marker_styles = ['o', 's', 'D']

# Create subplots for each metric
fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True, dpi=300)
fig.suptitle('Model performance across folds using CALPHAD descriptors', fontsize=34, fontweight='bold', y=0.92)

# Plot R²
for i, fold in enumerate(folds):
    r2_values = [r2_data[model][i] for model in models]
    axes[0].plot(models, r2_values, marker=marker_styles[i], markersize=10,
                 linestyle='-' if i == 0 else ('--' if i == 1 else '-.'),
                 color=fold_colors[i], linewidth=3.5, label=fold)

axes[0].set_title(r'$R^2$ Score', fontsize=30, fontweight='bold', pad=20)
axes[0].set_ylabel(r'$R^2$', fontsize=28, fontweight='bold', labelpad=15)
axes[0].set_ylim(0.75, 1.01)
axes[0].tick_params(axis='y', labelsize=26, labelcolor='black')
for label in axes[0].get_yticklabels():
    label.set_fontweight('bold')

# Plot MAE
for i, fold in enumerate(folds):
    mae_values = [mae_data[model][i] for model in models]
    axes[1].plot(models, mae_values, marker=marker_styles[i], markersize=10,
                 linestyle='-' if i == 0 else ('--' if i == 1 else '-.'),
                 color=fold_colors[i], linewidth=3.5, label=fold)

axes[1].set_title('Mean Absolute Error (MAE)', fontsize=30, fontweight='bold', pad=20)
axes[1].set_ylabel('MAE', fontsize=28, fontweight='bold', labelpad=15)
axes[1].set_ylim(0.5, 2.2)
axes[1].tick_params(axis='y', labelsize=26, labelcolor='black')
for label in axes[1].get_yticklabels():
    label.set_fontweight('bold')

# Plot RMSE
for i, fold in enumerate(folds):
    rmse_values = [rmse_data[model][i] for model in models]
    axes[2].plot(models, rmse_values, marker=marker_styles[i], markersize=10,
                 linestyle='-' if i == 0 else ('--' if i == 1 else '-.'),
                 color=fold_colors[i], linewidth=3.5, label=fold)

axes[2].set_title('Root Mean Squared Error (RMSE)', fontsize=30, fontweight='bold', pad=20)
axes[2].set_xlabel('Regression models', fontsize=28, fontweight='bold', labelpad=15)
axes[2].set_ylabel('RMSE', fontsize=28, fontweight='bold', labelpad=15)
axes[2].set_ylim(0.8, 3.0)
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

    # Set white background
    ax.set_facecolor('#ffffff')

    # Make the axis borders more visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

# Adjust layout to prevent label cutoff and add extra padding
plt.tight_layout(rect=[0, 0, 1, 0.90], pad=3.0)

# Save the plot
plt.savefig('Fold_CALPHAD_model_performance_comparison.tiff', format='tiff', dpi=300, bbox_inches='tight')