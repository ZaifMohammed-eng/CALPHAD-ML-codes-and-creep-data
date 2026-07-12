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
conditions = ['With CALPHAD', 'Without CALPHAD']

# Data for each metric for Fold2
r2_data = {
    'With CALPHAD': [0.794, 0.838, 0.946, 0.955, 0.968, 0.969, 0.972, 0.976, 0.977],
    'Without CALPHAD': [0.731, 0.822, 0.947, 0.903, 0.959, 0.965, 0.966, 0.974, 0.976]
}

mae_data = {
    'With CALPHAD': [2.054, 1.724, 1.061, 0.936, 0.599, 0.634, 0.7, 0.585, 0.557],
    'Without CALPHAD': [2.227, 1.726, 1.024, 1.359, 0.587, 0.657, 0.802, 0.618, 0.597]
}

rmse_data = {
    'With CALPHAD': [2.581, 2.288, 1.322, 1.194, 1.102, 1.0, 0.949, 0.873, 0.845],
    'Without CALPHAD': [2.95, 2.398, 1.3, 1.768, 1.151, 1.061, 1.046, 0.9, 0.878]
}

# Define colors and markers for each condition (use only two to match conditions)
condition_colors = ['#0066cc', '#cc3311']
marker_styles = ['o', 's']

# Create subplots for each metric with higher resolution
fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True, dpi=300)
fig.suptitle('Model performance for Fold 2 with and without CALPHAD descriptors', fontsize=34, fontweight='bold', y=0.92)

# Plot R²
for i, condition in enumerate(conditions):
    r2_values = r2_data[condition]
    axes[0].plot(models, r2_values, marker=marker_styles[i], markersize=10,
                 linestyle='-' if i == 0 else '--',
                 color=condition_colors[i], linewidth=3.5, label=condition)

axes[0].set_title(r'$R^2$ Score', fontsize=30, fontweight='bold', pad=20)
axes[0].set_ylabel(r'$R^2$', fontsize=28, fontweight='bold', labelpad=15)
axes[0].set_ylim(0.7, 1.01)
axes[0].tick_params(axis='y', labelsize=26, labelcolor='black')
for label in axes[0].get_yticklabels():
    label.set_fontweight('bold')

# Plot MAE (lower is better)
for i, condition in enumerate(conditions):
    mae_values = mae_data[condition]
    axes[1].plot(models, mae_values, marker=marker_styles[i], markersize=10,
                 linestyle='-' if i == 0 else '--',
                 color=condition_colors[i], linewidth=3.5, label=condition)

axes[1].set_title('Mean Absolute Error (MAE)', fontsize=30, fontweight='bold', pad=20)
axes[1].set_ylabel('MAE', fontsize=28, fontweight='bold', labelpad=15)
axes[1].set_ylim(0.5, 2.4)  # Adjusted for better visualization
axes[1].tick_params(axis='y', labelsize=26, labelcolor='black')
for label in axes[1].get_yticklabels():
    label.set_fontweight('bold')

# Plot RMSE (lower is better)
for i, condition in enumerate(conditions):
    rmse_values = rmse_data[condition]
    axes[2].plot(models, rmse_values, marker=marker_styles[i], markersize=10,
                 linestyle='-' if i == 0 else '--',
                 color=condition_colors[i], linewidth=3.5, label=condition)

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
plt.savefig('Fold2_CALPHAD_vs_NoCALPHAD_model_performance.tiff', format='tiff', dpi=300, bbox_inches='tight')