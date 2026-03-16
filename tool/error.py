import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # 导入seaborn以使用其强大的调色板功能

# --- Data Section ---

# This data represents the percentage of samples with prediction error ≤ 15 dB
plt.rcParams['font.sans-serif'] = 'Arial'
data_internal_no_implant = [60, 72, 78, 82, 74, 74, 85]
data_external_no_implant = [65, 50, 45, 45, 45, 30, 45]
data_internal_with_implant = [74, 69, 74, 74, 74, 83, 69]
data_external_with_implant = [54, 42, 50, 54, 62, 58, 42]

# Consolidate all data and labels for plotting
all_data = [
    data_internal_no_implant,
    data_external_no_implant,
    data_internal_with_implant,
    data_external_with_implant
]
cohort_labels = [
    'Internal (Without OCR)',
    'External (Without OCR)',
    'Internal (With OCR)',
    'External (With OCR)'
]

# --- Plotting Section ---

# Create a sequential blue color palette inspired by your provided code.
# "Blues_r" creates a palette from dark to light blue.
colors = sns.color_palette("Blues_r", n_colors=4) 

# X-axis categories
categories = ['AC-0.25kHz', 'AC-0.5kHz', 'AC-1kHz', 'AC-2kHz', 'AC-4kHz', 'AC-8kHz', 'AC-PTA']
x = np.arange(len(categories))
n_cohorts = len(all_data)
bar_width = 0.20

# Create a single, wider figure
fig, ax = plt.subplots(figsize=(16, 8))

# Loop to plot the bars for each cohort
for i in range(n_cohorts):
    # Calculate the offset for each group of bars to center them
    offset = (i - (n_cohorts - 1) / 2) * bar_width
    positions = x + offset
    bars = ax.bar(positions, all_data[i], bar_width, label=cohort_labels[i], color=colors[i], zorder=3)
    
    # Add numerical labels on top of each bar
    ax.bar_label(bars, padding=3, fontsize=8.5, fmt='%.1f')

# --- Styling the Chart (Nature Style) ---

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set titles and labels in English
ax.set_title('Model Performance: Samples with Prediction Error within ±15 dB', fontsize=18, pad=20)
ax.set_ylabel('Percentage of Samples (%)', fontsize=12)
ax.set_ylim(0, 115) # Add a little space at the top
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)

# Add subtle reference lines and a light grid
ax.axhline(y=80, color='dimgray', linestyle='--', linewidth=1)
# ax.axhline(y=90, color='darkgray', linestyle='--', linewidth=1)
ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.2, zorder=0)
ax.set_axisbelow(True)

# Format the legend
ax.legend(title='Validation Cohort', frameon=False, fontsize=10, title_fontsize=11)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()