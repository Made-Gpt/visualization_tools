import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set font and style for English paper
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
# sns.set_style("whitegrid")
sns.set_style("white")

# Create data
data = {
    'Method': ['EmbeddedOcc', 'SplaSCC (Ours)'],
    'Params(M)': [133.604, 133.857],
    'Time(ms)': [127.51, 115.63],
    'Memory(MiB)': [3.464, 3.130],
    'mIoU(%)': [None, None]  # No values shown in the table
}

df = pd.DataFrame(data)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(9, 5))
fig.suptitle('Efficiency Analysis Comparison', fontsize=16, fontweight='bold')

# Color settings 
# colors = ['#3498db', '#e74c3c']  # Blue and red
colors = ['#7EA6E0', '#FFC339']  # Blue and red

# Customizable data ranges and tick settings for each subplot
# You can modify these ranges and number of ticks as needed
params_range = [133.0, 134.0]      # Y-axis range for Parameters
time_range = [110, 130]            # Y-axis range for Time
memory_range = [3.0, 3.5]          # Y-axis range for Memory

# Number of ticks on Y-axis for each subplot
params_ticks = 6                   # Number of ticks for Parameters
time_ticks = 5                     # Number of ticks for Time  
memory_ticks = 6                   # Number of ticks for Memory

# Parameters comparison
ax1 = axes[0]
bars1 = ax1.bar(df['Method'], df['Params(M)'], color=colors, alpha=1, edgecolor='black', linewidth=1)
ax1.set_title('Parameters (M)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Parameters (Million)', fontsize=10)
ax1.set_ylim(params_range[0], params_range[1])
ax1.locator_params(axis='y', nbins=params_ticks)  # Set number of Y-axis ticks
ax1.tick_params(axis='x', rotation=15)

# Add value labels
for i, (bar, value) in enumerate(zip(bars1, df['Params(M)'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{value:.3f}M', ha='center', va='bottom', fontweight='bold')

# Time comparison
ax2 = axes[1]
bars2 = ax2.bar(df['Method'], df['Time(ms)'], color=colors, alpha=1, edgecolor='black', linewidth=1)
ax2.set_title('Inference Time (ms)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Inference Time (ms)', fontsize=10)
ax2.set_ylim(time_range[0], time_range[1])
ax2.locator_params(axis='y', nbins=time_ticks)  # Set number of Y-axis ticks
ax2.tick_params(axis='x', rotation=15)

# Add value labels
for i, (bar, value) in enumerate(zip(bars2, df['Time(ms)'])):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{value:.2f}ms', ha='center', va='bottom', fontweight='bold')

# Memory usage comparison
ax3 = axes[2]
bars3 = ax3.bar(df['Method'], df['Memory(MiB)'], color=colors, alpha=1, edgecolor='black', linewidth=1)
ax3.set_title('Memory Usage (MiB)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Memory Usage (MiB)', fontsize=10)
ax3.set_ylim(memory_range[0], memory_range[1])
ax3.locator_params(axis='y', nbins=memory_ticks)  # Set number of Y-axis ticks
ax3.tick_params(axis='x', rotation=15)

# Add value labels
for i, (bar, value) in enumerate(zip(bars3, df['Memory(MiB)'])):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0025,
             f'{value:.3f}MiB', ha='center', va='bottom', fontweight='bold')

# Adjust layout
plt.tight_layout() 

# Add improvement information text box
# improvement_text = f"""Improvements:
# • Time reduction: {((127.51-115.63)/127.51*100):.1f}%
# • Memory reduction: {((3.464-3.130)/3.464*100):.1f}%
# • Parameters increase: {((133.857-133.604)/133.604*100):.2f}%"""
# 
# fig.text(0.02, 0.02, improvement_text, fontsize=9, 
#          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

plt.show()

# Print numerical comparison
print("=== Efficiency Analysis Comparison Results ===")
print(f"Parameters - EmbeddedOcc: {df['Params(M)'][0]:.3f}M, SplaSCC: {df['Params(M)'][1]:.3f}M")
print(f"Inference Time - EmbeddedOcc: {df['Time(ms)'][0]:.2f}ms, SplaSCC: {df['Time(ms)'][1]:.2f}ms")
print(f"Memory Usage - EmbeddedOcc: {df['Memory(MiB)'][0]:.3f}MiB, SplaSCC: {df['Memory(MiB)'][1]:.3f}MiB")
print(f"\nImprovements:")
print(f"• Inference time reduction: {((127.51-115.63)/127.51*100):.1f}%")
print(f"• Memory usage reduction: {((3.464-3.130)/3.464*100):.1f}%")