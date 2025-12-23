import numpy as np
import matplotlib.pyplot as plt

# Set style for better visualization
plt.style.use('seaborn-v0_8-darkgrid')

# Create figure with high DPI for better quality
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

# Generate x values from -3 to 3
x = np.linspace(-3, 3, 500)

# Calculate y = exp(x)
y = np.exp(x)

# Plot the main curve
ax.plot(x, y, linewidth=3, color='#2E86AB', label='y = exp(x)', alpha=0.9)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Mark special points
special_x = np.array([-3, -2, -1, 0, 1, 2, 3])
special_y = np.exp(special_x)

# Plot special points
ax.scatter(special_x, special_y, s=100, c='#A23B72', zorder=5, 
           edgecolors='white', linewidth=2, label='Key Points')

# Add annotations for key points
for i, (xi, yi) in enumerate(zip(special_x, special_y)):
    if yi < 15:  # Only annotate points that fit well in the plot
        ax.annotate(f'({xi:.0f}, {yi:.2f})', 
                   xy=(xi, yi), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                 color='gray', lw=1.5))

# Add horizontal line at y=1
ax.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='y = 1')

# Add vertical line at x=0
ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='x = 0')

# Set labels and title
ax.set_xlabel('Input (x)', fontsize=14, fontweight='bold')
ax.set_ylabel('Output (y = exp(x))', fontsize=14, fontweight='bold')
ax.set_title('Exponential Function: y = exp(x)', fontsize=16, fontweight='bold', pad=20)

# Set axis limits
ax.set_xlim(-3, 3)
ax.set_ylim(0, 22)

# Add legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Add text box with key insights
textstr = '\n'.join([
    'Key Observations:',
    '• exp(0) = 1 (reference point)',
    '• exp(-3) ≈ 0.05 (nearly zero)',
    '• exp(3) ≈ 20.09 (rapid growth)',
    '• Function always positive',
    '• Steeper growth for x > 0'
])

props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Tight layout
plt.tight_layout()

# Save the figure
plt.savefig('exp_function_curve.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

print("\n" + "="*60)
print("Detailed Value Table:")
print("="*60)
print(f"{'Input (x)':<12} {'Output (exp(x))':<20} {'Change Rate':<15}")
print("-"*60)

for i in range(len(special_x)):
    if i > 0:
        rate = (special_y[i] - special_y[i-1]) / (special_x[i] - special_x[i-1])
        print(f"{special_x[i]:>8.1f}     {special_y[i]:>15.4f}     {rate:>12.4f}")
    else:
        print(f"{special_x[i]:>8.1f}     {special_y[i]:>15.4f}     {'N/A':>12}")

print("="*60)
