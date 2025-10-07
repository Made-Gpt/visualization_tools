import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Create figure with transparent background
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate meshgrid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
 
# Gaussian function: Z = exp(-(X^2 + Y^2))
# This represents N(0, I) distribution
Z = np.exp(-(X**2 + Y**2))
 
# Create the surface plot with gradient colors
surf = ax.plot_surface(X, Y, Z, 
                       cmap=cm.twilight,  # Purple to cyan gradient
                       alpha=0.9,
                       linewidth=0,
                       antialiased=True,
                       edgecolor='none')

# Remove all labels and titles
# ax.set_xlabel('X', fontsize=12)
# ax.set_ylabel('Y', fontsize=12)
# ax.set_zlabel('Density', fontsize=12)
# ax.set_title('3D Gaussian Distribution N(0, I)', fontsize=14, pad=20)

# Remove axis tick labels and numbers
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Set viewing angle
ax.view_init(elev=25, azim=45)

# Hide default grid and axis
ax.grid(False)
ax.set_axis_off()

# Create custom 2D cross axes at the bottom
z_bottom = 0  # Place at bottom of the surface

# X-axis line (horizontal)
ax.plot([-3, 3], [0, 0], [z_bottom, z_bottom], 'k-', linewidth=3, alpha=0.8)
# Y-axis line (vertical) 
ax.plot([0, 0], [-3, 3], [z_bottom, z_bottom], 'k-', linewidth=3, alpha=0.8)

# Add small tick marks on the axes
tick_size = 0.2
for i in range(-3, 4):
    if i != 0:  # Skip center point
        # X-axis ticks
        ax.plot([i, i], [-tick_size, tick_size], [z_bottom, z_bottom], 'k-', linewidth=2, alpha=0.6)
        # Y-axis ticks
        ax.plot([-tick_size, tick_size], [i, i], [z_bottom, z_bottom], 'k-', linewidth=2, alpha=0.6)

# Set transparent background for the entire figure
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Remove colorbar
# fig.colorbar(surf, shrink=0.5, aspect=5)

# Save with transparent background
plt.savefig('gaussian_noise_3d.png', 
            dpi=300, 
            transparent=True, 
            bbox_inches='tight',
            facecolor='none',
            edgecolor='none')

print("Image saved as 'gaussian_noise_3d.png' with transparent background")

# Display the plot
plt.show()