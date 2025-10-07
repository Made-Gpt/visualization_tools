import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageChops
from pathlib import Path
import colorsys

 
def center_crop_by_ratio(image_path, ratio=0.9):
    """
    Forcefully crop the image from the center based on a given ratio.
    For example, ratio=0.9 means the output image will be 90% of the original height and width.
    
    Args:
        image_path (str): Path to the input image file (will be overwritten).
        ratio (float): Fraction to keep (0 < ratio <= 1).
    """
    assert 0 < ratio <= 1, "Ratio must be between 0 and 1."

    image = Image.open(image_path)
    width, height = image.size

    new_width = int(width * ratio)
    new_height = int(height * ratio)

    left = (width - new_width) // 2
    upper = (height - new_height) // 2
    right = left + new_width
    lower = upper + new_height

    cropped_image = image.crop((left, upper, right, lower))
    cropped_image.save(image_path)
    print(f"Cropped image saved. New size: {new_width} x {new_height}")


def enhance_colors_hsv(colors, brightness_factor=1.5, saturation_factor=1.0):
    """
    Enhance colors using HSV color space to increase brightness while preserving saturation
    
    Args:
        colors (np.ndarray): RGB colors in [0,1] range
        brightness_factor (float): Factor to increase brightness
    
    Returns:
        np.ndarray: Enhanced colors
    """
    enhanced_colors = np.zeros_like(colors)
    
    for i in range(len(colors)):
        # Convert to HSV color space
        h, s, v = colorsys.rgb_to_hsv(colors[i, 0], colors[i, 1], colors[i, 2])
        
        # Only increase brightness (V), keep hue (H) and saturation (S)
        s = min(s * saturation_factor, 1.0)
        v = min(v * brightness_factor, 1.0)
        
        # Convert back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        enhanced_colors[i] = [r, g, b]
    
    return enhanced_colors


def enhance_colors_gamma(colors, gamma=0.7):
    """
    Enhance colors using gamma correction to increase brightness
    
    Args:
        colors (np.ndarray): RGB colors in [0,1] range
        gamma (float): Gamma value, < 1.0 increases brightness
    
    Returns:
        np.ndarray: Enhanced colors
    """
    return np.power(colors, gamma)


def enhance_colors_log(colors, scale=1.2):
    """
    Enhance colors using logarithmic transformation
    
    Args:
        colors (np.ndarray): RGB colors in [0,1] range
        scale (float): Scale factor for enhancement
    
    Returns:
        np.ndarray: Enhanced colors
    """
    # Avoid log(0)
    colors_safe = np.clip(colors, 1e-8, 1.0)
    enhanced = np.log(1 + colors_safe * scale) / np.log(1 + scale)
    return np.clip(enhanced, 0, 1)


def enhance_colors_adaptive(colors, target_brightness=0.6):
    """
    Adaptively enhance colors based on original brightness
    
    Args:
        colors (np.ndarray): RGB colors in [0,1] range
        target_brightness (float): Target brightness level
    
    Returns:
        np.ndarray: Enhanced colors
    """
    # Calculate perceptual brightness for each pixel
    brightness = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    
    # Calculate enhancement factor
    enhancement_factor = np.where(brightness > 0, 
                                 np.minimum(target_brightness / brightness, 3.0),
                                 1.0)
    
    # Apply enhancement
    enhanced_colors = colors * enhancement_factor[:, np.newaxis]
    return np.clip(enhanced_colors, 0, 1)


class GaussianEllipsoidsVisualizer:
    """Efficient Gaussian ellipsoids visualizer"""
    
    def __init__(self, ply_file_path):
        """
        Initialize the visualizer
        Args:
            ply_file_path: Path to the .ply file
        """
        self.ply_file_path = ply_file_path
        self.load_ply_data()
    
    def load_ply_data(self): 
        """Load Gaussian ellipsoid data from .ply file"""
        print(f"Loading PLY file: {self.ply_file_path}")
        plydata = PlyData.read(self.ply_file_path)
        vertex = plydata['vertex']
        
        # Extract attributes and apply coordinate transformation
        xyz = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
        self.xyz = xyz[:, [0, 2, 1]] * -1  # [-1, 1, -1] 
        # self.xyz[:, 2] += 0.10

        self.opacity = vertex['opacity']
        self.colors = np.column_stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']])
        self.scales = np.column_stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']])
        self.rotations = np.column_stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']])
        
        # Convert log scale back to original scale
        self.scales = np.exp(self.scales)
        # Convert logit opacity back to original opacity
        self.opacity = 1 / (1 + np.exp(-self.opacity)) 

        # Normalize colors to [0,1] range 
        self.colors = np.abs(self.colors) / (np.sqrt(1 * np.pi))
        self.colors = np.clip(self.colors, 0, 1)
        
        # Enhanced color processing - choose one method:
        # Method 1: HSV enhancement (recommended)
        self.colors = enhance_colors_hsv(self.colors, brightness_factor=1.5)
        
        # Method 2: Gamma correction (alternative)
        # self.colors = enhance_colors_gamma(self.colors, gamma=0.7)
        
        # Method 3: Logarithmic transformation (alternative)
        # self.colors = enhance_colors_log(self.colors, scale=1.2)
        
        # Method 4: Adaptive enhancement (alternative)
        # self.colors = enhance_colors_adaptive(self.colors, target_brightness=0.6)
        
        print(f"Successfully loaded {len(self.xyz)} Gaussian ellipsoids")
        print(f"Position range: X[{self.xyz[:,0].min():.2f}, {self.xyz[:,0].max():.2f}], "
              f"Y[{self.xyz[:,1].min():.2f}, {self.xyz[:,1].max():.2f}], "
              f"Z[{self.xyz[:,2].min():.2f}, {self.xyz[:,2].max():.2f}]")
    
    def filter_and_sample(self, max_points, alpha_threshold=0.01):
        """Filter and sample data""" 
        # Filter points with low opacity
        valid_mask = self.opacity.flatten() > alpha_threshold
        
        xyz_filtered = self.xyz[valid_mask]
        colors_filtered = self.colors[valid_mask]
        scales_filtered = self.scales[valid_mask]
        rotations_filtered = self.rotations[valid_mask]
        opacity_filtered = self.opacity[valid_mask]
        
        print(f"After filtering: {len(xyz_filtered)} ellipsoids remaining (opacity threshold: {alpha_threshold})")
        
        # Random sampling if too many points
        if len(xyz_filtered) > max_points:
            indices = np.random.choice(len(xyz_filtered), max_points, replace=False)
            xyz_filtered = xyz_filtered[indices]
            colors_filtered = colors_filtered[indices]
            scales_filtered = scales_filtered[indices]
            rotations_filtered = rotations_filtered[indices]
            opacity_filtered = opacity_filtered[indices]
            print(f"Randomly sampled to {max_points} ellipsoids")
        
        return xyz_filtered, colors_filtered, scales_filtered, rotations_filtered, opacity_filtered

    def create_ellipsoid_surface(self, center, scale, rotation, resolution=12):
        """Create ellipsoid surface points"""
        # Create unit sphere points
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

        # Scale to ellipsoid
        x_ellipsoid = x_sphere * scale[0]
        y_ellipsoid = y_sphere * scale[1]
        z_ellipsoid = z_sphere * scale[2]
        
        # Apply quaternion rotation
        rot_matrix = R.from_quat(rotation).as_matrix()
        
        # Batch process all points
        points = np.stack([x_ellipsoid.flatten(), y_ellipsoid.flatten(), z_ellipsoid.flatten()], axis=0)
        rotated_points = rot_matrix @ points
        
        # Reshape and translate to center position
        x_final = rotated_points[0].reshape(x_ellipsoid.shape) + center[0]
        y_final = rotated_points[1].reshape(y_ellipsoid.shape) + center[1]
        z_final = rotated_points[2].reshape(z_ellipsoid.shape) + center[2]
        
        return x_final, y_final, z_final
    
    def visualize_3d(self, max_ellipsoids=1000, alpha_threshold=0.01, 
                     use_wireframe=True, ellipsoid_resolution=12,
                     figsize=(15, 15), output_path=None):
        """ 
        3D ellipsoid visualization
        Args:
            max_ellipsoids: Maximum number of ellipsoids to display
            alpha_threshold: Opacity threshold
            use_wireframe: Whether to use wireframe mode
            ellipsoid_resolution: Ellipsoid resolution
            figsize: Figure size
            output_path: Output file path
        """
        print("Starting 3D ellipsoid visualization...")
        
        xyz, colors, scales, rotations, opacity = self.filter_and_sample(max_ellipsoids, alpha_threshold)
        
        # fig = plt.figure(figsize=figsize) 
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        light_source = LightSource(azdeg=315, altdeg=45)
        
        # Create visualization for each ellipsoid
        print(f"Rendering {len(xyz)} ellipsoids...")
        for i in range(len(xyz)):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(xyz)}")
                
            center = xyz[i]
            scale = scales[i]
            rotation = rotations[i] 
            # alpha_val = np.clip(opacity[i], alpha_threshold * 100, 1) 
            alpha_val = np.clip(opacity[i], alpha_threshold * 1, 1) 
            color = colors[i] 
            
            # Create ellipsoid surface
            x_ellipsoid, y_ellipsoid, z_ellipsoid = self.create_ellipsoid_surface(
                center, scale, rotation, ellipsoid_resolution)
            
            # Draw wireframe or surface
            if use_wireframe: 
                ax.plot_wireframe(x_ellipsoid, y_ellipsoid, z_ellipsoid, 
                                alpha=alpha_val, color=color, linewidth=0.5)
            
            ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, 
                          color=color, alpha=alpha_val, 
                          shade=True, antialiased=True, lightsource=light_source, 
                          rcount=ellipsoid_resolution, ccount=ellipsoid_resolution)
        
        # Set extremely tight display area to completely eliminate margins
        margin_factor = 0.05 

        data_range_x = xyz[:,0].max() - xyz[:,0].min()
        data_range_y = xyz[:,1].max() - xyz[:,1].min()
        data_range_z = xyz[:,2].max() - xyz[:,2].min()
        
        # Set the tightest possible limits to maximize content and eliminate ALL blank space
        ax.set_xlim(xyz[:,0].min() - data_range_x * margin_factor, 
                    xyz[:,0].max() + data_range_x * margin_factor)
        ax.set_ylim(xyz[:,1].min() - data_range_y * margin_factor, 
                    xyz[:,1].max() + data_range_y * margin_factor)
        ax.set_zlim(xyz[:,2].min() - data_range_z * margin_factor, 
                    xyz[:,2].max() + data_range_z * margin_factor)
        # ax.set_box_aspect([2.0,1.8,1])  
     
        # Set camera viewpoint 
        # origin = np.array([0.0, 3.0, 3.0])  # [0.0, 3.0, 1.8]
        # target = np.array([0.35, -1.0, 1.8])  # [0.0, 0.0, 1.8]
    
        # origin = np.array([0.0, 3.0, 1.8])  # [0.0, 3.0, 1.8] 
        # target = np.array([0.0, 0.0, 1.8])  # [0.0, 0.0, 1.8]

        # origin = np.array([2.8, 3.0, 2.0])  # [0.0, 3.0, 1.8] 
        # target = np.array([1.0, -1.0, 1.8])  # [0.0, 0.0, 1.8]
        
        # origin = np.array([1.6, 3.0, 2.0])  # [0.0, 3.0, 1.8] 
        # target = np.array([1.0, -1.0, 1.8])  # [0.0, 0.0, 1.8]
        
        # scene0003_02/0033
        # origin = np.array([1.0, 3.0, 2.0])  # [0.0, 3.0, 1.8] 
        # target = np.array([1.0, -1.0, 1.8])  # [0.0, 0.0, 1.8]

        # scene0006_02/0033 
        # origin = np.array([0.8, 3.0, 1.5])  # [0.0, 3.0, 1.8] 
        # target = np.array([1.2, -1.0, 1.8])  # [0.0, 0.0, 1.8]
        origin = np.array([2.8, 3.0, 3.5])  # [0.0, 3.0, 1.8] 
        target = np.array([1.2, -1.0, 1.8])  # [0.0, 0.0, 1.8]

        # scene0626_01/0050
        # origin = np.array([1.2, 3.0, 1.5])  # [0.0, 3.0, 1.8] 
        # target = np.array([1.0, -1.0, 1.8])  # [0.0, 0.0, 1.8]

        camera_vec = origin - target
        x, y, z = camera_vec[0], camera_vec[1], camera_vec[2]
        
        azimuth_rad = np.arctan2(y, x)
        azimuth_deg = np.rad2deg(azimuth_rad)
        dist_xy = np.sqrt(x**2 + y**2)
        elevation_rad = np.arctan2(z, dist_xy)
        elevation_deg = np.rad2deg(elevation_rad)
 
        ax.view_init(elev=elevation_deg, azim=azimuth_deg)
        
        # Remove axes completely and set figure to fill entire space
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.axis('off')
        ax.grid(False)

        # Remove ALL background elements to eliminate margins completely
        
        ax.xaxis.pane.fill = True 
        ax.yaxis.pane.fill = True 
        ax.zaxis.pane.fill = True  
        ax.xaxis.pane.set_alpha(0.0) 
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        
        # Make figure use entire space with zero margins
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        fig.tight_layout(pad=0)
        
        # Save image with absolute zero padding to eliminate ALL margins
        if output_path:
            print(f"Saving 3D visualization to: {output_path}")
            plt.savefig(output_path, dpi=300, pad_inches=0,
                       facecolor='none', edgecolor='none', 
                       format='png', pil_kwargs={'optimize': True},  
                       transparent=True) 
            # center_crop_by_ratio(output_path, ratio=0.6) 
            center_crop_by_ratio(output_path, ratio=0.7)
        else:
            plt.show()
        
        plt.close()
        print("3D visualization completed")

 
def main():
    parser = argparse.ArgumentParser(description='Gaussian Ellipsoids Visualization Tool')
    parser.add_argument('--ply_dir', type=str, required=True, help='PLY file directory')
    parser.add_argument('--ply_name', type=str, required=True, help='PLY file name (without extension)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--max_ellipsoids_3d', type=int, default=1000, help='Maximum number of ellipsoids for 3D visualization')
    parser.add_argument('--max_points_2d', type=int, default=3000, help='Maximum number of points for 2D visualization')
    parser.add_argument('--alpha_threshold', type=float, default=0.01, help='Opacity threshold')
    parser.add_argument('--resolution', type=int, default=12, help='Ellipsoid resolution')
    parser.add_argument('--figsize', type=int, nargs=2, default=[15, 15], help='Figure size') 
    parser.add_argument('--projection_plane', type=str, default='xy', choices=['xy', 'xz', 'yz'], 
                       help='2D projection plane')
    
    args = parser.parse_args()
    
    # Construct full PLY file path
    ply_file_path = os.path.join(args.ply_dir, f"{args.ply_name}.ply")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create visualizer
        visualizer = GaussianEllipsoidsVisualizer(ply_file_path)
        
        # 3D visualization
        output_3d = output_dir / f"{args.ply_name}_3d.png"
        visualizer.visualize_3d(
            max_ellipsoids=args.max_ellipsoids_3d,
            alpha_threshold=args.alpha_threshold,
            ellipsoid_resolution=args.resolution,
            use_wireframe=False, 
            figsize=tuple(args.figsize),
            output_path=str(output_3d)
        )
        
        print(f"\nAll visualizations completed! Output files saved in: {output_dir}")
        
    except FileNotFoundError:
        print(f"File not found: {ply_file_path}")
        print("Please ensure the file path is correct!")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main() 
