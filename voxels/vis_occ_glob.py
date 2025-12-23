import os
os.environ['ETS_TOOLKIT'] = 'qt'
os.environ['QT_API'] = 'pyqt5'

import open3d as o3d
import numpy as np
from mayavi import mlab
from pathlib import Path
from PIL import Image
import json


def remove_white_background(image_path):
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)
    
    # set background as transparent 
    white_areas = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
    data[white_areas] = [255, 255, 255, 0]  # transparent
    
    # processed 
    new_img = Image.fromarray(data, 'RGBA')
    new_img.save(image_path.replace('.png', '_t.png'))


def load_camera_config(config_path="camera_config_global.json"):
    """Load camera configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[INFO] Loaded camera configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"[WARNING] Config file {config_path} not found, using default settings")
        return None
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in {config_path}, using default settings")
        return None


def get_camera_params(config, scene_name, use_zoom=False):
    """Get camera parameters for specific scene (global view - top down)"""
    if config is None:
        default_params = {
            'azimuth': 0,
            'elevation': 0,
            'parallel_scale_factor': 0.6,
            'center_offset': [0.0, 0.0, 0.0],
            'image_size': [1600, 1600]
        }
        return default_params
    
    # Try to get scene-specific config
    if scene_name in config:
        base_params = config[scene_name].copy()
        
        # If zoom is requested and zoom config exists
        if use_zoom and 'zoom' in base_params:
            print(f"[INFO] Using ZOOM config for {scene_name}")
            zoom_config = base_params['zoom']
            # Use azimuth/elevation from zoom if present, otherwise from base
            result_params = {
                'azimuth': zoom_config.get('azimuth', base_params.get('azimuth', 0)),
                'elevation': zoom_config.get('elevation', base_params.get('elevation', 0)),
                'parallel_scale_factor': zoom_config.get('parallel_scale_factor', 0.3),
                'center_offset': zoom_config.get('center_offset', [0.0, 0.0, 0.0]),
                'image_size': zoom_config.get('image_size', [2400, 2400])
            }
            return result_params
        else:
            if use_zoom:
                print(f"[WARNING] Zoom requested but no zoom config found for {scene_name}, using normal view")
            else:
                print(f"[INFO] Using camera config for {scene_name}")
            # Add default image_size if not present
            if 'image_size' not in base_params:
                base_params['image_size'] = [1600, 1600]
            return base_params
    
    # Fall back to default (top-down view)
    print(f"[INFO] No specific config for {scene_name}, using default top-down view")
    default_params = config.get('default', {
        'azimuth': 0,
        'elevation': 0,
        'parallel_scale_factor': 0.6,
        'center_offset': [0.0, 0.0, 0.0],
        'image_size': [1600, 1600]
    })
    if 'image_size' not in default_params:
        default_params['image_size'] = [1600, 1600]
    return default_params


def setup_camera_view(voxel_centers, camera_params):
    """Setup camera view for the scene using camera parameters from config"""
    # Calculate scene bounds using actual rendered voxel centers
    min_bounds = np.min(voxel_centers, axis=0)
    max_bounds = np.max(voxel_centers, axis=0)
    center = (min_bounds + max_bounds) / 2
    scene_size = np.max(max_bounds - min_bounds)
    
    # Apply center offset from config
    center_offset = camera_params.get('center_offset', [0.0, 0.0, 0.0])
    center[0] += center_offset[0]
    center[1] += center_offset[1]
    center[2] += center_offset[2]

    print(f"[INFO] Scene center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(f"[INFO] Scene size: {scene_size:.2f}")
    print(f"[INFO] Scene bounds: X=[{min_bounds[0]:.2f}, {max_bounds[0]:.2f}], Y=[{min_bounds[1]:.2f}, {max_bounds[1]:.2f}], Z=[{min_bounds[2]:.2f}, {max_bounds[2]:.2f}]")
    
    # Always use parallel projection for top-down view
    mlab.gcf().scene.camera.parallel_projection = True
    print("[INFO] Parallel projection enabled (top-down view)")
    
    # Get camera parameters (default to top-down view)
    azimuth = camera_params.get('azimuth', 0)
    elevation = camera_params.get('elevation', 0)
    parallel_scale_factor = camera_params.get('parallel_scale_factor', 0.6)
    
    # Set view angle and focal point
    mlab.view(azimuth=azimuth, elevation=elevation, focalpoint=center)
    
    # Set parallel scale
    parallel_scale = scene_size * parallel_scale_factor
    mlab.gcf().scene.camera.parallel_scale = parallel_scale
    
    print(f"[INFO] Camera view: azimuth={azimuth}, elevation={elevation}, parallel_scale={parallel_scale:.2f}")   
  

def visualize_voxels_with_original_colors(pcd_path, voxel_size=0.08, show_3d=True, save_image=False, 
                                         output_path=None, camera_params=None, image_size=None):
    """Visualize point cloud with original colors and voxelization"""
    # Load point cloud  
    pcd = o3d.io.read_point_cloud(str(pcd_path)) 
    points = np.asarray(pcd.points) 
    colors = np.asarray(pcd.colors) 
 
    print(f"[INFO] Loaded {points.shape[0]} points from {pcd_path}")

    # Convert colors to 0-255 range
    colors = (colors * 255).astype(np.uint8)

    # Convert coordinates to voxel indices
    voxel_coords = np.floor(points / voxel_size).astype(int)

    # Build voxel dictionary with average colors
    voxel_dict = {}
    for idx, voxel in enumerate(voxel_coords):
        key = tuple(voxel)
        if key not in voxel_dict:
            voxel_dict[key] = {'colors': [colors[idx]], 'count': 1}
        else:
            voxel_dict[key]['colors'].append(colors[idx])
            voxel_dict[key]['count'] += 1
 
    # Get image size from parameters or use default (square for top-down)
    if image_size is None:
        image_size = [1600, 1600]
    
    # Create visualization 
    mlab.figure(size=tuple(image_size), bgcolor=(1.0, 1.0, 1.0))  # White background
    print(f"[INFO] Figure size: {image_size[0]}x{image_size[1]}")
    
    # Group voxels by similar colors for better performance
    color_groups = {}
    all_voxel_centers = []  # Store all voxel centers for proper centering
    
    for voxel_idx, data in voxel_dict.items():
        avg_color = np.mean(data['colors'], axis=0).astype(int)
        color_key = tuple(avg_color)
        
        if color_key not in color_groups:
            color_groups[color_key] = []
        
        # Calculate voxel center
        center = np.array(voxel_idx) * voxel_size + voxel_size / 2
        color_groups[color_key].append(center)
        all_voxel_centers.append(center)

    # Convert to numpy array for easier manipulation
    all_voxel_centers = np.array(all_voxel_centers)
    
    # Print center information for debugging
    actual_center = np.mean(all_voxel_centers, axis=0)
    print(f"[INFO] Actual voxel center: [{actual_center[0]:.2f}, {actual_center[1]:.2f}, {actual_center[2]:.2f}]")
    
    # Render each color group
    for color_rgb, centers in color_groups.items():
        centers = np.array(centers)
        color_normalized = np.array(color_rgb) / 255.0
        
        mlab.points3d(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            scale_factor=voxel_size * 1.0, 
            mode="cube",
            color=tuple(color_normalized),
            opacity=1.0,
            resolution=8,
        )

    print(f"[INFO] Rendered {len(all_voxel_centers)} voxels in {len(color_groups)} color groups")
    
    # Setup camera view using camera parameters (top-down)
    if camera_params is None:
        camera_params = {
            'azimuth': 0, 
            'elevation': 0, 
            'parallel_scale_factor': 0.6, 
            'center_offset': [0.0, 0.0, 0.0],
            'image_size': [1600, 1600]
        }
    setup_camera_view(all_voxel_centers, camera_params)
    
    # Save image if needed
    if save_image and output_path: 
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if image_size is None:
            image_size = [1600, 1600]
        mlab.savefig(str(output_path), size=tuple(image_size))
        remove_white_background(str(output_path))
        print(f"[INFO] Image saved to: {output_path}")
    
    # Show 3D interface if requested
    if show_3d:
        print("[INFO] Showing 3D interactive interface...")
        mlab.show()
    else:
        print("[INFO] Skipping 3D interface, only saving image...")


if __name__ == "__main__":
    import sys

    # Get path info from command line
    if len(sys.argv) != 9: 
        print("Usage: python3 vis_occ_glob.py <pcd_root> <pcd_fold> <pcd_scene> <pcd_name> <pcd_ext> <output_folder> <show_3d> <use_zoom>")
        sys.exit(1)
    
    pcd_root = Path(sys.argv[1])
    pcd_fold = sys.argv[2]
    pcd_scene = sys.argv[3]
    pcd_name = sys.argv[4]
    pcd_ext = sys.argv[5]
    output_folder = sys.argv[6]
    show_3d = sys.argv[7].lower() in ['true', '1', 'yes', 'y']
    use_zoom = sys.argv[8].lower() in ['true', '1', 'yes', 'y']
    
    # For global view, read frame-specific file but with top-down view
    pcd_file = pcd_root / pcd_fold / pcd_scene / (pcd_name + pcd_ext)
    
    # Load camera configuration
    config_path = Path(__file__).parent / "camera_config_global.json"
    camera_config = load_camera_config(config_path)
    camera_params = get_camera_params(camera_config, pcd_scene, use_zoom)
    
    # Extract image size from camera params
    image_size = camera_params.get('image_size', [1600, 1600])
    
    # Set up output path
    img_ext = ".png"
    output_dir = pcd_root / output_folder / pcd_scene
    # Add suffix to filename if zoom is used
    if use_zoom:
        output_path = output_dir / (pcd_name + "_global_zoom" + img_ext)
    else:
        output_path = output_dir / (pcd_name + "_global" + img_ext)
    
    # Run visualization
    visualize_voxels_with_original_colors(
        pcd_file, 
        voxel_size=0.08, 
        show_3d=show_3d, 
        save_image=True, 
        output_path=output_path,
        camera_params=camera_params,
        image_size=image_size
    )
