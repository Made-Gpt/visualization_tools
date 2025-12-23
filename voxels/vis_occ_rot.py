#!/usr/bin/env python3
"""Rotating Voxel Visualization Tool - Generate rotating animation frames"""

import os
import sys

# Parse offscreen mode from command line first
OFFSCREEN_MODE = len(sys.argv) == 12 and sys.argv[11].lower() in ['true', '1', 'yes']

# Set environment for offscreen rendering
if OFFSCREEN_MODE:
    # Try OSMesa first (preferred for offscreen), fall back to qt with Xvfb
    os.environ['ETS_TOOLKIT'] = 'qt4'
    os.environ['QT_API'] = 'pyqt5'
else:
    os.environ['ETS_TOOLKIT'] = 'qt4'
    os.environ['QT_API'] = 'pyqt5'

import open3d as o3d
import numpy as np
from mayavi import mlab
from pathlib import Path
from PIL import Image
import json


def remove_white_background(image_path):
    """Remove white background and make it transparent"""
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)
    
    # Set white areas as transparent
    white_areas = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
    data[white_areas] = [255, 255, 255, 0]
    
    # Save processed image
    new_img = Image.fromarray(data, 'RGBA')
    new_img.save(image_path.replace('.png', '_t.png'))


def load_camera_config(config_path="camera_config.json"):
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


def get_camera_params(config, scene_name, pcd_name):
    """Get camera parameters for specific scene and pcd"""
    if config is None:
        return {'elevation': 50, 'parallel_scale_factor': 0.5, 'center_offset': [0.0, 0.0, 0.0]}
    
    # Try to get scene-specific config
    if scene_name in config:
        scene_config = config[scene_name]
        if pcd_name in scene_config:
            base_params = scene_config[pcd_name].copy()
            print(f"[INFO] Using camera config for {scene_name}/{pcd_name}")
            if 'image_size' not in base_params:
                base_params['image_size'] = [1600, 1200]
            return base_params
    
    # Fall back to default
    print(f"[INFO] No specific config for {scene_name}/{pcd_name}, using default")
    default_params = config.get('default', {})
    if 'image_size' not in default_params:
        default_params['image_size'] = [1600, 1200]
    return default_params


def setup_camera_view_rotation(voxel_centers, azimuth, camera_params):
    """Setup camera view for the scene with specified azimuth (rotation angle)"""
    # Calculate scene bounds
    min_bounds = np.min(voxel_centers, axis=0)
    max_bounds = np.max(voxel_centers, axis=0)
    center = (min_bounds + max_bounds) / 2
    scene_size = np.max(max_bounds - min_bounds)
    
    # Apply center offset from config
    center_offset = camera_params.get('center_offset', [0.0, 0.0, 0.0])
    center[0] += center_offset[0]
    center[1] += center_offset[1]
    center[2] += center_offset[2]

    # Always use parallel projection
    mlab.gcf().scene.camera.parallel_projection = True
    
    # Get camera parameters (elevation and scale)
    elevation = camera_params.get('elevation', 50)
    parallel_scale_factor = camera_params.get('parallel_scale_factor', 0.5)
    
    # Set view angle with rotating azimuth
    mlab.view(azimuth=azimuth, elevation=elevation, focalpoint=center)
    
    # Set parallel scale
    parallel_scale = scene_size * parallel_scale_factor
    mlab.gcf().scene.camera.parallel_scale = parallel_scale
    
    print(f"[INFO] Camera view: azimuth={azimuth:.1f}, elevation={elevation}, parallel_scale={parallel_scale:.2f}")


def visualize_voxels_rotation(pcd_path, voxel_size=0.08, output_dir=None, 
                              camera_params=None, image_size=None,
                              num_frames=36, start_angle=0, end_angle=360,
                              offscreen=False):
    """Generate rotating visualization frames"""
    
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
 
    # Get image size
    if image_size is None:
        image_size = [1600, 1200]
    
    # Group voxels by color
    color_groups = {}
    all_voxel_centers = []
    
    for voxel_idx, data in voxel_dict.items():
        avg_color = np.mean(data['colors'], axis=0).astype(int)
        color_key = tuple(avg_color)
        
        if color_key not in color_groups:
            color_groups[color_key] = []
        
        center = np.array(voxel_idx) * voxel_size + voxel_size / 2
        color_groups[color_key].append(center)
        all_voxel_centers.append(center)

    all_voxel_centers = np.array(all_voxel_centers)
    
    print(f"[INFO] Total voxels: {len(all_voxel_centers)}, Color groups: {len(color_groups)}")
    print(f"[INFO] Generating {num_frames} rotation frames from {start_angle}° to {end_angle}°")
    
    # Note: offscreen mode is handled by xvfb-run in the shell script
    # Don't use mlab.options.offscreen as it causes segfault
    
    # Generate rotation frames
    angles = np.linspace(start_angle, end_angle, num_frames, endpoint=False)
    
    for frame_idx, azimuth in enumerate(angles):
        # Create new figure for each frame
        fig = mlab.figure(size=tuple(image_size), bgcolor=(1.0, 1.0, 1.0))
        
        # Render voxels
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
        
        # Setup camera with current rotation angle
        setup_camera_view_rotation(all_voxel_centers, azimuth, camera_params)
        
        # Save frame
        output_path = output_dir / f"frame_{frame_idx:04d}.png"
        mlab.savefig(str(output_path), size=tuple(image_size))
        
        # Remove white background
        remove_white_background(str(output_path))
        
        print(f"[INFO] Saved frame {frame_idx+1}/{num_frames}: {output_path.name} (azimuth={azimuth:.1f}°)")
        
        # Close figure to free memory
        mlab.close(all=True)
    
    print(f"[INFO] ✅ All {num_frames} frames saved to {output_dir}")


if __name__ == "__main__":
    import sys

    # Get path info from command line
    if len(sys.argv) != 12:
        print("Usage: python3 vis_occ_rot.py <pcd_root> <pcd_fold> <pcd_scene> <pcd_name> <pcd_ext> <output_folder> <num_frames> <start_angle> <end_angle> <image_size_width> <offscreen>")
        sys.exit(1)
    
    pcd_root = Path(sys.argv[1])
    pcd_fold = sys.argv[2]
    pcd_scene = sys.argv[3]
    pcd_name = sys.argv[4]
    pcd_ext = sys.argv[5]
    output_folder = sys.argv[6]
    num_frames = int(sys.argv[7])
    start_angle = float(sys.argv[8])
    end_angle = float(sys.argv[9])
    image_width = int(sys.argv[10])
    offscreen = sys.argv[11].lower() in ['true', '1', 'yes']
    
    # Calculate height for 4:3 aspect ratio
    image_height = int(image_width * 3 / 4)
    image_size = [image_width, image_height]
    
    pcd_file = pcd_root / pcd_fold / pcd_scene / (pcd_name + pcd_ext)
    
    # Load camera configuration
    config_path = Path(__file__).parent / "camera_config.json"
    camera_config = load_camera_config(config_path)
    camera_params = get_camera_params(camera_config, pcd_scene, pcd_name)
    
    # Override image size
    camera_params['image_size'] = image_size
    
    # Set up output directory
    output_dir = pcd_root / output_folder / pcd_scene / (pcd_name + "_rotation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run visualization
    visualize_voxels_rotation(
        pcd_file, 
        voxel_size=0.08, 
        output_dir=output_dir,
        camera_params=camera_params,
        image_size=image_size,
        num_frames=num_frames,
        start_angle=start_angle,
        end_angle=end_angle,
        offscreen=offscreen
    )
