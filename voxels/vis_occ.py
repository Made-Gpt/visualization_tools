import os
os.environ['ETS_TOOLKIT'] = 'qt'
os.environ['QT_API'] = 'pyqt5'

import open3d as o3d
import numpy as np
from mayavi import mlab
from pathlib import Path
from PIL import Image


def remove_white_background(image_path):
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)
    
    # set background as taransparent 
    white_areas = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
    data[white_areas] = [255, 255, 255, 0]  # taransparent
    
    # processed 
    new_img = Image.fromarray(data, 'RGBA')
    new_img.save(image_path.replace('.png', '_t.png'))


def setup_lighting():
    """Setup proper lighting for the scene"""
    # Use simpler lighting approach that works with all mayavi versions
    scene = mlab.gcf().scene
    
    # Enable lighting and set ambient light
    scene.light_manager.light_mode = 'vtk'
    
    # Get the default light and configure it
    if hasattr(scene.light_manager, 'lights') and len(scene.light_manager.lights) > 0:
        light = scene.light_manager.lights[0]
        if hasattr(light, 'intensity'):
            light.intensity = 0.8
        if hasattr(light, 'elevation'):
            light.elevation = 45
        if hasattr(light, 'azimuth'):
            light.azimuth = 45
    
    # Set general lighting parameters
    if hasattr(scene.light_manager, 'ambient'):
        scene.light_manager.ambient = 0.3
    
    print("[INFO] Lighting setup complete with enhanced ambient lighting")

def setup_camera_view(voxel_centers, view_preset="default", custom_params=None, use_parallel=True):
    """Setup camera view for the scene using actual voxel centers"""
    # Calculate scene bounds using actual rendered voxel centers
    min_bounds = np.min(voxel_centers, axis=0)
    max_bounds = np.max(voxel_centers, axis=0)
    # Use the true geometric center without manual adjustments
    center = (min_bounds + max_bounds) / 2
    # center[2] -= 0.025
    # center[2] -= 0.065 
    # center[2] -= 0.1 
    # center[2] -= 0.2 
    # center[2] -= 0.5 
    # center[2] -= 1.0
    # center[2] -= 0.65 
    # center[2] += 0.065  
    # center[2] += 0.1 
    # center[2] += 0.2
    # center[0] += 0.2 
    # center[0] += 0.5
    # center[0] -= 0.2 
    # center[0] -= 0.5
    scene_size = np.max(max_bounds - min_bounds) 

    print(f"[INFO] Scene center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(f"[INFO] Scene size: {scene_size:.2f}")
    print(f"[INFO] Scene bounds: X=[{min_bounds[0]:.2f}, {max_bounds[0]:.2f}], Y=[{min_bounds[1]:.2f}, {max_bounds[1]:.2f}], Z=[{min_bounds[2]:.2f}, {max_bounds[2]:.2f}]")
    
    # Set up parallel projection
    if use_parallel:
        mlab.gcf().scene.camera.parallel_projection = True
        print("[INFO] Parallel projection enabled")
    else:
        mlab.gcf().scene.camera.parallel_projection = False
        print("[INFO] Perspective projection (default)")
    
    if custom_params:
        # Use custom parameters
        azimuth = custom_params.get('azimuth', 45)
        elevation = custom_params.get('elevation', 60)
        focalpoint = custom_params.get('focalpoint', center)
        
        # Set view angle and focal point first
        mlab.view(azimuth=azimuth, elevation=elevation, focalpoint=focalpoint)
        
        # Handle zoom differently for parallel vs perspective projection
        if use_parallel:
            # For parallel projection, use parallel_scale instead of distance
            if 'zoom_factor' in custom_params:
                parallel_scale = scene_size * custom_params['zoom_factor']
            elif 'distance_factor' in custom_params:
                # Convert distance factor to zoom factor (inverse relationship)
                parallel_scale = scene_size / custom_params['distance_factor']
            else:
                parallel_scale = scene_size * 0.8  # Default zoom
            
            mlab.gcf().scene.camera.parallel_scale = parallel_scale
            print(f"[INFO] Custom camera view: azimuth={azimuth}, elevation={elevation}, parallel_scale={parallel_scale:.2f}")
        else:
            # For perspective projection, use distance
            if 'distance_factor' in custom_params:
                distance = scene_size * custom_params['distance_factor']
            else:
                distance = custom_params.get('distance', scene_size * 2) 
             
            mlab.view(azimuth=azimuth, elevation=elevation, 
                     distance=distance, focalpoint=focalpoint)
            print(f"[INFO] Custom camera view: azimuth={azimuth}, elevation={elevation}, distance={distance:.2f}")
        
    elif view_preset == "top":
        # Top-down view
        mlab.view(azimuth=0, elevation=90, focalpoint=center)
        if use_parallel:
            mlab.gcf().scene.camera.parallel_scale = scene_size * 0.6
        else:
            mlab.view(azimuth=0, elevation=90, distance=scene_size * 1.5, focalpoint=center)
        print("[INFO] Camera view: Top-down") 
        
    elif view_preset == "side": 
        # Side view
        mlab.view(azimuth=0, elevation=0, focalpoint=center)
        if use_parallel:
            mlab.gcf().scene.camera.parallel_scale = scene_size * 0.8
        else:
            mlab.view(azimuth=0, elevation=0, distance=scene_size * 2, focalpoint=center)
        print("[INFO] Camera view: Side")

    elif view_preset == "isometric":
        # Isometric view (good for 3D scenes)
        mlab.view(azimuth=45, elevation=35, focalpoint=center)
        if use_parallel:
            mlab.gcf().scene.camera.parallel_scale = scene_size * 0.8
        else:
            mlab.view(azimuth=45, elevation=35, distance=scene_size * 2, focalpoint=center)
        print("[INFO] Camera view: Isometric")
        
    elif view_preset == "bird":
        # Bird's eye view
        mlab.view(azimuth=45, elevation=70, focalpoint=center)
        if use_parallel:
            mlab.gcf().scene.camera.parallel_scale = scene_size * 0.7
        else:
            mlab.view(azimuth=45, elevation=70, distance=scene_size * 1.8, focalpoint=center)
        print("[INFO] Camera view: Bird's eye")

    else: 
        # Default diagonal view - better angle for room scenes
        # mlab.view(azimuth=75, elevation=50, focalpoint=center) 
        # mlab.view(azimuth=75, elevation=75, focalpoint=center) 
        # mlab.view(azimuth=125, elevation=65, focalpoint=center) 
        # mlab.view(azimuth=115, elevation=65, focalpoint=center) 
        # mlab.view(azimuth=65, elevation=65, focalpoint=center) 

        # mlab.view(azimuth=145, elevation=50, focalpoint=center)  # 0000_00/0012 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.48 # Adjust this value to control zoom

        # mlab.view(azimuth=-15, elevation=55, focalpoint=center)  # 0025_00/0082
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.55 # Adjust this value to control zoom 
    
        # center[2] += 0.1
        # mlab.view(azimuth=145, elevation=70, focalpoint=center) # 0168_01/0070 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.50 # Adjust this value to control zoom

        # center[2] += 0.065
        # mlab.view(azimuth=145, elevation=70, focalpoint=center) # 0168_01/0070 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.50 # Adjust this value to control zoom

        # center[2] -= 0.1
        # center[0] += 0.2
        # mlab.view(azimuth=-105, elevation=45, focalpoint=center) # 0168_01/0070 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.50 # Adjust this value to control zoom

        # mlab.view(azimuth=45, elevation=45, focalpoint=center) # 0292_02/0012
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.50 # Adjust this value to control zoom

        # mlab.view(azimuth=45, elevation=45, focalpoint=center) # 0107_00/0033
        # mlab.view(azimuth=155, elevation=60, focalpoint=center) 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom

        # mlab.view(azimuth=-55, elevation=50, focalpoint=center) # 0525_00/0005
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.50 # Adjust this value to control zoom
        
        # center[2] += 0.065  
        # mlab.view(azimuth=255, elevation=70, focalpoint=center) # 0468_00/0020
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom

        # center[2] -= 0.1  
        # mlab.view(azimuth=-10, elevation=45, focalpoint=center) # 0623_00/0059
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.5 # Adjust this value to control zoom

        # center[2] -= 0.65  
        # center[0] += 0.5
        # mlab.view(azimuth=230, elevation=35, focalpoint=center) # 0626_01/0050
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.5 # Adjust this value to control zoom

        # mlab.view(azimuth=-100, elevation=55, focalpoint=center) # 0640_02/0035
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom
        
        # center[2] -= 0.065  # mini
        # center[2] += 0.065  # base
        # mlab.view(azimuth=45, elevation=50, focalpoint=center) # 0706_00/0012
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom

        # center[2] -= 0.1 
        # mlab.view(azimuth=-45, elevation=45, focalpoint=center) # 0673_02/0035
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.50 # Adjust this value to control zoom
    
        # center[2] += 0.2  # mini
        # center[0] += 0.5  # mini
        # mlab.view(azimuth=65, elevation=50, focalpoint=center) # 0056_00/0059
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.38 # Adjust this value to control zoom

        # center[2] -= 1.0  # mini
        # center[0] += 0.5  # mini
        # mlab.view(azimuth=255, elevation=40, focalpoint=center) # 0058_00/0074
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom

        # mlab.view(azimuth=65, elevation=65, focalpoint=center) # 0000_00/0005
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.55 # Adjust this value to control zoom
        
        # center[2] += 0.1  # label 
        # mlab.view(azimuth=-120, elevation=68, focalpoint=center) # 0000_00/0053
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom

        # mlab.view(azimuth=160, elevation=75, focalpoint=center) # 0000_00/0018
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom

        # mlab.view(azimuth=160, elevation=75, focalpoint=center) # 0025_00/0082
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom

        # center[2] -= 0.10
        # center[0] -= 0.15 
        # mlab.view(azimuth=25, elevation=65, focalpoint=center) # 0024_00/0012  
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.40  # Adjust this value to control zoom
 
        # center[2] -= 0.10  
        # center[0] -= 0.15  
        # mlab.view(azimuth=105, elevation=65, focalpoint=center) # 0030_01/0053 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.40  # Adjust this value to control zoom

        # center[2] += 0.45
        # center[0] -= 0.40  
        # mlab.view(azimuth=238, elevation=55, focalpoint=center) # 0031_00/0087 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom
        
        # mlab.view(azimuth=5, elevation=60, focalpoint=center) # 0032_00/0076 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.60 # Adjust this value to control zoom

        # center[2] -= 0.20 
        # mlab.view(azimuth=165, elevation=50, focalpoint=center) # 0038_01/0018
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.40 # Adjust this value to control zoom

        # center[2] += 0.1
        # mlab.view(azimuth=35, elevation=35, focalpoint=center) # 0040_00/0005  
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.50 # Adjust this value to control zoom

        # mlab.view(azimuth=35, elevation=55, focalpoint=center) # 0052_02/0079
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.55 # Adjust this value to control zoom

        # center[2] += 0.1
        # mlab.view(azimuth=228, elevation=45, focalpoint=center) # 0059_00/0056
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.42 # Adjust this value to control zoom

        # center[0] -= 0.15
        # mlab.view(azimuth=-45, elevation=45, focalpoint=center) # 0070_00/0017 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.50  # pred  
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.52  # label

        # center[2] += 0.1 
        # mlab.view(azimuth=160, elevation=45, focalpoint=center) # 0072_00/0005
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.48 
    
        # V1
        # center[2] += 0.2  # pred
        # center[2] += 0.1  # label 
        # center[0] -= 0.15  # label  
        # mlab.view(azimuth=128, elevation=55, focalpoint=center) # 0089_02/0048
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.48 

        # center[2] += 0.2 
        # center[0] += 0.15
        # mlab.view(azimuth=65, elevation=45, focalpoint=center) # 0089_02/0048
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.48 

        # center[2] -= 0.35 
        # center[0] -= 0.30
        # mlab.view(azimuth=-10, elevation=55, focalpoint=center) # 0092_01/0050
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.40

        # mlab.view(azimuth=110, elevation=55, focalpoint=center) # 0111_00/0046
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom

        # center[2] += 0.15 
        # center[0] -= 0.15
        # mlab.view(azimuth=120, elevation=65, focalpoint=center) # 0101_02/0035
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.40

        # center[2] -= 0.05 
        # mlab.view(azimuth=85, elevation=55, focalpoint=center) # 0010_00/0059

        # center[2] += 0.15
        # center[0] -= 0.1 
        # mlab.view(azimuth=170, elevation=55, focalpoint=center)  # 0006_02/0033

        # label
        # center[2] += 0.12
        # mlab.view(azimuth=310, elevation=55, focalpoint=center)  # 0362_01/0070 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.55

        # pred
        # mlab.view(azimuth=310, elevation=55, focalpoint=center)  # 0362_01/0070 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.58
 
        # center[2] -= 0.10  
        # center[0] -= 0.15  
        # mlab.view(azimuth=105, elevation=65, focalpoint=center)  # 0030_01/0053 
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.40
        
        # center[2] -= 0.35  
        # center[0] += 0.35
        # mlab.view(azimuth=145, elevation=50, focalpoint=center) 

        # center[2] -= 0.12
        # center[0] -= 0.15 
        # mlab.view(azimuth=15, elevation=65, focalpoint=center) # 0024_00/0012  

        # center[2] += 0.1  
        # mlab.view(azimuth=160, elevation=65, focalpoint=center) # 0072_00/0005

        # center[2] += 0.15 
        # center[0] -= 0.15
        # mlab.view(azimuth=120, elevation=65, focalpoint=center) # 0101_02/0035
        
        # mlab.view(azimuth=145, elevation=50, focalpoint=center)  # 0000_00/0012  
        # mlab.view(azimuth=75, elevation=50, focalpoint=center)  # 0000_00/0005 
        # mlab.view(azimuth=135, elevation=50, focalpoint=center)  # 0000_00/0005
       
        # center[2] -= 0.20
        # center[0] += 0.50 
        # mlab.view(azimuth=130, elevation=50, focalpoint=center) # 0089_02/0048
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.55 # 0089_02/0012
        
        # center[2] -= 0.10
        # center[0] += 0.50 
        # mlab.view(azimuth=165, elevation=69, focalpoint=center) # 0089_02/0048
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.48  

        # mlab.view(azimuth=-45, elevation=55, focalpoint=center) # 0000_00/0033
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom

        # center[2] += 0.2  # mini  
        # center[0] += 0.5  # mini  
 
        # mlab.view(azimuth=-45, elevation=55, focalpoint=center) # 0000_00/0033
        # mlab.gcf().scene.camera.parallel_scale = scene_size * 0.45 # Adjust this value to control zoom

        center[2] += 0.10
        center[0] += 0.50 
        mlab.view(azimuth=290, elevation=40, focalpoint=center) # 0089_02/0097
        mlab.gcf().scene.camera.parallel_scale = scene_size * 0.60 
        print("[INFO] Camera view: Default diagonal")   
  
def visualize_voxels_with_original_colors(pcd_path, voxel_size=0.08, show_3d=True, save_image=False, output_path=None,
                                         camera_view="default", custom_camera=None, use_parallel=True):
    """Visualize point cloud with original colors and voxelization"""
    # Load point cloud  
    pcd = o3d.io.read_point_cloud(str(pcd_path)) 
    points = np.asarray(pcd.points) 
    # print(f"points shape: {points.shape}") 
    # points = points[:, [0, 2, 1]] * -1  
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
 
    # Create visualization 
    mlab.figure(size=(1600, 1200), bgcolor=(1.0, 1.0, 1.0))  # White background)
    
    # Group voxels by similar colors for better performance
    color_groups = {}
    all_voxel_centers = []  # Store all voxel centers for proper centering
    
    for voxel_idx, data in voxel_dict.items():
        avg_color = np.mean(data['colors'], axis=0).astype(int)
        color_key = tuple(avg_color)
        
        if color_key not in color_groups:
            color_groups[color_key] = []
        
        # Calculate voxel center - this is the key fix
        # Use the center of the voxel grid cell
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
    
    # Setup camera view using actual rendered voxel centers
    setup_camera_view(all_voxel_centers, camera_view, custom_camera, use_parallel)
    
    # Save image if needed
    if save_image and output_path: 
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True) 
        mlab.savefig(str(output_path), size=(1600, 1200))
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

    # 从命令行参数获取路径信息
    if len(sys.argv) != 7:
        print("使用方法: python3 vis.py <pcd_root> <pcd_fold> <pcd_scene> <pcd_name> <pcd_ext> <output_folder>")
        sys.exit(1)
    
    pcd_root = Path(sys.argv[1])
    pcd_fold = sys.argv[2]
    pcd_scene = sys.argv[3]
    pcd_name = sys.argv[4]
    pcd_ext = sys.argv[5]
    output_folder = sys.argv[6]
    
    pcd_file = pcd_root / pcd_fold / pcd_scene / (pcd_name + pcd_ext)
    
    # Ask about projection type
    print("选择投影方式:")
    print("1. 平行投影 (推荐，无透视变形)")
    print("2. 透视投影 (默认)")
    projection_choice = input("输入投影方式 (1 或 2): ").strip()
    use_parallel = projection_choice == "1"
    
    # Ask about camera view
    print("\n选择相机视角:")
    print("1. 默认视角 (对角线)")
    print("2. 俯视图")
    print("3. 侧视图") 
    print("4. 等轴测视图")
    print("5. 鸟瞰图")
    print("6. 自定义视角")
    
    camera_choice = input("输入视角选择 (1-6): ").strip()
    
    camera_view = "default"
    custom_camera = None
    
    if camera_choice == "2":
        camera_view = "top"
    elif camera_choice == "3":
        camera_view = "side"
    elif camera_choice == "4":
        camera_view = "isometric"
    elif camera_choice == "5":
        camera_view = "bird"
    elif camera_choice == "6":
        print("\n自定义相机参数:")
        azimuth = float(input("方位角 (0-360, 推荐45): ") or "45")
        elevation = float(input("仰角 (0-90, 推荐60): ") or "60")
        distance_factor = float(input("距离因子 (1.0-5.0, 推荐2.0): ") or "2.0")

        custom_camera = {
            'azimuth': azimuth,
            'elevation': elevation,
            'distance_factor': distance_factor  # Will be multiplied by scene size
        }
        camera_view = "custom"
    
    # Ask about 3D interface
    show_3d_choice = input("\n需要3D交互界面吗? (y/n): ").strip().lower()
    show_3d = show_3d_choice in ['y', 'yes', '是', 'Y']
    
    # Ask about saving image
    save_choice = input("保存图像吗? (y/n): ").strip().lower()
    save_image = save_choice in ['y', 'yes', '是', 'Y']
    
    # Set up output path for saving image
    output_path = None
    if save_image:
        img_ext = ".png"  # You can change this to .jpg if preferred
        output_dir = pcd_root / output_folder / pcd_scene
        output_path = output_dir / (pcd_name + img_ext)
    
    # Run visualization with original colors
    visualize_voxels_with_original_colors(pcd_file, voxel_size=0.08, 
                                        show_3d=show_3d, save_image=save_image, output_path=output_path,
                                        camera_view=camera_view, custom_camera=custom_camera, use_parallel=use_parallel)