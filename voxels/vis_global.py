import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path


def load_point_cloud(pcd_path):
    """Load point cloud and return points and colors as numpy arrays"""
    try:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        if len(points) == 0:
            print(f"Warning: Empty point cloud: {pcd_path}")
            return None, None
            
        print(f"Loaded {len(points)} points from {pcd_path.name}")
        return points, colors
        
    except Exception as e:
        print(f"Error loading {pcd_path}: {e}")
        return None, None


def save_point_cloud(points, colors, output_path):
    """Save concatenated points and colors to a new PLY file"""
    try:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PLY file
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"Saved concatenated point cloud ({len(points)} points) to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")
        return False


def concatenate_frames_progressively(pcd_root, pcd_fold, pcd_scene, pcd_names, pcd_ext, output_folder):
    """
    Progressively concatenate point cloud frames
    
    Args:
        pcd_root: Root directory path
        pcd_fold: Point cloud folder name
        pcd_scene: Scene name
        pcd_names: List of frame names to concatenate
        pcd_ext: File extension
        output_folder: Output folder name
    """
    print(f"=== Progressive Point Cloud Concatenation ===")
    print(f"Scene: {pcd_scene}")
    print(f"Input folder: {pcd_fold}")
    print(f"Output folder: {output_folder}")
    print(f"Frames to process: {pcd_names}")
    print("=" * 50)
    
    # Setup paths
    input_dir = Path(pcd_root) / pcd_fold / pcd_scene
    output_dir = Path(pcd_root) / output_folder / pcd_scene
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return False
    
    # Initialize accumulation variables
    accumulated_points = None
    accumulated_colors = None
    successful_frames = []
    
    # Process frames progressively
    for i, frame_name in enumerate(pcd_names):
        frame_path = input_dir / (frame_name + pcd_ext)
        
        print(f"\nProcessing frame {i+1}/{len(pcd_names)}: {frame_name}")
        
        # Check if frame file exists
        if not frame_path.exists():
            print(f"Warning: Frame file does not exist: {frame_path}")
            continue
        
        # Load current frame
        points, colors = load_point_cloud(frame_path)
        
        if points is None or colors is None:
            print(f"Warning: Failed to load frame: {frame_name}")
            continue
        
        # Add to accumulation
        if accumulated_points is None:
            # First valid frame
            accumulated_points = points.copy()
            accumulated_colors = colors.copy()
        else:
            # Concatenate with existing data
            accumulated_points = np.concatenate([accumulated_points, points], axis=0)
            accumulated_colors = np.concatenate([accumulated_colors, colors], axis=0)
        
        successful_frames.append(frame_name)
        
        # Save current accumulated result
        # 使用当前帧名作为输出文件名（表示累积到这一帧）
        output_name = frame_name  # 改为使用帧名
        output_path = output_dir / (output_name + pcd_ext)
        
        success = save_point_cloud(accumulated_points, accumulated_colors, output_path)
        
        if success:
            print(f"✓ Accumulated {len(successful_frames)} frames -> {output_name}{pcd_ext}")
            print(f"  Total points: {len(accumulated_points)}")
            print(f"  Frames included: {successful_frames}")
        else:
            print(f"✗ Failed to save accumulated result: {output_name}{pcd_ext}")
    
    print(f"\n=== Concatenation Complete ===")
    print(f"Successfully processed: {len(successful_frames)}/{len(pcd_names)} frames")
    print(f"Output directory: {output_dir}")
    print(f"Generated files: {[frame + pcd_ext for frame in successful_frames]}")
    
    return len(successful_frames) > 0
  
def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 8: 
        print("Usage: python3 vis_global.py <pcd_root> <pcd_fold> <pcd_scene> <output_folder> <pcd_ext> <frame1> <frame2> [...]")
        print("Example: python3 vis_global.py /path/to/root vis_occ_da_label scene0000_00 vis_label_global .ply pcd_00004 pcd_00005 pcd_00012")
        sys.exit(1)

    # Parse command line arguments
    pcd_root = sys.argv[1]
    pcd_fold = sys.argv[2]
    pcd_scene = sys.argv[3]
    output_folder = sys.argv[4]
    pcd_ext = sys.argv[5]
    
    # Remaining arguments are frame names
    pcd_names = sys.argv[6:]
    
    if len(pcd_names) < 2:
        print("Error: At least 2 frames are required for concatenation")
        sys.exit(1)
    
    # Run progressive concatenation
    success = concatenate_frames_progressively(
        pcd_root=pcd_root,
        pcd_fold=pcd_fold,
        pcd_scene=pcd_scene,
        pcd_names=pcd_names,
        pcd_ext=pcd_ext,
        output_folder=output_folder
    )
    
    if success:
        print("\n✓ Progressive concatenation completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Progressive concatenation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

