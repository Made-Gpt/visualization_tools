#!/bin/bash

# Gaussian Splatting Visualization - Simple Version
# Usage: bash vis_gs.sh

# ============ Configuration ============ 
ROOT="/media/made/MyPassport/DATASET/ScanNet++/outputs/occscannet/base/"  # "/media/made/MyPassport/DATASET/ScanNet++/outputs/occscannet/base/" 
PLY_PATH="vis_occ_da_gaussian_cam"         # PLY file path (fixed typo)  # "vis_occ_da_gaussian_cam" 
PLY_NAME="pcd_00018"                       # PLY file name 
SCENE_PATH="scene0089_02"                  # Scene path (removed trailing comma) 
OUTPUT_PATH="vis_occ_da_render_gs"         # Output directory  
MAX_ELLIPSOIDS_3D=15600                    # Number of 3D ellipsoids   
MAX_POINTS_2D=15600                        # Number of 2D points  
FIG_SIZE=(20 15)   
ALPHA_THRES=0.002                          # 0.012
RESOLUTION=24                              # Ellipsoid resolution , '24'
PROJECTION="xz"                            # Projection plane: xy/xz/yz  
# ======================================
 
PLY_DIR="$ROOT/$PLY_PATH/$SCENE_PATH"

# Check if file exists 
if [ ! -f "$PLY_DIR/$PLY_NAME.ply" ]; then 
    echo "Error: PLY file not found: $PLY_DIR/$PLY_NAME.ply"
    exit 1
fi 
 
# Create output directory 
OUTPUT_DIR="$ROOT/$OUTPUT_PATH/$SCENE_PATH"
mkdir -p "$OUTPUT_DIR"

# Run Python script with correct parameters
python3 vis_gs.py --ply_dir "$PLY_DIR" \
                  --ply_name "$PLY_NAME" \
                  --output_dir "$OUTPUT_DIR" \
                  --max_ellipsoids_3d $MAX_ELLIPSOIDS_3D \
                  --max_points_2d $MAX_POINTS_2D \
                  --resolution $RESOLUTION \
                  --alpha_threshold $ALPHA_THRES \
                  --figsize ${FIG_SIZE[@]} \
                  --projection_plane "$PROJECTION"

echo "Visualization completed successfully!"
