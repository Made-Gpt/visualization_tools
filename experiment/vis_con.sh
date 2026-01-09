#!/bin/bash
# filepath: /media/made/MyPassport/DATASET/ScanNet++/outputs/demo/covariance/vis_con.sh

# Gaussian Semantic Analysis and Filtering 
# Usage: bash vis_con.sh
 
# ============ Configuration ============ 
ROOT="/media/made/MyPassport/DATASET/ScanNet++/outputs/occscannet/mini"
PLY_PATH="vis_occ_gaussian_loc_meta_cam"    # PLY file path with semantic info
PLY_NAME="pcd_00033"                       # PLY file name    
SCENE_PATH="scene0006_02"                  # Scene path      
OUTPUT_PATH="vis_occ_entropy"              # Output directory   
 
# Entropy filtering parameters
ENTROPY_LOW=1e-6                           # Minimum entropy threshold
ENTROPY_HIGH=1.0                           # Maximum entropy threshold  
OPACITY_THRES=0.01                         # Opacity threshold

# Confidence computation parameters
CONFIDENCE_METHOD="power"                # Confidence calculation method: exp, sigmoid, power
 
# Save gaussians flags
SAVE_FILTERED=false                        # Set to true to enable filtering and save filtered PLY file
SAVE_CONFIDENCE_COLORED=true               # Set to true to save confidence-colored gaussians
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

echo "=== Gaussian Semantic Analysis ==="
echo "Input file: $PLY_DIR/$PLY_NAME.ply"
echo "Output directory: $OUTPUT_DIR"
echo "Entropy range: [$ENTROPY_LOW, $ENTROPY_HIGH]"
echo "Opacity threshold: $OPACITY_THRES"
echo "Confidence method: $CONFIDENCE_METHOD"
echo "Save filtered gaussians: $SAVE_FILTERED"
echo "Save confidence-colored gaussians: $SAVE_CONFIDENCE_COLORED"
echo "=================================="

# Build command arguments
CMD_ARGS="--ply_dir \"$PLY_DIR\" \
          --ply_name \"$PLY_NAME\" \
          --output_dir \"$OUTPUT_DIR\" \
          --entropy_low $ENTROPY_LOW \
          --entropy_high $ENTROPY_HIGH \
          --opacity_threshold $OPACITY_THRES \
          --confidence_method $CONFIDENCE_METHOD" 
  
# Add save flags if enabled
if [ "$SAVE_FILTERED" = true ]; then
    CMD_ARGS="$CMD_ARGS --save_filtered"
fi

if [ "$SAVE_CONFIDENCE_COLORED" = true ]; then
    CMD_ARGS="$CMD_ARGS --save_confidence_colored"
fi

# Run Python script
eval "python3 vis_con.py $CMD_ARGS"

# Check if script executed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Analysis Completed Successfully! ==="
    echo "Output files:"
    echo "  - Entropy distribution plot: $OUTPUT_DIR/${PLY_NAME}_entropy_dist.png"
    echo "  - Confidence comparison plot: $OUTPUT_DIR/${PLY_NAME}_confidence_comparison.png"
    if [ "$SAVE_FILTERED" = true ]; then
        echo "  - Filtered gaussians: $OUTPUT_DIR/${PLY_NAME}_filtered.ply"
    fi
    if [ "$SAVE_CONFIDENCE_COLORED" = true ]; then
        if [ "$SAVE_FILTERED" = true ]; then
            echo "  - Confidence-colored (filtered): $OUTPUT_DIR/${PLY_NAME}_confidence_${CONFIDENCE_METHOD}.ply"
        else
            echo "  - Confidence-colored (all data): $OUTPUT_DIR/${PLY_NAME}_all_confidence_${CONFIDENCE_METHOD}.ply"
        fi
    fi
    echo "========================================"
else
    echo "Error: Script execution failed!"
    exit 1
fi