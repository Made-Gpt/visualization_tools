#!/bin/bash

echo "🎨 Voxel Visualization Tool"
echo "=========================="
echo ""

# Method selection
echo "🔬 Select method:"
echo "  1. Label"
echo "  2. SplatSSC"
echo "  3. EmbodiedOcc"
echo "  4. Ours" 
echo "  5. Ours_hist" 
echo "  6. Ours_curr" 
echo "  7. Ours_loc" 
echo ""
read -p "Select method number: " method_choice

# Load method configuration from JSON
METHOD_CONFIG_FILE="method_config.json"

case $method_choice in
    1)
        METHOD_NAME="Label"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Label']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Label']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Label']['output_folder'])")
        ;;
    2)
        METHOD_NAME="SplatSSC"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['SplatSSC']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['SplatSSC']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['SplatSSC']['output_folder'])")
        ;;
    3)
        METHOD_NAME="EmbodiedOcc"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['EmbodiedOcc']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['EmbodiedOcc']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['EmbodiedOcc']['output_folder'])")
        ;;
    4)
        METHOD_NAME="Ours"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours']['output_folder'])")
        ;; 
    5)  
        METHOD_NAME="Ours_hist" 
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_hist']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_hist']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_hist']['output_folder'])")
        ;;
    6)  
        METHOD_NAME="Ours_curr"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_curr']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_curr']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_curr']['output_folder'])")
        ;;
    7)  
        METHOD_NAME="Ours_loc"
        PCD_ROOT=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_loc']['pcd_root'])")
        PCD_FOLD=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_loc']['pcd_fold'])")
        OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$METHOD_CONFIG_FILE'))['Ours_loc']['output_folder'])")
        ;;
    *) 
        echo "❌ Invalid method selection"
        exit 1
        ;; 
esac 
 
echo "✅ Selected method: $METHOD_NAME" 
echo "" 
 
# Configuration  
PCD_EXT=".ply"
   
# Scene list
PCD_SCENES=("scene0000_00" "scene0003_02" "scene0006_02" "scene0010_00" "scene0013_01" "scene0024_00" "scene0025_00" "scene0030_01" "scene0031_00" "scene0032_00" 
            "scene0038_01" "scene0040_00" "scene0052_02" "scene0056_00" "scene0059_00" "scene0062_02" "scene0070_00" "scene0072_00" "scene0089_02" "scene0092_01"
            "scene0106_02" "scene0107_00" "scene0115_01" "scene0122_00" "scene0142_00" "scene0160_00" "scene0168_01" "scene0169_01" "scene0173_00" "scene0234_00"
            "scene0253_00" "scene0271_00" "scene0271_01" "scene0272_01" "scene0273_01" "scene0276_00" "scene0279_02" "scene0280_01" "scene0296_00" "scene0296_01"
            "scene0298_00" "scene0340_01" "scene0345_01" "scene0362_01" "scene0403_01" "scene0416_03" "scene0420_01" "scene0468_00" "scene0474_02" "scene0487_01"  
            "scene0500_00" "scene0525_00" "scene0623_00" "scene0626_01" "scene0640_02" "scene0643_00" "scene0652_00" "scene0673_02" "scene0706_00" "scene0101_02" "scene0111_00") 

# Frame list  
PCD_NAMES=("pcd_00004" "pcd_00005" "pcd_00012" "pcd_00017" "pcd_00018" "pcd_00020" "pcd_00033" "pcd_00035" "pcd_00041" "pcd_00044" "pcd_00046" "pcd_00048" "pcd_00050" "pcd_00053" "pcd_00056" "pcd_00059"  
           "pcd_00060" "pcd_00061" "pcd_00063" "pcd_00070" "pcd_00072" "pcd_00074" "pcd_00075" "pcd_00076" "pcd_00079" "pcd_00080" "pcd_00082" "pcd_00084" "pcd_00087" "pcd_00091" "pcd_00094" "pcd_00095" "pcd_00097") 

# Select scene
echo "📍 Available scenes:"  
for i in "${!PCD_SCENES[@]}"; do  
    echo "  $((i+1)). ${PCD_SCENES[i]}"
done
echo ""
read -p "Select scene number: " scene_choice
scene_idx=$((scene_choice-1))
 
if [ $scene_idx -lt 0 ] || [ $scene_idx -ge ${#PCD_SCENES[@]} ]; then
    echo "❌ Invalid scene selection"
    exit 1
fi

# Select frame
echo ""
echo "🎬 Available frames:"
for i in "${!PCD_NAMES[@]}"; do
    echo "  $((i+1)). ${PCD_NAMES[i]}"
done
echo ""
read -p "Select frame number: " name_choice
name_idx=$((name_choice-1))

if [ $name_idx -lt 0 ] || [ $name_idx -ge ${#PCD_NAMES[@]} ]; then
    echo "❌ Invalid frame selection"
    exit 1
fi

# Ask about zoom
echo ""
read -p "Use zoom in mode for detailed view? (y/n) [default: n]: " zoom_choice
use_zoom="false"
if [[ "$zoom_choice" =~ ^[Yy]$ ]]; then
    use_zoom="true"
fi

# Ask about 3D interface
echo ""
read -p "Show 3D interactive interface? (y/n) [default: n]: " show_3d_choice
show_3d="false"
if [[ "$show_3d_choice" =~ ^[Yy]$ ]]; then
    show_3d="true"
fi

# Create output directory
OUTPUT_DIR="$PCD_ROOT/$OUTPUT_FOLDER/${PCD_SCENES[scene_idx]}" 
mkdir -p "$OUTPUT_DIR" 

echo ""
echo "🎯 Settings:"
echo "  Method: $METHOD_NAME"
echo "  Scene: ${PCD_SCENES[scene_idx]}"
echo "  Frame: ${PCD_NAMES[name_idx]}"
echo "  Output: $OUTPUT_DIR"
echo "  Zoom Mode: $use_zoom"
echo "  3D Interface: $show_3d"
echo "  Projection: Parallel (Fixed)"
echo ""

# Run Python script
python3 vis_occ.py "$PCD_ROOT" "$PCD_FOLD" "${PCD_SCENES[scene_idx]}" "${PCD_NAMES[name_idx]}" "$PCD_EXT" "$OUTPUT_FOLDER" "$show_3d" "$use_zoom"

echo ""
echo "✅ Done!"
 