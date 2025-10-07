#!/bin/bash

# 3D Gaussian Rendering with Mitsuba v3
# Usage: ./vis_gs.sh

# 配置参数
PLY_ROOT="/media/made/MyPassport/DATASET/ScanNet++/outputs/occscannet/base"
PLY_FOLD="vis_occ_da_gaussian_cam"  # 或者其他高斯文件夹名称
OUTPUT_FOLDER="vis_occ_da_mitsuba_gs"
PLY_EXT=".ply" 
  
PCD_SCENES=("scene0000_00" "scene0003_02" "scene0006_02" "scene0025_00" 
            "scene0107_00" "scene0168_01" "scene0362_01" "scene0626_01" "scene0673_02" "scene0706_00")  
PCD_NAMES=("pcd_00005" "pcd_00012" "pcd_00020" "pcd_00033" "pcd_00035" "pcd_00050" "pcd_00070" "pcd_00076")  
  
# 渲染参数 
WIDTH=1024 
HEIGHT=1024
SPP=128  # Samples per pixel (reduced for faster rendering)
MAX_GAUSSIANS=5000  # Maximum gaussians per scene

# 椭球网格质量参数 (Higher values = better quality, slower rendering)
N_THETA=24  # Longitude divisions (horizontal, around Z-axis) - Academic quality: 24-32
N_PHI=16   # Latitude divisions (vertical, north to south pole) - Academic quality: 12-20
  
# 光照参数 (Academic quality lighting)  
AMBIENT_LIGHT=0.1    # Environment lighting strength (0.2-0.6) - Reduced for deeper colors 
MAIN_LIGHT=3.0       # Primary directional light strength (2.0-4.0) - Reduced for deeper colors 
FILL_LIGHT=2.5       # Fill light strength (1.5-2.5) - Reduced for deeper colors 
TOP_LIGHT=1.0        # Top light strength (1.0-2.0) - Reduced for deeper colors    
 
# 函数：选择场景 
select_scene() { 
    echo "Available scenes:"
    for i in "${!PCD_SCENES[@]}"; do
        echo "  $((i+1)). ${PCD_SCENES[$i]}"
    done

    while true; do
        read -p "Select scene (1-${#PCD_SCENES[@]}) or 'all' for all scenes: " scene_choice
        
        if [[ "$scene_choice" == "all" ]]; then
            SELECTED_SCENES=("${PCD_SCENES[@]}")
            break
        elif [[ "$scene_choice" =~ ^[0-9]+$ ]] && [ "$scene_choice" -ge 1 ] && [ "$scene_choice" -le "${#PCD_SCENES[@]}" ]; then
            SELECTED_SCENES=("${PCD_SCENES[$((scene_choice-1))]}")
            break
        else
            echo "Invalid selection. Please enter a number between 1 and ${#PCD_SCENES[@]} or 'all'."
        fi
    done 
}

# 函数：选择文件名
select_pcd_name() {
    echo "Available PCD names:"
    for i in "${!PCD_NAMES[@]}"; do
        echo "  $((i+1)). ${PCD_NAMES[$i]}"
    done
    
    while true; do
        read -p "Select PCD name (1-${#PCD_NAMES[@]}) or 'all' for all names: " name_choice
        
        if [[ "$name_choice" == "all" ]]; then
            SELECTED_NAMES=("${PCD_NAMES[@]}")
            break
        elif [[ "$name_choice" =~ ^[0-9]+$ ]] && [ "$name_choice" -ge 1 ] && [ "$name_choice" -le "${#PCD_NAMES[@]}" ]; then
            SELECTED_NAMES=("${PCD_NAMES[$((name_choice-1))]}")
            break
        else
            echo "Invalid selection. Please enter a number between 1 and ${#PCD_NAMES[@]} or 'all'."
        fi
    done
} 

# 检查基础路径是否存在
if [ ! -d "$PLY_ROOT/$PLY_FOLD" ]; then
    echo "Error: Input folder $PLY_ROOT/$PLY_FOLD does not exist!"
    echo "Please check PLY_ROOT and PLY_FOLD parameters."
    exit 1
fi

echo "=== Mitsuba v3 Gaussian Rendering (Academic Quality) ==="
echo "Base path: $PLY_ROOT/$PLY_FOLD"
echo "Output folder: $PLY_ROOT/$OUTPUT_FOLDER"
echo "Render resolution: ${WIDTH}x${HEIGHT}" 
echo "Samples per pixel: $SPP"
echo "Max gaussians: $MAX_GAUSSIANS"
echo ""
echo "Academic Quality Settings:"
echo "  - Ellipsoid mesh: ${N_THETA}x${N_PHI} (theta x phi divisions)"
echo "  - Lighting: Ambient=${AMBIENT_LIGHT}, Main=${MAIN_LIGHT}, Fill=${FILL_LIGHT}, Top=${TOP_LIGHT}"
echo "  - PNG output: Both transparent and solid background versions"
echo ""

# 选择场景和PCD名称
select_scene
select_pcd_name

echo ""
echo "Selected scenes: ${SELECTED_SCENES[@]}"
echo "Selected PCD names: ${SELECTED_NAMES[@]}"

# 创建输出文件夹
mkdir -p "$PLY_ROOT/$OUTPUT_FOLDER"

# 检查Python环境和依赖
echo ""
echo "Checking dependencies..." 
python3 -c "import mitsuba as mi; print(f'Mitsuba version: {mi.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Mitsuba not found or not properly installed"
    echo "Please install Mitsuba v3: pip install mitsuba"
fi

python3 -c "import plyfile; print('PLYfile: OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: plyfile not found"
    echo "Please install: pip install plyfile"
fi

python3 -c "import tqdm; print('tqdm: OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: tqdm not found"
    echo "Please install: pip install tqdm"
fi

echo ""
echo "Starting Gaussian rendering..."

# 渲染每个选定的场景和PCD组合
TOTAL_JOBS=0
COMPLETED_JOBS=0

# 计算总任务数
for scene in "${SELECTED_SCENES[@]}"; do
    for pcd_name in "${SELECTED_NAMES[@]}"; do
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
done

echo "Total rendering jobs: $TOTAL_JOBS"
echo ""

# 执行渲染任务
for scene in "${SELECTED_SCENES[@]}"; do
    for pcd_name in "${SELECTED_NAMES[@]}"; do 
        COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
        
        # 构建输入路径: PLY_ROOT/PLY_FOLD/SCENE/PCD_NAME
        INPUT_PATH="$PLY_ROOT/$PLY_FOLD/$scene"
        PLY_FILE="$pcd_name$PLY_EXT"
        FULL_PLY_PATH="$INPUT_PATH/$PLY_FILE"
        
        # 构建输出路径: OUTPUT_FOLDER/SCENE/PCD_NAME.exr
        SCENE_OUTPUT_DIR="$PLY_ROOT/$OUTPUT_FOLDER/$scene"
        mkdir -p "$SCENE_OUTPUT_DIR"  # 创建场景输出目录
        OUTPUT_PATH="$SCENE_OUTPUT_DIR/${pcd_name}.exr" 
        
        echo "[$COMPLETED_JOBS/$TOTAL_JOBS] Processing: $scene/$pcd_name"
        echo "  Input: $FULL_PLY_PATH"
        echo "  Output: $OUTPUT_PATH"
        
        # 检查输入文件是否存在
        if [ ! -f "$FULL_PLY_PATH" ]; then
            echo "  ⚠️  Warning: PLY file not found: $FULL_PLY_PATH"
            echo "  Skipping..."
            echo ""
            continue
        fi 
        
        # 检查输出文件是否已存在
        if [ -f "$OUTPUT_PATH" ]; then
            echo "  ℹ️  Output already exists, skipping..."
            echo ""
            continue
        fi
        
        # 运行Python脚本渲染单个文件
        python3 vis_gs.py \
            --input_file "$FULL_PLY_PATH" \
            --output_file "$OUTPUT_PATH" \
            --width $WIDTH \
            --height $HEIGHT \
            --spp $SPP \
            --max_gaussians $MAX_GAUSSIANS \
            --n_theta $N_THETA \
            --n_phi $N_PHI \
            --ambient_light $AMBIENT_LIGHT \
            --main_light $MAIN_LIGHT \
            --fill_light $FILL_LIGHT \
            --top_light $TOP_LIGHT
        
        if [ $? -eq 0 ]; then 
            echo "  ✅ Completed: $OUTPUT_PATH"
        else
            echo "  ❌ Failed to render: $FULL_PLY_PATH"
        fi
        echo ""
    done
done

# 检查渲染结果
echo ""
echo "=== Rendering Complete ==="
echo "Results saved to: $PLY_ROOT/$OUTPUT_FOLDER"
echo "File count: $(find "$PLY_ROOT/$OUTPUT_FOLDER" -name "*.exr" 2>/dev/null | wc -l) EXR images"
echo "Directory structure:"
find "$PLY_ROOT/$OUTPUT_FOLDER" -type d 2>/dev/null | sort | sed "s|$PLY_ROOT/$OUTPUT_FOLDER|  .|g"

# 检查是否有任何成功的渲染
EXR_COUNT=$(find "$PLY_ROOT/$OUTPUT_FOLDER" -name "*.exr" 2>/dev/null | wc -l)
if [ "$EXR_COUNT" -gt 0 ]; then

    # 可选：转换EXR到PNG用于预览 (只转换本次运行的文件)
    echo "" 
    read -p "Convert current batch EXR files to PNG for preview? (y/n): " convert_choice
    if [[ $convert_choice =~ ^[Yy]$ ]]; then
        echo ""
        echo "Choose PNG brightness level:"
        echo "  1. Dark (--powc 0.8)"
        echo "  2. Light (--powc 0.5)"
        
        while true; do
            read -p "Select brightness (1-2): " brightness_choice
            
            if [[ "$brightness_choice" == "1" ]]; then
                POWC_VALUE="1.0"
                BRIGHTNESS_DESC="dark"
                BRIGHTNESS_SUFFIX="_d"
                break
            elif [[ "$brightness_choice" == "2" ]]; then
                POWC_VALUE="0.5"
                BRIGHTNESS_DESC="light"
                BRIGHTNESS_SUFFIX="_l"
                break
            else
                echo "Invalid selection. Please enter 1 or 2."
            fi
        done
        
        echo "Converting current batch EXR to PNG with oiiotool ($BRIGHTNESS_DESC mode)..." 
        
        # Check if oiiotool is available 
        if ! command -v oiiotool &> /dev/null; then
            echo "❌ Error: oiiotool not found. Please install OpenImageIO."
            echo "   Try: sudo apt install openimageio-tools  (Ubuntu/Debian)"
            echo "   Or:  brew install openimageio  (macOS)"
            echo "   Or:  conda install openimageio  (Conda)" 
        else  
            # Convert only the files from current run (selected scenes and PCD names)
            for scene in "${SELECTED_SCENES[@]}"; do
                for pcd_name in "${SELECTED_NAMES[@]}"; do
                    exr_file="$PLY_ROOT/$OUTPUT_FOLDER/$scene/${pcd_name}.exr"
                    
                    if [ -f "$exr_file" ]; then 
                        png_file="$PLY_ROOT/$OUTPUT_FOLDER/$scene/${pcd_name}${BRIGHTNESS_SUFFIX}.png"
                        
                        echo "  Converting: $scene/${pcd_name}.exr -> $scene/${pcd_name}${BRIGHTNESS_SUFFIX}.png (--powc $POWC_VALUE)"
                        oiiotool "$exr_file" --powc "$POWC_VALUE" -o "$png_file"
                         
                        if [ $? -eq 0 ]; then
                            echo "  ✅ Saved PNG with transparency ($BRIGHTNESS_DESC): $png_file"
                        else
                            echo "  ❌ Failed to convert: $exr_file"
                        fi
                    else
                        echo "  ⚠️  EXR file not found: $scene/${pcd_name}.exr (probably failed to render)"
                    fi
                done
            done 
            echo "PNG conversion complete!"  
        fi
    fi
    
else
    echo "Error: Rendering failed!"
    exit 1
fi 

echo ""
echo "=== Summary ==="
echo "Input: $PLY_ROOT/$PLY_FOLD"
echo "Output: $PLY_ROOT/$OUTPUT_FOLDER"
echo "Resolution: ${WIDTH}x${HEIGHT}, SPP: $SPP, Max Gaussians: $MAX_GAUSSIANS"
