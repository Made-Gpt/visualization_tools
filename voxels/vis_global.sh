#!/bin/bash
# filepath: /media/made/MyPassport/DATASET/ScanNet++/outputs/demo/mayavi_vis/vis_global.sh

# Progressive Point Cloud Concatenation Script
# Usage: bash vis_global.sh

# ============ Configuration ============ 
PCD_ROOT="/media/made/MyPassport/DATASET/ScanNet++/outputs/occscannet/base"   
PCD_FOLD="vis_occ_da_label"        # Input folder with individual frames
OUTPUT_FOLDER="vis_label_global"   # Output folder for concatenated results
PCD_EXT=".ply" 
 
# Target scenes to process
PCD_SCENES=("scene0000_00" "scene0006_02" "scene0024_00" "scene0030_01" "scene0040_00" "scene0072_00" "scene0089_02" "scene0092_01" 
            "scene0101_02" "scene0106_02" "scene0107_00" "scene0111_00" "scene0168_01" "scene0169_01" "scene0173_00" "scene0272_01" "scene0362_01" 
            "scene0468_00" "scene0623_00" "scene0626_01" "scene0673_02" "scene0706_00" 
            ) 

# "scene0000_00" "scene0003_02" "scene0006_02" "scene0010_00" "scene0013_01" 
# "scene0024_00" "scene0025_00" "scene0030_01" "scene0031_00" "scene0032_00" 
# "scene0038_01" "scene0040_00" "scene0052_02" "scene0059_00" "scene0062_02" 
# "scene0070_00" "scene0072_00" "scene0089_02" "scene0092_01" "scene0106_02" 
# "scene0107_00" "scene0115_01" "scene0122_00" "scene0142_00" "scene0160_00" 
# "scene0168_01" "scene0169_01" "scene0173_00" "scene0272_01" "scene0276_00" 
# "scene0279_02" "scene0362_01" "scene0416_01" "scene0468_00" "scene0474_02" 
# "scene0487_01" "scene0525_00" "scene0623_00" "scene0626_01" "scene0640_02" 
# "scene0643_00" "scene0652_00" "scene0673_02" "scene0706_00" "scene0101_02" 
# "scene0111_00"
 
# Frame names to concatenate (in order)
PCD_NAMES=("pcd_00004" "pcd_00005" "pcd_00012" "pcd_00018" "pcd_00020" 
           "pcd_00033" "pcd_00035" "pcd_00041" "pcd_00044" "pcd_00046" "pcd_00048" 
           "pcd_00050" "pcd_00053" "pcd_00056" "pcd_00059" "pcd_00063" "pcd_00070" 
           "pcd_00072" "pcd_00074" "pcd_00076" "pcd_00079" "pcd_00080" "pcd_00082" 
           "pcd_00084" "pcd_00087" "pcd_00094" "pcd_00095" "pcd_00097")

# ======================================

# Function to display menu and get user choice
select_scene() {
    echo "=== 可用场景列表 ==="
    for i in "${!PCD_SCENES[@]}"; do  
        echo "$((i+1)). ${PCD_SCENES[i]}"
    done
    echo "0. 处理所有场景"
    echo "==================="
    
    read -p "选择场景编号 (0 表示处理所有场景): " scene_choice
    
    if [ "$scene_choice" = "0" ]; then
        echo "选择: 处理所有场景"
        return 0
    elif [ "$scene_choice" -ge 1 ] && [ "$scene_choice" -le ${#PCD_SCENES[@]} ]; then
        scene_idx=$((scene_choice-1))
        echo "选择: ${PCD_SCENES[scene_idx]}"
        return $scene_idx
    else
        echo "无效的选择，退出"
        exit 1
    fi
}
 
# Function to process a single scene
process_scene() {
    local scene_name=$1
    echo ""
    echo "=== 处理场景: $scene_name ==="
    
    # Check if input directory exists
    input_dir="$PCD_ROOT/$PCD_FOLD/$scene_name"
    if [ ! -d "$input_dir" ]; then
        echo "错误: 输入目录不存在: $input_dir"
        return 1
    fi
    
    # Check which frames exist for this scene
    existing_frames=()
    for frame in "${PCD_NAMES[@]}"; do
        if [ -f "$input_dir/${frame}${PCD_EXT}" ]; then
            existing_frames+=("$frame")
        fi
    done
    
    if [ ${#existing_frames[@]} -lt 2 ]; then
        echo "警告: 场景 $scene_name 中找到的帧数少于2个，跳过"
        echo "找到的帧: ${existing_frames[*]}"
        return 1
    fi
    
    echo "找到 ${#existing_frames[@]} 个有效帧: ${existing_frames[*]}"
    
    # Create output directory
    output_dir="$PCD_ROOT/$OUTPUT_FOLDER/$scene_name"
    mkdir -p "$output_dir"
    
    # Build command arguments
    cmd_args="\"$PCD_ROOT\" \"$PCD_FOLD\" \"$scene_name\" \"$OUTPUT_FOLDER\" \"$PCD_EXT\""
    for frame in "${existing_frames[@]}"; do
        cmd_args="$cmd_args \"$frame\""
    done
    
    # Run Python script
    echo "执行命令: python3 vis_global.py $cmd_args"
    eval "python3 vis_global.py $cmd_args"
    
    if [ $? -eq 0 ]; then
        echo "✓ 场景 $scene_name 处理完成"
        echo "输出目录: $output_dir"
        return 0
    else
        echo "✗ 场景 $scene_name 处理失败"
        return 1
    fi
}

# Main execution
echo "=== Progressive Point Cloud Concatenation ==="
echo "输入文件夹: $PCD_FOLD"
echo "输出文件夹: $OUTPUT_FOLDER"
echo "文件扩展名: $PCD_EXT"
echo "可处理帧数: ${#PCD_NAMES[@]}"
echo "============================================="

# Check if Python script exists
if [ ! -f "vis_global.py" ]; then
    echo "错误: vis_global.py 脚本不存在"
    exit 1
fi

# Get user selection
select_scene
choice_result=$?

# Process based on user choice
if [ $choice_result -eq 0 ]; then
    # Process all scenes
    echo ""
    echo "开始处理所有 ${#PCD_SCENES[@]} 个场景..."
    
    success_count=0
    total_count=${#PCD_SCENES[@]}
    
    for scene in "${PCD_SCENES[@]}"; do
        if process_scene "$scene"; then
            ((success_count++))
        fi
    done
    
    echo ""
    echo "=== 批量处理完成 ==="
    echo "成功处理: $success_count/$total_count 个场景"
    echo "===================="
    
else
    # Process single scene
    scene_idx=$choice_result
    scene_name="${PCD_SCENES[scene_idx]}"
    
    process_scene "$scene_name"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=== 单场景处理完成 ==="
        echo "场景: $scene_name"
        echo "===================="
    fi
fi

echo ""
echo "脚本执行完成!"

