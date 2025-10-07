#!/bin/bash
  
# 配置参数
# PCD_ROOT="/media/made/MyPassport/DATASET/ScanNet++/outputs/occscannet/base"   
# PCD_FOLD="vis_occ_da_label"  # 或者 "vis_occ_da_pred_cam", "vis_occ_da_label_cam", "vis_occ_da_label" "vis_label_global"
# OUTPUT_FOLDER="vis_occ_da_mayavi/label" # "pred", "label"  
PCD_ROOT="/media/made/MyPassport/DATASET/ScanNet++/outputs/occscannet/base"  
PCD_FOLD="vis_label_global"  # 或者 "vis_occ_da_pred_cam", "vis_occ_da_label_cam", "vis_occ_da_label" "vis_label_global"
OUTPUT_FOLDER="vis_occ_da_mayavi/label_global" # "pred", "label"  
PCD_EXT=".ply" 
     
# 场景列表 
PCD_SCENES=("scene0000_00" "scene0003_02" "scene0006_02" "scene0010_00" "scene0013_01" "scene0024_00" "scene0025_00" "scene0030_01" "scene0031_00" "scene0032_00" 
            "scene0038_01" "scene0040_00" "scene0052_02" "scene0056_00" "scene0059_00" "scene0062_02" "scene0070_00" "scene0072_00" "scene0089_02" "scene0092_01"
            "scene0106_02" "scene0107_00" "scene0115_01" "scene0122_00" "scene0142_00" "scene0160_00" "scene0168_01" "scene0169_01" "scene0173_00"
            "scene0272_01" "scene0276_00" "scene0279_02" "scene0362_01" "scene0416_01" "scene0468_00" "scene0474_02" "scene0487_01" "scene0525_00" 
            "scene0623_00" "scene0626_01" "scene0640_02" "scene0643_00" "scene0652_00" "scene0673_02" "scene0706_00" "scene0101_02" "scene0111_00") 
   
# 点云文件列表 
PCD_NAMES=("pcd_00004" "pcd_00005" "pcd_00012" "pcd_00017" "pcd_00018" "pcd_00020" "pcd_00033" "pcd_00035" "pcd_00041" "pcd_00044" "pcd_00046" "pcd_00048" "pcd_00050" "pcd_00053" 
           "pcd_00056" "pcd_00059" "pcd_00063" "pcd_00070" "pcd_00072" "pcd_00074" "pcd_00076" "pcd_00079" "pcd_00080" "pcd_00082" "pcd_00084" "pcd_00087" "pcd_00094" "pcd_00095" "pcd_00097") 
   
echo "可用场景:"  
for i in "${!PCD_SCENES[@]}"; do  
    echo "$((i+1)). ${PCD_SCENES[i]}"
done

read -p "选择场景编号: " scene_choice
scene_idx=$((scene_choice-1))
 
if [ $scene_idx -lt 0 ] || [ $scene_idx -ge ${#PCD_SCENES[@]} ]; then
    echo "无效的场景选择"
    exit 1
fi

echo "可用点云文件:"
for i in "${!PCD_NAMES[@]}"; do
    echo "$((i+1)). ${PCD_NAMES[i]}"
done

read -p "选择点云文件编号: " name_choice
name_idx=$((name_choice-1))

if [ $name_idx -lt 0 ] || [ $name_idx -ge ${#PCD_NAMES[@]} ]; then
    echo "无效的点云文件选择"
    exit 1
fi

OUTPUT_DIR="$PCD_ROOT/$OUTPUT_FOLDER/${PCD_SCENES[scene_idx]}" 
mkdir -p "$OUTPUT_DIR" 
echo "Save to $OUTPUT_DIR"

# 启动Python脚本
python3 vis_occ.py "$PCD_ROOT" "$PCD_FOLD" "${PCD_SCENES[scene_idx]}" "${PCD_NAMES[name_idx]}" "$PCD_EXT" "$OUTPUT_FOLDER"
 