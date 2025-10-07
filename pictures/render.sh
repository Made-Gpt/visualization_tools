#! /bin/bash 

# "/media/made/MyPassport/DATASET/ScanNet++/outputs/occscannet/miniline/" 
# "vis_occ_da_random_cam"

# "/media/made/MyPassport/DATASET/ScanNet++/outputs/occscannet/base/" 
# "vis_occ_da_csz_cam" 
# "vis_occ_da_ori_gs"

root="path/to/your/folder"   
fold="your_data_folder" 
 
# define scene and name list 
# 'scene0000_00', 'scene0025_00' 
scene_list=("scene0089_00")
# "pcd_00005" "pcd_00012" "pcd_00018" "pcd_00033" "pcd_00035" "pcd_00041" "pcd_00048" "pcd_00053" "pcd_00059" "pcd_00063" "pcd_00070" "pcd_00073" "pcd_00076" "pcd_00080"
name_list=("pcd_00048")
render_fold="vis_occ_da_render"  

for scene in "${scene_list[@]}"; do 
    for name in "${name_list[@]}"; do 
        
        render_dir="$root/$render_fold/$scene"
        mkdir -p "$render_dir"

        python pointflow_fig_colorful.py "$root" "$fold" "$scene" "$name"

        mitsuba mitsuba_scene.xml
    
        oiiotool mitsuba_scene.exr --powc 0.9 -o "$render_dir/${name}.jpg" 
    done 
done 

