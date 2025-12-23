#!/bin/bash

echo "🎬 Rotating Voxel Visualization Tool"
echo "===================================="
echo ""

# Ask for quick mode
read -p "⚡ Use quick mode? (only select method/scene/frame, all else default, y/n): " quick_mode
if [[ "$quick_mode" =~ ^[Yy]$ ]]; then
    QUICK_MODE=true
    echo "✅ Quick mode enabled: all settings will use defaults"
else
    QUICK_MODE=false
fi
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

PCD_SCENE="${PCD_SCENES[$scene_idx]}"
echo "✅ Selected scene: $PCD_SCENE"
echo ""

# Select frame
echo "🎞️  Available frames:"
for i in "${!PCD_NAMES[@]}"; do
    echo "  $((i+1)). ${PCD_NAMES[i]}"
done
echo ""
read -p "Select frame number: " frame_choice
frame_idx=$((frame_choice-1))

if [ $frame_idx -lt 0 ] || [ $frame_idx -ge ${#PCD_NAMES[@]} ]; then
    echo "❌ Invalid frame selection"
    exit 1
fi

PCD_NAME="${PCD_NAMES[$frame_idx]}"
echo "✅ Selected frame: $PCD_NAME"
echo ""

# Rotation parameters
if [ "$QUICK_MODE" = true ]; then
    # Quick mode: use all defaults
    num_frames=36
    start_angle=0
    end_angle=360
    image_width=1600
    echo "🎬 Rotation settings: Using defaults (36 frames, 0-360°, 1600px)"
else
    # Interactive mode: ask for each parameter
    echo "🎬 Rotation settings:"
    echo ""
    read -p "Number of frames (default: 36): " num_frames
    if [ -z "$num_frames" ]; then
        num_frames=36
    fi

    read -p "Start angle in degrees (default: 0): " start_angle
    if [ -z "$start_angle" ]; then
        start_angle=0
    fi

    read -p "End angle in degrees (default: 360): " end_angle
    if [ -z "$end_angle" ]; then
        end_angle=360
    fi

    read -p "Image width in pixels (default: 1600): " image_width
    if [ -z "$image_width" ]; then
        image_width=1600
    fi
fi

echo ""
echo "📊 Configuration summary:"
echo "  Method: $METHOD_NAME"
echo "  Scene: $PCD_SCENE"
echo "  Frame: $PCD_NAME"
echo "  Frames: $num_frames"
echo "  Rotation: ${start_angle}° to ${end_angle}°"
echo "  Image size: ${image_width}x$((image_width*3/4)) (4:3 aspect ratio)"
echo "  Output: $PCD_ROOT/$OUTPUT_FOLDER/$PCD_SCENE/${PCD_NAME}_rotation/"
echo ""

if [ "$QUICK_MODE" = true ]; then
    # Quick mode: use offscreen by default
    OFFSCREEN="true"
    echo "💡 Quick mode: Running in offscreen mode (no window)"
else
    # Interactive mode: ask user
    read -p "Show window during rendering? (y/n, default: n): " show_window
    if [[ ! "$show_window" =~ ^[Yy]$ ]]; then
        OFFSCREEN="true"
        echo "💡 Running in offscreen mode (no window)"
    else
        OFFSCREEN="false"
        echo "💡 Running with window (slower)"
    fi
fi

echo ""
echo "🎨 Generating rotation frames..."
echo "================================"
echo ""

# Run visualization with or without xvfb
if [[ "$OFFSCREEN" == "true" ]]; then
    # Use xvfb-run for offscreen rendering to avoid segfault
    xvfb-run -a -s "-screen 0 1920x1200x24" python3 vis_occ_rot.py \
        "$PCD_ROOT" \
        "$PCD_FOLD" \
        "$PCD_SCENE" \
        "$PCD_NAME" \
        "$PCD_EXT" \
        "$OUTPUT_FOLDER" \
        "$num_frames" \
        "$start_angle" \
        "$end_angle" \
        "$image_width" \
        "$OFFSCREEN"
else
    # Run with visible window
    python3 vis_occ_rot.py \
        "$PCD_ROOT" \
        "$PCD_FOLD" \
        "$PCD_SCENE" \
        "$PCD_NAME" \
        "$PCD_EXT" \
        "$OUTPUT_FOLDER" \
        "$num_frames" \
        "$start_angle" \
        "$end_angle" \
        "$image_width" \
        "$OFFSCREEN"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Frame generation complete!"
    echo ""
    echo "📁 Output directory:"
    echo "   $PCD_ROOT/$OUTPUT_FOLDER/$PCD_SCENE/${PCD_NAME}_rotation/"
    echo ""
    
    # GIF creation
    if [ "$QUICK_MODE" = true ]; then
        # Quick mode: auto-create GIF with defaults
        merge_gif="y"
        gif_fps=10
        SCALE_DOWN=true
        echo "💡 Quick mode: Auto-creating GIF with defaults (10fps, ${PCD_SCENE}_${PCD_NAME}_rotation.gif, scaled 50%)"
    else
        # Interactive mode: ask user
        read -p "🎞️  Merge frames to GIF? (y/n, default: n): " merge_gif
    fi
    
    if [[ "$merge_gif" =~ ^[Yy]$ ]]; then
        if [ "$QUICK_MODE" != true ]; then
            echo ""
            read -p "GIF frame rate (fps, default: 10): " gif_fps
            if [ -z "$gif_fps" ]; then
                gif_fps=10
            fi
            
            read -p "Scale down GIF? (half size for smaller file, y/n, default: y): " scale_gif
            if [[ ! "$scale_gif" =~ ^[Nn]$ ]]; then
                SCALE_DOWN=true
                echo "💡 GIF will be scaled to 50% size"
            else
                SCALE_DOWN=false
                echo "💡 GIF will use full resolution"
            fi
        fi
        
        OUTPUT_DIR="$PCD_ROOT/$OUTPUT_FOLDER/$PCD_SCENE/${PCD_NAME}_rotation"
        # Auto-generate filename: scene_frame_rotation.gif
        gif_name="${PCD_SCENE}_${PCD_NAME}_rotation.gif"
        GIF_PATH="$OUTPUT_DIR/$gif_name"
        
        echo ""
        echo "🔄 Creating GIF..."
        
        # Try ffmpeg first (more efficient and better quality)
        if command -v ffmpeg &> /dev/null; then
            if [ "$SCALE_DOWN" = true ]; then
                # Scale down to 50% with white background (replace transparent areas)
                ffmpeg -f lavfi -i "color=white:s=${image_width}x$((image_width*3/4))" \
                    -framerate $gif_fps -pattern_type glob -i "$OUTPUT_DIR/frame_*_t.png" \
                    -filter_complex "[0:v]scale=$((image_width/2)):$((image_width*3/8))[bg];[1:v]fps=$gif_fps,scale=$((image_width/2)):$((image_width*3/8)):flags=lanczos[fg];[bg][fg]overlay=shortest=1,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
                    -loop 0 "$GIF_PATH" -y > /dev/null 2>&1
            else
                # Full resolution with white background (replace transparent areas)
                ffmpeg -f lavfi -i "color=white:s=${image_width}x$((image_width*3/4))" \
                    -framerate $gif_fps -pattern_type glob -i "$OUTPUT_DIR/frame_*_t.png" \
                    -filter_complex "[0:v][1:v]scale2ref[bg][fg];[1:v]fps=$gif_fps[fg];[bg][fg]overlay=shortest=1,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
                    -loop 0 "$GIF_PATH" -y > /dev/null 2>&1
            fi
            
            if [ $? -eq 0 ]; then
                echo "✅ GIF created successfully!"
                echo "   $GIF_PATH"
                # Show file size
                GIF_SIZE=$(du -h "$GIF_PATH" | cut -f1)
                echo "   Size: $GIF_SIZE"
            else
                echo "❌ GIF creation failed"
            fi
        elif command -v convert &> /dev/null; then
            echo "⚠️  ffmpeg not found, trying ImageMagick (may be slower)..."
            # Calculate delay for GIF (in centiseconds)
            delay=$((100 / gif_fps))
            
            if [ "$SCALE_DOWN" = true ]; then
                convert -delay $delay -loop 0 -resize 50% "$OUTPUT_DIR"/frame_*_t.png "$GIF_PATH"
            else
                convert -delay $delay -loop 0 "$OUTPUT_DIR"/frame_*_t.png "$GIF_PATH"
            fi
            
            if [ $? -eq 0 ]; then
                echo "✅ GIF created successfully!"
                echo "   $GIF_PATH"
                GIF_SIZE=$(du -h "$GIF_PATH" | cut -f1)
                echo "   Size: $GIF_SIZE"
            else
                echo "❌ GIF creation failed (may be out of memory)"
                echo "💡 Try running again with scaled down option or use ffmpeg"
            fi
        else
            echo "❌ Neither ffmpeg nor ImageMagick found"
            echo "💡 Install ffmpeg (recommended): sudo apt-get install ffmpeg"
            echo "💡 Or install ImageMagick: sudo apt-get install imagemagick"
        fi
    fi
    
    echo ""
    echo "💡 Additional options:"
    echo "   • Create video: ffmpeg -framerate 30 -pattern_type glob -i '$PCD_ROOT/$OUTPUT_FOLDER/$PCD_SCENE/${PCD_NAME}_rotation/frame_*_t.png' -c:v libx264 -pix_fmt yuv420p output.mp4"
    echo "   • Create GIF: convert -delay 10 -loop 0 '$PCD_ROOT/$OUTPUT_FOLDER/$PCD_SCENE/${PCD_NAME}_rotation/frame_*_t.png' output.gif"
    echo ""
else
    echo ""
    echo "❌ Visualization failed"
    exit 1
fi
