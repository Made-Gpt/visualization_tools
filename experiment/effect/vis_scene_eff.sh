#!/bin/bash

# Scene Efficiency Visualization Script - Accumulated Anchors vs Frame

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Scene Efficiency Visualization - Accumulated Anchors"
echo ""

CONFIGS=(
    "SplatSSC:effect/splatssc_scene.txt"
    "STGS:effect/stgs_scene.txt"
) 

# EmbodiedOcc baseline (fixed anchor count)
EMBODIEDOCC_ANCHORS=16200

echo "🔍 Checking data files..."
for config in "${CONFIGS[@]}"; do
    config_name="${config%%:*}"
    file_path="${config#*:}"
    
    if [ ! -f "$file_path" ]; then
        echo "❌ Missing file: $file_path"
        exit 1
    fi
    echo "✅ $config_name: $file_path"
done
echo "✅ EmbodiedOcc: Fixed baseline at $EMBODIEDOCC_ANCHORS anchors"

echo ""
echo "🎯 Generating plots..."

python3 vis_scene_eff.py \
    --configs "${CONFIGS[@]}" \
    --embodiedocc-anchors $EMBODIEDOCC_ANCHORS \
    --output "effect/anchor_comparison.png" \
    --figsize 10 10 \
    --dpi 1200

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Done! Output plots saved in effect/:" 
    echo "   - anchor_comparison.png (with numbers, 1:1)"
    echo "   - anchor_comparison_clean.png (no numbers, 1:1)"
    echo "   - anchor_comparison_4x3.png (with numbers, 4:3)"
    echo "   - anchor_comparison_clean_4x3.png (no numbers, 4:3)"
    echo "   - anchor_comparison_16x9.png (with numbers, 16:9)"
    echo "   - anchor_comparison_clean_16x9.png (no numbers, 16:9)"
     
    if command -v xdg-open &> /dev/null; then
        echo "💡 Run 'xdg-open effect/anchor_comparison.png' to view"
    fi
else
    echo "❌ Failed!"
    exit 1
fi
