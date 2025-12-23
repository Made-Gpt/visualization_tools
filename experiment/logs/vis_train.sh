#!/bin/bash

# Training Loss and mIoU Visualization Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Training Loss and mIoU Visualization"
echo ""

CONFIGS=(
    "sequence_embed:logs/splatssc_base.log"
    "sequence_mono:logs/splatssc_base_e2e.log"
    "mono:logs/mono.log"
) 

echo "🔍 Checking log files..."
for config in "${CONFIGS[@]}"; do
    config_name="${config%%:*}"
    log_path="${config#*:}"
    
    if [ ! -f "$log_path" ]; then
        echo "❌ Missing log: $log_path"
        exit 1
    fi
    echo "✅ $config_name: $log_path"
done

echo ""
echo "🎯 Generating plots..."

# Note: sequences process 30 frames/iter, mono processes 1 frame/iter
python3 vis_train.py \
    --configs "${CONFIGS[@]}" \
    --output "logs/loss_comparison.png" \
    --figsize 20 10 \
    --smooth \ 
    --frames-per-iter 30 30 1 
 
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Done! Output plots saved in logs/:" 
    echo "   - loss_comparison_loss.png (with numbers)"
    echo "   - loss_comparison_miou.png (with numbers)"
    echo "   - loss_comparison_loss_clean.png (no numbers, 1:1)"
    echo "   - loss_comparison_miou_clean.png (no numbers, 1:1)"
    echo "   - loss_comparison_loss_clean_4x3.png (no numbers, 4:3)"
    echo "   - loss_comparison_miou_clean_4x3.png (no numbers, 4:3)"
    echo "   - loss_comparison_loss_supersmooth.png (super smooth, 1:1)"
    echo "   - loss_comparison_miou_supersmooth.png (super smooth, 1:1)"
    echo "   - loss_comparison_loss_supersmooth_4x3.png (super smooth, 4:3)"
    echo "   - loss_comparison_miou_supersmooth_4x3.png (super smooth, 4:3)"
    echo "   - loss_comparison_loss_supersmooth_uncertainty.png (with uncertainty, 1:1) ⭐"
    echo "   - loss_comparison_loss_supersmooth_uncertainty_4x3.png (with uncertainty, 4:3) ⭐"
    
    if command -v xdg-open &> /dev/null; then
        echo "💡 Run 'xdg-open logs/loss_comparison_loss.png' to view"
    fi
else
    echo "❌ Failed!"
    exit 1
fi 
 