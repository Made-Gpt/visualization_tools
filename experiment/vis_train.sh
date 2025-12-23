#!/bin/bash

# Training Loss and mIoU Visualization Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Training Loss and mIoU Visualization"
echo ""

# Load training configuration
TRAIN_CONFIG_FILE="train_config.json"

if [ ! -f "$TRAIN_CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $TRAIN_CONFIG_FILE"
    exit 1 
fi

# Get available models
echo "📋 Available models:"
MODELS=($(python3 -c "import json; config=json.load(open('$TRAIN_CONFIG_FILE')); print(' '.join(config.keys()))"))

for i in "${!MODELS[@]}"; do
    model_name="${MODELS[$i]}"
    desc=$(python3 -c "import json; config=json.load(open('$TRAIN_CONFIG_FILE')); print(config['$model_name'].get('description', 'No description'))")
    echo "  $((i+1)). $model_name - $desc"
done

echo ""
read -p "Select model (1-${#MODELS[@]}): " model_choice

if [[ ! "$model_choice" =~ ^[0-9]+$ ]] || [ "$model_choice" -lt 1 ] || [ "$model_choice" -gt "${#MODELS[@]}" ]; then
    echo "❌ Invalid model selection"
    exit 1
fi

SELECTED_MODEL="${MODELS[$((model_choice-1))]}"
echo "✅ Selected model: $SELECTED_MODEL"
echo ""

# Extract configuration directly from JSON
OUTPUT_FOLDER=$(python3 -c "import json; print(json.load(open('$TRAIN_CONFIG_FILE'))['$SELECTED_MODEL']['output_folder'])")
LOG_FORMAT_JSON=$(python3 -c "import json; print(json.dumps(json.load(open('$TRAIN_CONFIG_FILE'))['$SELECTED_MODEL'].get('log_format', 'tgsformer')))")
FRAMES_PER_ITER=($(python3 -c "import json; print(' '.join(map(str, json.load(open('$TRAIN_CONFIG_FILE'))['$SELECTED_MODEL']['frames_per_iter'])))"))

# Get log names
LOG_NAMES=($(python3 -c "import json; print(' '.join(json.load(open('$TRAIN_CONFIG_FILE'))['$SELECTED_MODEL']['logs'].keys()))"))

# Build configs array and validate
echo "🔍 Checking log files..."
CONFIGS=()
for log_name in "${LOG_NAMES[@]}"; do
    # Get log info (could be string or dict)
    LOG_INFO=$(python3 -c "import json; info=json.load(open('$TRAIN_CONFIG_FILE'))['$SELECTED_MODEL']['logs']['$log_name']; print(json.dumps(info) if isinstance(info, dict) else info)")
    
    # Extract path
    LOG_PATH=$(python3 -c "import json; info=json.loads('$LOG_INFO') if '$LOG_INFO'.startswith('{') else '$LOG_INFO'; print(info.get('path', info) if isinstance(info, dict) else info)")
    
    if [ ! -f "$LOG_PATH" ]; then
        echo "❌ Missing log: $LOG_PATH"
        exit 1
    fi
    echo "✅ $log_name: $LOG_PATH"
    
    # Add to configs array
    CONFIGS+=("$log_name:$LOG_INFO")
done

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

echo ""
echo "🎯 Generating plots..."
echo "   Output folder: $OUTPUT_FOLDER"
echo "   Log format: $LOG_FORMAT_JSON"
echo "   Frames per iter: ${FRAMES_PER_ITER[@]}"
echo ""

# Note: sequences process 30 frames/iter, mono processes 1 frame/iter
python3 vis_train.py \
    --configs "${CONFIGS[@]}" \
    --output "$OUTPUT_FOLDER/loss_comparison.png" \
    --figsize 20 10 \
    --smooth \
    --log-format "$LOG_FORMAT_JSON" \
    --frames-per-iter ${FRAMES_PER_ITER[@]} 
 
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Done! Output plots saved in $OUTPUT_FOLDER/:" 
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
        echo "💡 Run 'xdg-open $OUTPUT_FOLDER/loss_comparison_loss.png' to view"
    fi
else
    echo "❌ Failed!"
    exit 1
fi 
 