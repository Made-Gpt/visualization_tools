#!/bin/bash
# Latency vs mIoU Visualization Script
# Usage: bash vis_latency.sh

set -e

CONFIG_FILE="effect_config.json"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# List available configurations
echo "📋 Available configurations in $CONFIG_FILE:"
echo ""

# Extract config names using Python
config_names=($(python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    configs = json.load(f)
    for i, (name, config) in enumerate(configs.items(), 1):
        desc = config.get('description', 'No description')
        print(f'{i}. {name}')
        print(f'   Description: {desc}', file=__import__('sys').stderr)
" 2>&1 | tee /dev/stderr | grep "^[0-9]" | cut -d' ' -f2))

echo ""
echo "Select a configuration (1-${#config_names[@]}):"
read -r choice

# Validate choice
if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#config_names[@]}" ]; then
    echo "❌ Invalid choice"
    exit 1
fi

CONFIG_NAME="${config_names[$((choice-1))]}"

echo ""
echo "🚀 Generating latency vs mIoU plot for: $CONFIG_NAME"
echo ""

# Run Python script
python3 vis_latency.py \
    --config "$CONFIG_NAME" \
    --figsize 10 10 \
    --dpi 600

echo ""
echo "✅ Done! Output plots saved in effect/:"
echo ""
echo "   📊 latency_miou metric:"
echo "      - latency_miou_4x3.png (with numbers, 4:3)"
echo "      - latency_miou_clean_4x3.png (no numbers, 4:3)"
echo "      - latency_miou_16x9.png (with numbers, 16:9)"
echo "      - latency_miou_clean_16x9.png (no numbers, 16:9)"
echo ""
echo "💡 Run 'xdg-open effect/' to view all plots"
