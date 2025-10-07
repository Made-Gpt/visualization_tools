#!/bin/bash

# Configuration script for visualizing Gaussian curves
# Usage: ./vis_curve.sh

# ==================== CONFIGURATION ====================
# Save mode: "combined", "separate", or "both"
SAVE_MODE="combined"

# Envelope mode: "none", "blue_envelope", or "curve_envelope"
# - "none": Just show original curves
# - "blue_envelope": Add thick line with specified color showing maximum values across curves
# - "curve_envelope": Add thick line with colors from the curve that has maximum value at each point
ENVELOPE_MODE="curve_envelope" 

# Envelope configuration (only used when ENVELOPE_MODE is not "none")
ENVELOPE_COLOR="darkblue"    # Color for blue_envelope mode
ENVELOPE_WIDTH=20            # Line width for envelope (make it more obvious)
ENVELOPE_ALPHA=0.8          # Transparency for envelope (0.0-1.0)
 
# Base save path (without extension)
SAVE_PATH="./output/gaussian_curves"

# Figure size (width,height)
if [ "$ENVELOPE_MODE" = "curve_envelope" ]; then
    FIGURE_SIZE="18,6"
else
    FIGURE_SIZE="12,6"
fi

# ==================== VALID COLORS ====================
# Here are all valid matplotlib color names you can use:
# Basic colors: 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'
# Extended colors: 'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond'
# 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk'
# 'crimson', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta'
# 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue'
# 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue'
# 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod'
# 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush'
# 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray'
# 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue'
# 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue'
# 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise'
# 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace'
# 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred'
# 'papayawhip', 'peachpuff', 'peru', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown'
# 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray'
# 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat'
# 'white', 'whitesmoke', 'yellowgreen'
# You can also use hex colors like '#FF0000' for red, '#00FF00' for green, etc.

# ==================== PARAMETER EXPLANATION ====================
# New parameters added:
# - "height": Controls curve height (1.0 = normal, >1.0 = taller, <1.0 = shorter)
# - "width": Controls curve width (1.0 = normal, >1.0 = wider, <1.0 = narrower)  
# - "line_alpha": Controls line transparency (0.0 = invisible, 1.0 = solid)
# - "fill_alpha": Controls fill transparency (0.0 = no fill, 1.0 = solid fill)
#
# Envelope parameters:
# - ENVELOPE_COLOR: Color for blue_envelope mode (ignored for curve_envelope)
# - ENVELOPE_WIDTH: Line width for envelope
# - ENVELOPE_ALPHA: Transparency for envelope line
#
# Beautiful Red-Green-Yellow combination used:
# - Crimson Red: #DC143C (vibrant red)
# - Forest Green: #228B22 (deep green)
# - Gold Yellow: #FFD700 (bright gold)

# ==================== CURVE CONFIGURATION ====================
# Create JSON config file for curves
CONFIG_FILE="./curve_config.json"

# ==================== CURVE CONFIGURATION ====================
# Create JSON config file for curves based on envelope mode
CONFIG_FILE="./curve_config.json"

if [ "$ENVELOPE_MODE" = "blue_envelope" ]; then
    cat > "$CONFIG_FILE" << 'EOF'
[ 
    {
        "x_min": -8,
        "x_max": 8,
        "mean": 3.0,
        "std_dev": 1,
        "height": 1.0,
        "width": 1.8,
        "color": "#EA6B66",
        "label": "Crimson Red",
        "fill_alpha": 0.15,
        "line_alpha": 0.8,
        "line_width": 10
    },
    { 
        "x_min": -8,
        "x_max": 8,
        "mean": -0.2,
        "std_dev": 1,
        "height": 0.35,
        "width": 1.0,
        "color": "#228B22",
        "label": "Forest Green",
        "fill_alpha": 0.2,
        "line_alpha": 0.5,
        "line_width": 10
    },
    { 
        "x_min": -8,
        "x_max": 8,
        "mean": -2.0,
        "std_dev": 1,
        "height": 0.30,
        "width": 1.8,
        "color": "#FFE599",
        "label": "Gold Yellow",
        "fill_alpha": 0.25,
        "line_alpha": 0.8,
        "line_width": 10
    } 
]
EOF
elif [ "$ENVELOPE_MODE" = "curve_envelope" ]; then
    cat > "$CONFIG_FILE" << 'EOF'
[
    {
        "x_min": -8,
        "x_max": 8,
        "mean": 2.2,
        "std_dev": 1,
        "height": 0.72,
        "width": 1.8,
        "color": "#EA6B66",
        "label": "Crimson Red",
        "fill_alpha": 0.15,
        "line_alpha": 0.8,
        "line_width": 10
    },
    { 
        "x_min": -8,
        "x_max": 8,
        "mean": -0.2,
        "std_dev": 1,
        "height": 0.40,
        "width": 1.0,
        "color": "#228B22",
        "label": "Forest Green",
        "fill_alpha": 0.2,
        "line_alpha": 0.5,
        "line_width": 10
    },
    { 
        "x_min": -8,
        "x_max": 8,
        "mean": -2.0,
        "std_dev": 1,
        "height": 0.72,
        "width": 1.8,
        "color": "#FFE599",
        "label": "Gold Yellow",
        "fill_alpha": 0.25,
        "line_alpha": 0.8,
        "line_width": 10
    } 
]
EOF
else
    # Default configuration for "none" envelope mode or any other mode
    cat > "$CONFIG_FILE" << 'EOF'
[
    {
        "x_min": -8,
        "x_max": 8,
        "mean": 2.0,
        "std_dev": 1,
        "height": 0.6,
        "width": 1.8,
        "color": "#EA6B66",
        "label": "Crimson Red",
        "fill_alpha": 0.15,
        "line_alpha": 0.8,
        "line_width": 10
    },
    { 
        "x_min": -8,
        "x_max": 8,
        "mean": -0.2,
        "std_dev": 1,
        "height": 0.32,
        "width": 1.0,
        "color": "#228B22",
        "label": "Forest Green",
        "fill_alpha": 0.2,
        "line_alpha": 0.5,
        "line_width": 10
    },
    { 
        "x_min": -8,
        "x_max": 8,
        "mean": -2.0,
        "std_dev": 1,
        "height": 0.40,
        "width": 1.8,
        "color": "#FFE599",
        "label": "Gold Yellow", 
        "fill_alpha": 0.25,
        "line_alpha": 0.8,
        "line_width": 10
    } 
]
EOF
fi


# ==================== EXECUTION ====================
echo "=== Gaussian Curve Visualization ==="
echo "Configuration:"
echo "  Save mode: $SAVE_MODE (combined=all in one, separate=individual files, both=both options)"
echo "  Envelope mode: $ENVELOPE_MODE"
if [ "$ENVELOPE_MODE" != "none" ]; then
    echo "  Envelope color: $ENVELOPE_COLOR"
    echo "  Envelope width: $ENVELOPE_WIDTH"
    echo "  Envelope alpha: $ENVELOPE_ALPHA"
fi
echo "  Save path: $SAVE_PATH"
echo "  Figure size: $FIGURE_SIZE"
echo "  Config file: $CONFIG_FILE"
echo ""

# Create output directory
mkdir -p "$(dirname "$SAVE_PATH")"

# Build Python command with envelope parameters
PYTHON_CMD="python3 vis_curve.py --config \"$CONFIG_FILE\" --save_mode \"$SAVE_MODE\" --save_path \"$SAVE_PATH\" --figure_size \"$FIGURE_SIZE\" --envelope_mode \"$ENVELOPE_MODE\""

if [ "$ENVELOPE_MODE" != "none" ]; then
    PYTHON_CMD="$PYTHON_CMD --envelope_color \"$ENVELOPE_COLOR\" --envelope_width $ENVELOPE_WIDTH --envelope_alpha $ENVELOPE_ALPHA"
fi

# Run the Python script
eval $PYTHON_CMD

echo ""
echo "=== Visualization Complete ==="

# Optional: Clean up config file
# rm "$CONFIG_FILE"

# [
#     {
#         "x_min": -8,
#         "x_max": 8,
#         "mean": 2.0,
#         "std_dev": 1,
#         "height": 1.0,
#         "width": 1.8,
#         "color": "#EA6B66",
#         "label": "Crimson Red",
#         "fill_alpha": 0.15,
#         "line_alpha": 0.8,
#         "line_width": 10
#     },
#     { 
#         "x_min": -8,
#         "x_max": 8,
#         "mean": -0.2,
#         "std_dev": 1,
#         "height": 0.35,
#         "width": 1.0,
#         "color": "#228B22",
#         "label": "Forest Green",
#         "fill_alpha": 0.2,
#         "line_alpha": 0.5,
#         "line_width": 10
#     },
#     { 
#         "x_min": -8,
#         "x_max": 8,
#         "mean": -2.0,
#         "std_dev": 1,
#         "height": 0.35,
#         "width": 1.8,
#         "color": "#FFE599",
#         "label": "Gold Yellow",
#         "fill_alpha": 0.25,
#         "line_alpha": 0.8,
#         "line_width": 10
#     } 
# ]

# [
#     {
#         "x_min": -8,
#         "x_max": 8,
#         "mean": 2.0,
#         "std_dev": 1,
#         "height": 0.6,
#         "width": 1.8,
#         "color": "#EA6B66",
#         "label": "Crimson Red",
#         "fill_alpha": 0.15,
#         "line_alpha": 0.8,
#         "line_width": 10
#     },
#     { 
#         "x_min": -8,
#         "x_max": 8,
#         "mean": -0.2,
#         "std_dev": 1,
#         "height": 0.32,
#         "width": 1.0,
#         "color": "#228B22",
#         "label": "Forest Green",
#         "fill_alpha": 0.2,
#         "line_alpha": 0.5,
#         "line_width": 10
#     },
#     { 
#         "x_min": -8,
#         "x_max": 8,
#         "mean": -2.0,
#         "std_dev": 1,
#         "height": 0.40,
#         "width": 1.8,
#         "color": "#FFE599",
#         "label": "Gold Yellow", 
#         "fill_alpha": 0.25,
#         "line_alpha": 0.8,
#         "line_width": 10
#     } 
# ]