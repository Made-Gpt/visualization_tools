#!/usr/bin/env python3
"""Latency vs mIoU Scatter Plot Visualization"""

import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
from typing import Dict, Tuple, List

# Plot style configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['legend.fontsize'] = 28

class LatencyParser:
    """Parse scene logs to extract mIoU, inference time, and parameters"""
    
    def __init__(self):
        # Pattern for single frame inference time
        self.time_pattern = re.compile(
            r'Overall Average Single Frame Inference Time:\s*([\d.]+)\s*ms'
        )
        
        # Pattern for semantic IoU
        self.miou_pattern = re.compile(
            r'Current global iou of sem is\s*([\d.]+)'
        )
        
        # Pattern for number of parameters
        self.params_pattern = re.compile(
            r'Number of params:\s*([\d]+)'
        )
    
    def parse_file(self, file_path: str) -> Dict:
        """Parse log file to extract latency, mIoU, and parameters"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        latency = None
        miou = None
        params = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract inference time
        time_match = self.time_pattern.search(content)
        if time_match:
            latency = float(time_match.group(1))
        
        # Extract semantic IoU
        miou_match = self.miou_pattern.search(content)
        if miou_match:
            miou = float(miou_match.group(1))
        
        # Extract number of parameters
        params_match = self.params_pattern.search(content)
        if params_match:
            params = int(params_match.group(1))
        
        return {
            'latency': latency,
            'miou': miou,
            'params': params
        }

class LatencyScatterPlotter:
    """Plot latency/params vs mIoU scatter plots"""
    
    def __init__(self):
        self.default_colors = ['#6BA3D0', '#91C788', '#E89C7D', '#F0B27A', '#9467bd']
        self.default_markers = ['o', 's', '^', 'D', 'v']
    
    def plot_scatter(self, methods_data: Dict,
                    save_path: str, figsize: Tuple[int, int] = (10, 10),
                    dpi: int = 600, x_metric: str = 'latency'):
        """Plot x_metric vs mIoU scatter plot (4 variants: 4:3 and 16:9, with/without numbers)
        
        Args:
            x_metric: 'latency' or 'params' to choose x-axis metric
        """
        
        base_path = save_path.rsplit('.', 1)[0]
        ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
        
        # 1. WITH numbers (4:3)
        self._plot_single_variant(
            methods_data, (figsize[0] * 4/3, figsize[1]),
            f"{base_path}_4x3.{ext}", dpi, with_numbers=True,
            markersize=3600, x_metric=x_metric
        )
        
        # 2. WITHOUT numbers (4:3)
        self._plot_single_variant(
            methods_data, (figsize[0] * 4/3, figsize[1]),
            f"{base_path}_clean_4x3.{ext}", dpi, with_numbers=False,
            markersize=3600, x_metric=x_metric
        )
        
        # 3. WITH numbers (16:9)
        self._plot_single_variant(
            methods_data, (figsize[0] * 16/9, figsize[1]),
            f"{base_path}_16x9.{ext}", dpi, with_numbers=True,
            markersize=4200, x_metric=x_metric
        )
        
        # 4. WITHOUT numbers (16:9)
        self._plot_single_variant(
            methods_data, (figsize[0] * 16/9, figsize[1]),
            f"{base_path}_clean_16x9.{ext}", dpi, with_numbers=False,
            markersize=4200, x_metric=x_metric
        )
    
    def _plot_single_variant(self, methods_data: Dict,
                            figsize: Tuple, output_path: str, dpi: int, 
                            with_numbers: bool, markersize: int = 400, x_metric: str = 'latency'):
        """Plot a single variant of the scatter plot"""
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#E8EDF2')
        
        # First pass: collect all data points for axis limits
        all_x_values = []
        all_mious = []
        for method_name, method_info in methods_data.items():
            data = method_info['data']
            x_val = data.get(x_metric)
            if x_val is not None and data['miou'] is not None:
                all_x_values.append(x_val)
                all_mious.append(data['miou'])
        
        # Calculate axis limits first
        if all_x_values and all_mious:
            x_min, x_max = min(all_x_values), max(all_x_values)
            y_min, y_max = min(all_mious), max(all_mious)
            x_range = x_max - x_min if x_max != x_min else x_max * 0.1
            y_range = y_max - y_min if y_max != y_min else y_max * 0.1
            x_limit_min = x_min - 0.36 * x_range
            x_limit_max = x_max + 0.20 * x_range 
            y_limit_min = y_min - 0.32 * y_range
            y_limit_max = y_max + 0.18 * y_range
        else:
            x_limit_min, x_limit_max = None, None
            y_limit_min, y_limit_max = None, None
        
        # Second pass: plot points and guide lines
        for method_name, method_info in methods_data.items():
            data = method_info['data']

            x_val = data.get(x_metric)
            if x_val is None or data['miou'] is None:
                continue

            miou = data['miou']
            
            color = method_info.get('color', self.default_colors[0])
            marker = method_info.get('marker', self.default_markers[0])
            
            # Draw vertical and horizontal guide lines from dot to axes
            if x_limit_min is not None and y_limit_min is not None:
                # Vertical line (from dot down to x-axis)
                ax.vlines(x=x_val, ymin=y_limit_min, ymax=miou, 
                         color=color, linestyle='--', linewidth=6.0, alpha=0.8, zorder=1)
                # Horizontal line (from dot left to y-axis)
                ax.hlines(y=miou, xmin=x_limit_min, xmax=x_val, 
                         color=color, linestyle='--', linewidth=6.0, alpha=0.8, zorder=1)
            
            # Plot scatter point (on top of lines)
            ax.scatter(x_val, miou, color=color, marker=marker, 
                      s=markersize, alpha=0.9, edgecolors='white', 
                      linewidths=2.5, zorder=3)
        
        # Set axis limits (already calculated)
        if x_limit_min is not None and x_limit_max is not None:
            ax.set_xlim(x_limit_min, x_limit_max)
            ax.set_ylim(y_limit_min, y_limit_max)
        
        if with_numbers:
            self._beautify_axis(ax)
        else:
            self._beautify_axis_no_numbers(ax)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        
        variant_desc = "with numbers" if with_numbers else "no numbers"
        aspect = f"{figsize[0]:.1f}:{figsize[1]:.1f}"
        print(f"Saved latency plot ({variant_desc}, {aspect}): {output_path}")
        plt.close()
    
    def _beautify_axis(self, ax):
        """Beautify axis - clean minimal style with numbers"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax.grid(True, color='white', linewidth=10.0, linestyle='-', alpha=1.0, axis='both')
        ax.set_axisbelow(True)
        
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        
        ax.tick_params(axis='both', length=10, width=2.5, labelsize=32)
    
    def _beautify_axis_no_numbers(self, ax):
        """Beautify axis - no tick labels"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax.grid(True, color='white', linewidth=10.0, linestyle='-', alpha=1.0, axis='both')
        ax.set_axisbelow(True) 
        
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        
        ax.tick_params(axis='both', length=0, width=0, labelsize=0)
        
        # Hide offset text (e.g., 1e8) in both axes
        ax.xaxis.offsetText.set_visible(False)
        ax.yaxis.offsetText.set_visible(False)

def main():
    parser = argparse.ArgumentParser(description='Latency vs mIoU Visualization')
    parser.add_argument('--config', required=True,
                       help='Configuration name from effect_config.json')
    parser.add_argument('--figsize', nargs=2, type=int, default=[10, 10])
    parser.add_argument('--dpi', type=int, default=600)
    
    args = parser.parse_args()
    
    # Load configuration
    config_file = 'effect_config.json'
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        return 1
    
    with open(config_file, 'r') as f:
        all_configs = json.load(f)
    
    if args.config not in all_configs:
        print(f"❌ Configuration '{args.config}' not found")
        print(f"Available: {', '.join(all_configs.keys())}")
        return 1
    
    config = all_configs[args.config]
    output_folder = config['output_folder']
    methods = config['methods']
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Parse all methods
    parser_obj = LatencyParser()
    methods_data = {}
    
    for method_name, method_config in methods.items():
        file_path = method_config['path']
        
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping {method_name}: file not found ({file_path})")
            continue
        
        data = parser_obj.parse_file(file_path)
        
        methods_data[method_name] = {
            'data': data,
            'color': method_config.get('color'),
            'marker': method_config.get('marker')
        }
        
        # Print summary
        print(f"✅ {method_name}:")
        if data['latency'] is not None:
            print(f"   Latency: {data['latency']:.2f} ms")
        else:
            print(f"   Latency: Not found")
        
        if data['miou'] is not None:
            print(f"   mIoU: {data['miou']:.4f}")
        else:
            print(f"   mIoU: Not found")
        
        if data['params'] is not None:
            print(f"   Params: {data['params']:,}")
        else:
            print(f"   Params: Not found")
    
    if not methods_data:
        print("❌ No valid data found")
        return 1
    
    # Plot latency vs mIoU
    has_latency_data = any(
        m['data']['latency'] is not None and m['data']['miou'] is not None
        for m in methods_data.values()
    )
    
    if has_latency_data:
        plotter = LatencyScatterPlotter()
        save_path = os.path.join(output_folder, f"latency_miou.png")
        plotter.plot_scatter(methods_data, save_path,
                            figsize=tuple(args.figsize), dpi=args.dpi, x_metric='latency')
        print(f"✅ Latency vs mIoU plots saved")
    else:
        print("⚠️  Skipping latency plot: no valid latency and mIoU pairs")
    
    # Plot params vs mIoU
    has_params_data = any(
        m['data']['params'] is not None and m['data']['miou'] is not None
        for m in methods_data.values()
    )
    
    if has_params_data:
        plotter = LatencyScatterPlotter()
        save_path = os.path.join(output_folder, f"params_miou.png")
        plotter.plot_scatter(methods_data, save_path,
                            figsize=tuple(args.figsize), dpi=args.dpi, x_metric='params')
        print(f"✅ Params vs mIoU plots saved")
    else:
        print("⚠️  Skipping params plot: no valid params and mIoU pairs")
    
    if not has_latency_data and not has_params_data:
        print("❌ No valid data pairs found")
        return 1
    
    print(f"\n✅ All plots saved to {output_folder}/")
    return 0

if __name__ == "__main__":
    exit(main())
