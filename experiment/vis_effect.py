#!/usr/bin/env python3
"""Efficiency Visualization Tool - Anchors, Features, and Memory Usage"""

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

class EfficiencyParser:
    """Parse scene efficiency logs to extract all metrics"""
    
    def __init__(self):
        # Pattern for accumulated anchors
        self.anchor_pattern = re.compile(
            r'Frame\s+(\d+):\s+Avg=\s*([\d.]+),\s+Std=\s*([\d.]+)\s+\(TOTAL accumulated'
        )
        
        # Pattern for instance features
        self.feature_pattern = re.compile(
            r'Frame\s+(\d+):\s+Avg=\s*([\d.]+),\s+Std=\s*([\d.]+)\s+\(Features'
        )
        
        # No regex needed - we'll parse by splitting
    
    def parse_file(self, file_path: str) -> Dict:
        """Parse efficiency file to extract all metrics"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = {
            'anchors': {'frames': [], 'avg': [], 'std': []},
            'features': {'frames': [], 'avg': [], 'std': []},
            'memory': {'frames': [], 'avg': [], 'std': []}
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        in_anchor_section = False
        in_feature_section = False
        in_memory_section = False
        
        for line in lines:
            # Check section headers
            if '--- Average ACCUMULATED Anchors at Each Frame Position ---' in line:
                in_anchor_section = True
                in_feature_section = False
                in_memory_section = False
                continue
            elif '--- Average ACCUMULATED Instance Features at Each Frame Position ---' in line:
                in_anchor_section = False
                in_feature_section = True
                in_memory_section = False
                continue
            elif '--- Memory Usage at Different Frame Positions ---' in line:
                in_anchor_section = False
                in_feature_section = False
                in_memory_section = True
                continue
            elif line.strip() == '':
                # Empty line - continue (don't reset sections)
                continue
            elif line.startswith('---') and not any(kw in line for kw in ['ACCUMULATED Anchors', 'Instance Features', 'Memory Usage']):
                # New section starting (that's not one of our three sections)
                # Lines starting with '---' are section headers, not table borders
                in_anchor_section = False
                in_feature_section = False
                in_memory_section = False
                continue
            
            # Parse anchors
            if in_anchor_section:
                match = self.anchor_pattern.search(line)
                if match:
                    frame = int(match.group(1))
                    avg = float(match.group(2))
                    std = float(match.group(3))
                    data['anchors']['frames'].append(frame)
                    data['anchors']['avg'].append(avg)
                    data['anchors']['std'].append(std)
            
            # Parse features
            elif in_feature_section:
                match = self.feature_pattern.search(line)
                if match:
                    frame = int(match.group(1))
                    avg = float(match.group(2))
                    std = float(match.group(3))
                    data['features']['frames'].append(frame)
                    data['features']['avg'].append(avg)
                    data['features']['std'].append(std)
            
            # Parse memory - simple split approach
            elif in_memory_section:
                if 'INFO' in line and '±' in line:
                    parts = line.split()
                    # Find the frame number (first pure digit after INFO)
                    frame_idx = None
                    for i, p in enumerate(parts):
                        if p.isdigit():
                            frame_idx = i
                            break
                    
                    if frame_idx is not None and frame_idx + 3 < len(parts):
                        try:
                            frame = int(parts[frame_idx])
                            # Gaussian (MiB) is the 4th column after frame
                            # parts[frame_idx+1] = Allocated(GB): X.XXX±X.XXX
                            # parts[frame_idx+2] = Peak(GB): X.XXX±X.XXX  
                            # parts[frame_idx+3] = Gaussian(MiB): X.XX±X.XX
                            gaussian_str = parts[frame_idx + 3]  # e.g., "0.23±0.06"
                            if '±' in gaussian_str:
                                avg_str, std_str = gaussian_str.split('±')
                                avg = float(avg_str)
                                std = float(std_str)
                                data['memory']['frames'].append(frame)
                                data['memory']['avg'].append(avg)
                                data['memory']['std'].append(std)
                        except (ValueError, IndexError):
                            pass
        
        return data

class EfficiencyCurvePlotter:
    """Plot efficiency curves with uncertainty bands"""
    
    def __init__(self):
        self.default_colors = ['#6BA3D0', '#91C788', '#E89C7D', '#F0B27A', '#9467bd']
        self.default_markers = ['o', 's', '^', 'D', 'v']
        self.linestyles = ['-', '-', '-', '-', '-']
    
    def plot_metrics(self, methods_data: Dict, metric: str,
                    save_path: str, figsize: Tuple[int, int] = (10, 10),
                    dpi: int = 600, show_std_band: bool = True):
        """Plot a specific metric for all methods (4 variants: 4:3 and 16:9, with/without numbers)
        
        Args:
            show_std_band: If True, show standard deviation shaded region
        """
        
        base_path = save_path.rsplit('.', 1)[0]
        ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
        
        # Metric titles
        metric_titles = {
            'anchors': 'Accumulated Anchors',
            'features': 'Instance Features',
            'memory': 'Gaussian Memory (MiB)'
        }
        
        # Common marker frames
        common_marker_frames = set(range(0, 35, 5))
        
        # Get global min/max for consistent Y-axis across all plots
        all_mins = []
        all_maxs = []
        all_avg_maxs = []  # Track max avg separately for tighter bounds
        all_std_maxs = []  # Track max std separately
        
        for method_name, method_info in methods_data.items():
            data = method_info['data'][metric]
            if data['avg']:
                avg = np.array(data['avg'])
                std = np.array(data['std'])
                all_mins.append(np.min(avg - std))
                all_maxs.append(np.max(avg + std))
                all_avg_maxs.append(np.max(avg))
                all_std_maxs.append(np.max(std))
        
        if all_mins and all_maxs:
            y_min_global = min(all_mins)
            
            # For features and memory: use tighter upper bound (max_avg + 1/3 * max_std)
            # For anchors: keep original full range (max_avg + max_std)
            if metric in ['features', 'memory']:
                max_avg = max(all_avg_maxs)
                max_std = max(all_std_maxs)
                y_max_global = max_avg + max_std / 3.0
            else:
                y_max_global = max(all_maxs)
            
            y_margin = (y_max_global - y_min_global) * 0.1
            y_min_global -= y_margin
            y_max_global += y_margin
        else:
            y_min_global, y_max_global = None, None
        
        # 1. WITH numbers (4:3)
        self._plot_single_variant(
            methods_data, metric, (figsize[0] * 4/3, figsize[1]), common_marker_frames,
            f"{base_path}_{metric}_4x3.{ext}", dpi, with_numbers=True,
            markersize=45, y_limits=(y_min_global, y_max_global), show_std_band=show_std_band
        )
        
        # 2. WITHOUT numbers (4:3)
        self._plot_single_variant(
            methods_data, metric, (figsize[0] * 4/3, figsize[1]), common_marker_frames,
            f"{base_path}_{metric}_clean_4x3.{ext}", dpi, with_numbers=False,
            markersize=45, y_limits=(y_min_global, y_max_global), show_std_band=show_std_band
        )
        
        # 3. WITH numbers (16:9)
        self._plot_single_variant(
            methods_data, metric, (figsize[0] * 16/9, figsize[1]), common_marker_frames,
            f"{base_path}_{metric}_16x9.{ext}", dpi, with_numbers=True,
            markersize=48, y_limits=(y_min_global, y_max_global), show_std_band=show_std_band
        )
        
        # 4. WITHOUT numbers (16:9)
        self._plot_single_variant(
            methods_data, metric, (figsize[0] * 16/9, figsize[1]), common_marker_frames,
            f"{base_path}_{metric}_clean_16x9.{ext}", dpi, with_numbers=False,
            markersize=48, y_limits=(y_min_global, y_max_global), show_std_band=show_std_band
        )
    
    def _plot_single_variant(self, methods_data: Dict, metric: str,
                            figsize: Tuple, common_marker_frames: set,
                            output_path: str, dpi: int, with_numbers: bool,
                            markersize: int = 35, y_limits: Tuple = (None, None),
                            show_std_band: bool = True):
        """Plot a single variant of the metric"""
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#E8EDF2')
        
        for method_name, method_info in methods_data.items():
            data = method_info['data'][metric]
            
            if not data['avg']:
                continue
            
            frames = np.array(data['frames'])
            avg = np.array(data['avg'])
            std = np.array(data['std'])
            
            color = method_info.get('color', self.default_colors[0])
            marker = method_info.get('marker', self.default_markers[0])
            linestyle = self.linestyles[0]
            
            # Plot uncertainty band if requested
            if show_std_band:
                ax.fill_between(frames, avg - std, avg + std,
                               color=color, alpha=0.25) 
            
            # Plot main line
            marker_indices = [idx for idx, fr in enumerate(frames) if fr in common_marker_frames]
            ax.plot(frames, avg, color=color, linewidth=12.0,
                   linestyle=linestyle, marker=marker, markersize=markersize,
                   markevery=marker_indices, alpha=0.9)
            
            # Draw horizontal dash line at the end value (final point)
            # x range: slightly wider than the curve's start/end points
            if len(avg) > 0 and len(frames) > 0:
                final_value = avg[-1]
                x_start = frames[0]
                x_end = frames[-1]
                x_range = x_end - x_start 
                # Extend by 5% on each side
                x_min = x_start - 0.05 * x_range
                x_max = x_end + 0.05 * x_range
                ax.hlines(y=final_value, xmin=x_min, xmax=x_max, 
                         color=color, linestyle='--', linewidth=6.0, 
                         alpha=0.85, zorder=1)
        
        # Set Y-axis limits if provided
        if y_limits[0] is not None and y_limits[1] is not None:
            ax.set_ylim(y_limits[0], y_limits[1])
        
        if with_numbers:
            self._beautify_axis(ax)
        else:
            self._beautify_axis_no_numbers(ax)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        
        variant_desc = "with numbers" if with_numbers else "no numbers"
        aspect = f"{figsize[0]:.1f}:{figsize[1]:.1f}"
        print(f"Saved {metric} plot ({variant_desc}, {aspect}): {output_path}")
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

def main():
    parser = argparse.ArgumentParser(description='Efficiency Visualization Tool')
    parser.add_argument('--config', required=True,
                       help='Configuration name from effect_config.json')
    parser.add_argument('--figsize', nargs=2, type=int, default=[10, 10])
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--no-std-band', action='store_true',
                       help='Do not show standard deviation shaded region')
    
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
    metrics = config['metrics']
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Parse all methods
    parser_obj = EfficiencyParser()
    methods_data = {}
    
    for method_name, method_config in methods.items():
        file_path = method_config['path']
        
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping {method_name}: file not found ({file_path})")
            continue
        
        data = parser_obj.parse_file(file_path)

        # If features are missing, fall back to anchors (frames/avg/std)
        # This makes the features plot robust/scalable when some methods
        # do not report instance features separately.
        if (not data['features']['avg'] or len(data['features']['avg']) == 0) and data['anchors']['avg']:
            data['features']['frames'] = list(data['anchors']['frames'])
            data['features']['avg'] = list(data['anchors']['avg'])
            data['features']['std'] = list(data['anchors']['std'])

        methods_data[method_name] = {
            'data': data,
            'color': method_config.get('color'),
            'marker': method_config.get('marker')
        }
        
        # Print summary
        print(f"✅ {method_name}:")
        if data['anchors']['avg']:
            print(f"   Anchors: {len(data['anchors']['frames'])} frames, "
                  f"range {data['anchors']['avg'][0]:.1f} → {data['anchors']['avg'][-1]:.1f}")
        if data['features']['avg']:
            print(f"   Features: {len(data['features']['frames'])} frames, "
                  f"range {data['features']['avg'][0]:.1f} → {data['features']['avg'][-1]:.1f}")
        if data['memory']['avg']:
            print(f"   Memory: {len(data['memory']['frames'])} frames, "
                  f"range {data['memory']['avg'][0]:.1f} → {data['memory']['avg'][-1]:.1f} MiB")
        else:
            print(f"   Memory: No data (parsed {len(data['memory']['frames'])} frames)")
    
    if not methods_data:
        print("❌ No valid data found")
        return 1
    
    # Plot each metric
    plotter = EfficiencyCurvePlotter()
    
    for metric in metrics:
        # Check if any method has this metric
        has_data = any(
            methods_data[m]['data'][metric]['avg']
            for m in methods_data
        )
        
        if not has_data:
            print(f"⚠️  Skipping {metric}: no data available")
            continue
        
        save_path = os.path.join(output_folder, f"comparison.png")
        plotter.plot_metrics(methods_data, metric, save_path,
                           figsize=tuple(args.figsize), dpi=args.dpi,
                           show_std_band=not args.no_std_band)
    
    print(f"\n✅ All plots saved to {output_folder}/")
    return 0

if __name__ == "__main__":
    exit(main())
