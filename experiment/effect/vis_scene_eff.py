#!/usr/bin/env python3
"""Scene Efficiency Visualization Tool - Accumulated Anchors vs Frame"""

import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from typing import Dict, Tuple, List

# Plot style configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['legend.fontsize'] = 28

class AnchorParser:
    """Parse scene efficiency logs to extract accumulated anchors"""
    
    def __init__(self):
        # Extract accumulated anchors data
        # Format: Frame  0: Avg=  2175.1, Std=  113.4 (TOTAL accumulated, across 16 scenes)
        self.anchor_pattern = re.compile(
            r'Frame\s+(\d+):\s+Avg=\s*([\d.]+),\s+Std=\s*([\d.]+)\s+\(TOTAL accumulated'
        )
    
    def parse_file(self, file_path: str) -> Dict:
        """Parse efficiency file to extract accumulated anchors"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = {
            'frames': [],
            'avg_anchors': [],
            'std_anchors': []
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Flag to identify we're in the ACCUMULATED section
        in_accumulated_section = False
        
        for line in lines:
            if '--- Average ACCUMULATED Anchors at Each Frame Position ---' in line:
                in_accumulated_section = True
                continue
            
            if in_accumulated_section:
                match = self.anchor_pattern.search(line)
                if match:
                    frame = int(match.group(1))
                    avg = float(match.group(2))
                    std = float(match.group(3))
                    
                    data['frames'].append(frame)
                    data['avg_anchors'].append(avg)
                    data['std_anchors'].append(std)
        
        return data

class AnchorCurvePlotter:
    """Plot accumulated anchors vs frame curves"""
    
    def __init__(self):
        # Colors matching vis_train.py style
        self.colors = ['#6BA3D0', '#91C788', '#E89C7D', '#F0B27A', '#9467bd']
        self.linestyles = ['-', '-', '-', '-', '-']
        self.markers = ['o', 's', '^', 'D', 'v']
        
    def plot_multiple_configs(self, config_data: Dict[str, Dict], 
                            save_path: str = 'anchor_curves.png',
                            figsize: Tuple[int, int] = (10, 10),
                            dpi: int = 600,
                            show_std_band: bool = True):
        """Plot accumulated anchors curves (6 types: with/without numbers, 1:1, 4:3, and 16:9)
        
        Args:
            show_std_band: If True, show standard deviation shaded region
        """
        
        base_path = save_path.rsplit('.', 1)[0]
        ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
        
        # Determine common marker frames (every 5 frames: 0, 5, 10, 15, 20, 25, 30)
        common_marker_frames = set(range(0, 35, 5))  # 0, 5, 10, 15, 20, 25, 30
        
        # Save plot WITH numbers (1:1 ratio)
        fig1 = plt.figure(figsize=figsize)
        ax = fig1.add_subplot(111)
        fig1.patch.set_facecolor('white')
        ax.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]
            
            frames = np.array(data['frames'])
            anchors = np.array(data['avg_anchors'])
            std = np.array(data['std_anchors'])
            
            # Plot std band if requested
            if show_std_band:
                ax.fill_between(frames, anchors - std, anchors + std,
                               color=color, alpha=0.25)
            
            # Show markers at common frame positions (0, 5, 10, 15, 20, 25, 30)
            marker_indices = [idx for idx, fr in enumerate(frames) if fr in common_marker_frames]
            ax.plot(frames, anchors, color=color, linewidth=12.0, 
                   linestyle=linestyle, marker=marker, markersize=35,
                   markevery=marker_indices, alpha=0.9)
        
        self._beautify_axis(ax)
        plt.tight_layout()
        main_path = f"{base_path}.{ext}"
        plt.savefig(main_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved plot with numbers (1:1): {main_path}")
        plt.close()
        
        # Save plot WITHOUT numbers (1:1 ratio)
        fig2 = plt.figure(figsize=figsize)
        ax_clean = fig2.add_subplot(111)
        fig2.patch.set_facecolor('white')
        ax_clean.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]
            
            frames = np.array(data['frames'])
            anchors = np.array(data['avg_anchors'])
            std = np.array(data['std_anchors'])
            
            # Plot std band if requested
            if show_std_band:
                ax_clean.fill_between(frames, anchors - std, anchors + std,
                                     color=color, alpha=0.25)
            
            marker_indices = [idx for idx, fr in enumerate(frames) if fr in common_marker_frames]
            ax_clean.plot(frames, anchors, color=color, linewidth=12.0, 
                         linestyle=linestyle, marker=marker, markersize=35,
                         markevery=marker_indices, alpha=0.9)
        
        self._beautify_axis_no_numbers(ax_clean)
        plt.tight_layout()
        clean_path = f"{base_path}_clean.{ext}"
        plt.savefig(clean_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved clean plot (1:1): {clean_path}")
        plt.close()
        
        # Save plot WITH numbers (4:3 ratio)
        fig3 = plt.figure(figsize=(figsize[0] * 4/3, figsize[1]))
        ax_43 = fig3.add_subplot(111)
        fig3.patch.set_facecolor('white')
        ax_43.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]
            
            frames = np.array(data['frames'])
            anchors = np.array(data['avg_anchors'])
            std = np.array(data['std_anchors'])
            
            # Plot std band if requested
            if show_std_band:
                ax_43.fill_between(frames, anchors - std, anchors + std,
                                  color=color, alpha=0.25)
            
            marker_indices = [idx for idx, fr in enumerate(frames) if fr in common_marker_frames]
            ax_43.plot(frames, anchors, color=color, linewidth=12.0, 
                      linestyle=linestyle, marker=marker, markersize=45,
                      markevery=marker_indices, alpha=0.9)
        
        self._beautify_axis(ax_43)
        plt.tight_layout()
        path_43 = f"{base_path}_4x3.{ext}"
        plt.savefig(path_43, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved plot with numbers (4:3): {path_43}")
        plt.close()
        
        # Save plot WITHOUT numbers (4:3 ratio)
        fig4 = plt.figure(figsize=(figsize[0] * 4/3, figsize[1]))
        ax_clean_43 = fig4.add_subplot(111)
        fig4.patch.set_facecolor('white')
        ax_clean_43.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]
            
            frames = np.array(data['frames'])
            anchors = np.array(data['avg_anchors'])
            std = np.array(data['std_anchors'])
            
            # Plot std band if requested
            if show_std_band:
                ax_clean_43.fill_between(frames, anchors - std, anchors + std,
                                        color=color, alpha=0.25)
            
            marker_indices = [idx for idx, fr in enumerate(frames) if fr in common_marker_frames]
            ax_clean_43.plot(frames, anchors, color=color, linewidth=12.0, 
                           linestyle=linestyle, marker=marker, markersize=45,
                           markevery=marker_indices, alpha=0.9)
        
        self._beautify_axis_no_numbers(ax_clean_43)
        plt.tight_layout()
        clean_43_path = f"{base_path}_clean_4x3.{ext}"
        plt.savefig(clean_43_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved clean plot (4:3): {clean_43_path}")
        plt.close()
        
        # Save plot WITH numbers (16:9 ratio)
        fig5 = plt.figure(figsize=(figsize[0] * 16/9, figsize[1]))
        ax_169 = fig5.add_subplot(111)
        fig5.patch.set_facecolor('white')
        ax_169.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]
            
            frames = np.array(data['frames'])
            anchors = np.array(data['avg_anchors'])
            std = np.array(data['std_anchors'])
            
            # Plot std band if requested
            if show_std_band:
                ax_169.fill_between(frames, anchors - std, anchors + std,
                                   color=color, alpha=0.25)
            
            marker_indices = [idx for idx, fr in enumerate(frames) if fr in common_marker_frames]
            ax_169.plot(frames, anchors, color=color, linewidth=12.0, 
                      linestyle=linestyle, marker=marker, markersize=48,
                      markevery=marker_indices, alpha=0.9)
        
        self._beautify_axis(ax_169)
        plt.tight_layout()
        path_169 = f"{base_path}_16x9.{ext}"
        plt.savefig(path_169, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved plot with numbers (16:9): {path_169}")
        plt.close()
        
        # Save plot WITHOUT numbers (16:9 ratio)
        fig6 = plt.figure(figsize=(figsize[0] * 16/9, figsize[1]))
        ax_clean_169 = fig6.add_subplot(111)
        fig6.patch.set_facecolor('white')
        ax_clean_169.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]
            
            frames = np.array(data['frames'])
            anchors = np.array(data['avg_anchors'])
            std = np.array(data['std_anchors'])
            
            # Plot std band if requested
            if show_std_band:
                ax_clean_169.fill_between(frames, anchors - std, anchors + std,
                                         color=color, alpha=0.25)
            
            marker_indices = [idx for idx, fr in enumerate(frames) if fr in common_marker_frames]
            ax_clean_169.plot(frames, anchors, color=color, linewidth=12.0, 
                           linestyle=linestyle, marker=marker, markersize=48,
                           markevery=marker_indices, alpha=0.9)
        
        self._beautify_axis_no_numbers(ax_clean_169)
        plt.tight_layout()
        clean_169_path = f"{base_path}_clean_16x9.{ext}"
        plt.savefig(clean_169_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved clean plot (16:9): {clean_169_path}")
        plt.close()
    
    def _beautify_axis(self, ax):
        """Beautify axis - clean minimal style with numbers"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax.grid(True, color='white', linewidth=10.0, linestyle='-', alpha=1.0, axis='both')
        ax.set_axisbelow(True)
        
        # X-axis: show 0, 5, 10, 15, 20, 25, 30 (7 grid lines)
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
        
        # X-axis: show 0, 5, 10, 15, 20, 25, 30 (7 grid lines)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        
        # Hide tick labels and ticks
        ax.tick_params(axis='both', length=0, width=0, labelsize=0)

def main():
    parser = argparse.ArgumentParser(description='Scene Efficiency Visualization - Accumulated Anchors')
    parser.add_argument('--configs', nargs='+', required=True,
                       help='Config format: name1:path1 name2:path2')
    parser.add_argument('--embodiedocc-anchors', type=float, default=None,
                       help='Fixed anchor count for EmbodiedOcc baseline (e.g., 16200)')
    parser.add_argument('--output', default='anchor_comparison.png')
    parser.add_argument('--figsize', nargs=2, type=int, default=[10, 10])
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--no-std-band', action='store_true',
                       help='Do not show standard deviation shaded region')
    
    args = parser.parse_args()
    
    config_paths = {}
    for config_str in args.configs:
        name, path = config_str.split(':', 1)
        config_paths[name] = path
    
    parser_obj = AnchorParser()
    config_data = {}
    
    for config_name, file_path in config_paths.items():
        data = parser_obj.parse_file(file_path)
        config_data[config_name] = data
        
        if data['frames']:
            print(f"{config_name}: {len(data['frames'])} frame positions, "
                  f"max anchors: {data['avg_anchors'][-1]:.1f}")
        else:
            print(f"{config_name}: No data found")
    
    # Add EmbodiedOcc baseline if specified
    if args.embodiedocc_anchors is not None:
        # Get frame range from existing configs
        max_frame = 0
        for data in config_data.values():
            if data['frames']:
                max_frame = max(max_frame, max(data['frames']))
        
        if max_frame > 0:
            # Create constant line at EmbodiedOcc anchor count
            config_data['EmbodiedOcc'] = {
                'frames': list(range(max_frame + 1)),
                'avg_anchors': [args.embodiedocc_anchors] * (max_frame + 1),
                'std_anchors': [0.0] * (max_frame + 1)
            }
            print(f"EmbodiedOcc: Fixed baseline at {args.embodiedocc_anchors:.1f} anchors")
    
    plotter = AnchorCurvePlotter()
    plotter.plot_multiple_configs(
        config_data, 
        save_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        show_std_band=not args.no_std_band
    )

    print(f"All plots saved!")

if __name__ == "__main__":
    main()
