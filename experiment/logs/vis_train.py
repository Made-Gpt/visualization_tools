#!/usr/bin/env python3
"""Training Loss and mIoU Visualization Tool"""

import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from typing import Dict, Tuple

# Plot style configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['legend.fontsize'] = 28

class LossParser:
    """Parse training logs to extract Loss and mIoU"""
    
    def __init__(self):
        # Extract total Loss (skip FocalLoss, LovaszLoss, Geo_Scal_Loss)
        self.train_pattern = re.compile(
            r'\[TRAIN\]\s+Epoch\s+(\d+)\s+Iter\s+(\d+)/(\d+).*?Geo_Scal_Loss:.*?Loss:\s+([\d.]+)\s+\('
        )
        # Extract mIoU (after EVAL)
        self.miou_pattern = re.compile(
            r'Current val iou of sem is ([\d.]+)'
        )
    
    def parse_log_file(self, log_path: str) -> Dict:
        """Parse log file to extract training loss and mIoU"""
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")
        
        data = {
            'train': {'total_iter': [], 'Loss': []},
            'val': {'epochs': [], 'mIoU': []},
            'epochs': 0,
            'iters_per_epoch': 0
        }
        
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        max_epoch = 0
        iters_per_epoch = 0
        current_epoch = 0
        
        for line in lines:
            # Extract training loss
            match = self.train_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                iter_in_epoch = int(match.group(2))
                max_iter = int(match.group(3))
                loss = float(match.group(4))
                
                max_epoch = max(max_epoch, epoch)
                iters_per_epoch = max_iter
                current_epoch = epoch
                total_iter = (epoch - 1) * iters_per_epoch + iter_in_epoch + 1
                
                data['train']['total_iter'].append(total_iter)
                data['train']['Loss'].append(loss)
            
            # Extract mIoU
            miou_match = self.miou_pattern.search(line)
            if miou_match:
                miou = float(miou_match.group(1))
                data['val']['epochs'].append(current_epoch)
                data['val']['mIoU'].append(miou)
        
        data['epochs'] = max_epoch
        data['iters_per_epoch'] = iters_per_epoch
        return data

class LossCurvePlotter:
    """Plot training loss and mIoU curves"""
    
    def __init__(self):
        # Order: sequence_embed (blue), sequence_mono (green), mono (orange)
        self.colors = ['#6BA3D0', '#91C788', '#E89C7D', '#F0B27A', '#9467bd']
        self.linestyles = ['-', '-', '-', '-', '-'] 
        # Different markers for mIoU plots 
        self.markers = ['o', 's', '^', 'D', 'v']  # circle, triangle_up, square, diamond, triangle_down
        
    def plot_multiple_configs(self, config_data: Dict[str, Dict], 
                            save_path: str = 'loss_curves.png',
                            figsize: Tuple[int, int] = (20, 10),
                            dpi: int = 600,  
                            smooth: bool = True,
                            frames_info: Dict = None):
        """Plot training loss and mIoU curves (individual subplots only)"""
        
        # Skip combined figure - directly save individual subplots
        base_path = save_path.rsplit('.', 1)[0]
        ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
        
        # Save individual subplots
        base_path = save_path.rsplit('.', 1)[0]
        ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
        
        # Save left subplot (Loss)
        fig1 = plt.figure(figsize=(figsize[0]//2, figsize[1]))
        ax_loss = fig1.add_subplot(111)
        fig1.patch.set_facecolor('white')
        ax_loss.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            frames_per_iter = 1
            if frames_info and config_name in frames_info:
                frames_per_iter = frames_info[config_name]['frames_per_iter']
            
            iters = np.array(data['train']['total_iter'])
            losses = np.array(data['train']['Loss'])
            iters_scaled = iters * frames_per_iter
            
            if smooth and len(losses) > 10:
                from scipy.ndimage import uniform_filter1d
                window_size = max(1, len(losses) // 100)
                losses = uniform_filter1d(losses, size=window_size)
            
            ax_loss.plot(iters_scaled, losses, color=color, linewidth=12.0, 
                        linestyle=linestyle, alpha=0.9)
        
        self._beautify_axis(ax_loss)
        plt.tight_layout()
        loss_path = f"{base_path}_loss.{ext}"
        plt.savefig(loss_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved loss subplot: {loss_path}")
        plt.close()
        
        # Determine common marker epochs (every 2 epochs, aiming for ~10 markers)
        max_epoch = max(max(data['val']['epochs']) if data['val']['epochs'] else 0 
                       for data in config_data.values())
        marker_step = max(1, max_epoch // 10)
        common_marker_epochs = set(range(0, max_epoch + 1, marker_step))
        
        # Save right subplot (mIoU)
        fig2 = plt.figure(figsize=(figsize[0]//2, figsize[1]))
        ax_miou = fig2.add_subplot(111)
        fig2.patch.set_facecolor('white')
        ax_miou.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]
            
            if data['val']['mIoU']:
                epochs = np.array(data['val']['epochs'])
                mious = np.array(data['val']['mIoU'])
                # Show markers only at common epoch positions
                marker_indices = [idx for idx, ep in enumerate(epochs) if ep in common_marker_epochs]
                ax_miou.plot(epochs, mious, color=color, linewidth=12.0, 
                            linestyle=linestyle, marker=marker, markersize=35, 
                            markevery=marker_indices, alpha=0.9)
        
        self._beautify_axis(ax_miou)
        ax_miou.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        plt.tight_layout()
        miou_path = f"{base_path}_miou.{ext}"
        plt.savefig(miou_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved mIoU subplot: {miou_path}")
        plt.close()
        
        # Save loss subplot WITHOUT numbers
        fig3 = plt.figure(figsize=(figsize[0]//2, figsize[1]))
        ax_loss_clean = fig3.add_subplot(111)
        fig3.patch.set_facecolor('white')
        ax_loss_clean.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            frames_per_iter = 1
            if frames_info and config_name in frames_info:
                frames_per_iter = frames_info[config_name]['frames_per_iter']
            
            iters = np.array(data['train']['total_iter'])
            losses = np.array(data['train']['Loss'])
            iters_scaled = iters * frames_per_iter
            
            if smooth and len(losses) > 10:
                from scipy.ndimage import uniform_filter1d
                window_size = max(1, len(losses) // 100)
                losses = uniform_filter1d(losses, size=window_size)
            
            ax_loss_clean.plot(iters_scaled, losses, color=color, linewidth=12.0, 
                              linestyle=linestyle, alpha=0.9)
        
        self._beautify_axis_no_numbers(ax_loss_clean)
        plt.tight_layout()
        loss_clean_path = f"{base_path}_loss_clean.{ext}"
        plt.savefig(loss_clean_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved loss subplot (no numbers): {loss_clean_path}")
        plt.close()
        
        # Save mIoU subplot WITHOUT numbers
        fig4 = plt.figure(figsize=(figsize[0]//2, figsize[1]))
        ax_miou_clean = fig4.add_subplot(111)
        fig4.patch.set_facecolor('white')
        ax_miou_clean.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]
            
            if data['val']['mIoU']:
                epochs = np.array(data['val']['epochs'])
                mious = np.array(data['val']['mIoU'])
                # Show markers only at common epoch positions
                marker_indices = [idx for idx, ep in enumerate(epochs) if ep in common_marker_epochs]
                ax_miou_clean.plot(epochs, mious, color=color, linewidth=12.0, 
                                  linestyle=linestyle, marker=marker, markersize=35,
                                  markevery=marker_indices, alpha=0.9)
        
        self._beautify_axis_no_numbers(ax_miou_clean)
        ax_miou_clean.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        plt.tight_layout()
        miou_clean_path = f"{base_path}_miou_clean.{ext}"
        plt.savefig(miou_clean_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved mIoU subplot (no numbers): {miou_clean_path}")
        plt.close()
        
        # Save loss subplot WITHOUT numbers (4:3 aspect ratio)
        fig3_43 = plt.figure(figsize=(figsize[0]//2 * 4/3, figsize[1]))
        ax_loss_clean_43 = fig3_43.add_subplot(111)
        fig3_43.patch.set_facecolor('white')
        ax_loss_clean_43.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            frames_per_iter = 1
            if frames_info and config_name in frames_info:
                frames_per_iter = frames_info[config_name]['frames_per_iter']
            
            iters = np.array(data['train']['total_iter'])
            losses = np.array(data['train']['Loss'])
            iters_scaled = iters * frames_per_iter
            
            if smooth and len(losses) > 10:
                from scipy.ndimage import uniform_filter1d
                window_size = max(1, len(losses) // 100)
                losses = uniform_filter1d(losses, size=window_size)
            
            ax_loss_clean_43.plot(iters_scaled, losses, color=color, linewidth=12.0, 
                              linestyle=linestyle, alpha=0.9)
        
        self._beautify_axis_no_numbers(ax_loss_clean_43)
        plt.tight_layout()
        loss_clean_43_path = f"{base_path}_loss_clean_4x3.{ext}"
        plt.savefig(loss_clean_43_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved loss subplot 4:3 (no numbers): {loss_clean_43_path}")
        plt.close()
        
        # Save mIoU subplot WITHOUT numbers (4:3 aspect ratio)
        fig4_43 = plt.figure(figsize=(figsize[0]//2 * 4/3, figsize[1]))
        ax_miou_clean_43 = fig4_43.add_subplot(111)
        fig4_43.patch.set_facecolor('white')
        ax_miou_clean_43.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            marker = self.markers[i % len(self.markers)]
            
            if data['val']['mIoU']:
                epochs = np.array(data['val']['epochs'])
                mious = np.array(data['val']['mIoU'])
                # Show markers only at common epoch positions
                marker_indices = [idx for idx, ep in enumerate(epochs) if ep in common_marker_epochs]
                ax_miou_clean_43.plot(epochs, mious, color=color, linewidth=12.0, 
                                  linestyle=linestyle, marker=marker, markersize=45,
                                  markevery=marker_indices, alpha=0.9)
        
        self._beautify_axis_no_numbers(ax_miou_clean_43)
        ax_miou_clean_43.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        plt.tight_layout()
        miou_clean_43_path = f"{base_path}_miou_clean_4x3.{ext}"
        plt.savefig(miou_clean_43_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved mIoU subplot 4:3 (no numbers): {miou_clean_43_path}")
        plt.close()
        
        # Save loss subplot with SUPER SMOOTH (no numbers)
        fig5 = plt.figure(figsize=(figsize[0]//2, figsize[1]))
        ax_loss_super = fig5.add_subplot(111)
        fig5.patch.set_facecolor('white')
        ax_loss_super.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            frames_per_iter = 1
            if frames_info and config_name in frames_info:
                frames_per_iter = frames_info[config_name]['frames_per_iter']
            
            iters = np.array(data['train']['total_iter'])
            losses = np.array(data['train']['Loss'])
            iters_scaled = iters * frames_per_iter
            
            # Super smooth - much larger window
            if len(losses) > 10:
                from scipy.ndimage import uniform_filter1d
                window_size = max(1, len(losses) // 20)  # 5x larger window
                losses = uniform_filter1d(losses, size=window_size)
            
            ax_loss_super.plot(iters_scaled, losses, color=color, linewidth=12.0, 
                              linestyle=linestyle, alpha=0.9)
        
        self._beautify_axis_no_numbers(ax_loss_super)
        plt.tight_layout()
        loss_super_path = f"{base_path}_loss_supersmooth.{ext}"
        plt.savefig(loss_super_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved loss subplot (super smooth): {loss_super_path}")
        plt.close()
        
        # Save mIoU subplot with SUPER SMOOTH (no numbers)
        fig6 = plt.figure(figsize=(figsize[0]//2, figsize[1]))
        ax_miou_super = fig6.add_subplot(111)
        fig6.patch.set_facecolor('white')
        ax_miou_super.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            
            if data['val']['mIoU']:
                epochs = np.array(data['val']['epochs'])
                mious = np.array(data['val']['mIoU'])
                
                # Super smooth for mIoU - use scipy interpolation
                if len(mious) > 3:
                    from scipy.interpolate import make_interp_spline
                    # Create smooth curve
                    epochs_smooth = np.linspace(epochs.min(), epochs.max(), 300)
                    spl = make_interp_spline(epochs, mious, k=3)
                    mious_smooth = spl(epochs_smooth)
                    ax_miou_super.plot(epochs_smooth, mious_smooth, color=color, linewidth=12.0, 
                                      linestyle=linestyle, alpha=0.9)
                else:
                    marker = self.markers[i % len(self.markers)]
                    # Show markers only at common epoch positions
                    marker_indices = [idx for idx, ep in enumerate(epochs) if ep in common_marker_epochs]
                    ax_miou_super.plot(epochs, mious, color=color, linewidth=12.0, 
                                      linestyle=linestyle, marker=marker, markersize=35,
                                      markevery=marker_indices, alpha=0.9)
        
        self._beautify_axis_no_numbers(ax_miou_super)
        ax_miou_super.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        plt.tight_layout()
        miou_super_path = f"{base_path}_miou_supersmooth.{ext}"
        plt.savefig(miou_super_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved mIoU subplot (super smooth): {miou_super_path}")
        plt.close()
        
        # Save loss subplot with SUPER SMOOTH (no numbers, 4:3 aspect ratio)
        fig5_43 = plt.figure(figsize=(figsize[0]//2 * 4/3, figsize[1]))
        ax_loss_super_43 = fig5_43.add_subplot(111)
        fig5_43.patch.set_facecolor('white')
        ax_loss_super_43.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            frames_per_iter = 1
            if frames_info and config_name in frames_info:
                frames_per_iter = frames_info[config_name]['frames_per_iter']
            
            iters = np.array(data['train']['total_iter'])
            losses = np.array(data['train']['Loss'])
            iters_scaled = iters * frames_per_iter
            
            # Super smooth - much larger window
            if len(losses) > 10:
                from scipy.ndimage import uniform_filter1d
                window_size = max(1, len(losses) // 20)  # 5x larger window
                losses = uniform_filter1d(losses, size=window_size)
            
            ax_loss_super_43.plot(iters_scaled, losses, color=color, linewidth=12.0, 
                              linestyle=linestyle, alpha=0.9)
        
        self._beautify_axis_no_numbers(ax_loss_super_43)
        plt.tight_layout()
        loss_super_43_path = f"{base_path}_loss_supersmooth_4x3.{ext}"
        plt.savefig(loss_super_43_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved loss subplot 4:3 (super smooth): {loss_super_43_path}")
        plt.close()
        
        # Save mIoU subplot with SUPER SMOOTH (no numbers, 4:3 aspect ratio)
        fig6_43 = plt.figure(figsize=(figsize[0]//2 * 4/3, figsize[1]))
        ax_miou_super_43 = fig6_43.add_subplot(111)
        fig6_43.patch.set_facecolor('white')
        ax_miou_super_43.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            
            if data['val']['mIoU']:
                epochs = np.array(data['val']['epochs'])
                mious = np.array(data['val']['mIoU'])
                
                # Super smooth for mIoU - use scipy interpolation
                if len(mious) > 3:
                    from scipy.interpolate import make_interp_spline
                    # Create smooth curve
                    epochs_smooth = np.linspace(epochs.min(), epochs.max(), 300)
                    spl = make_interp_spline(epochs, mious, k=3)
                    mious_smooth = spl(epochs_smooth)
                    ax_miou_super_43.plot(epochs_smooth, mious_smooth, color=color, linewidth=12.0, 
                                      linestyle=linestyle, alpha=0.9) 
                else:
                    marker = self.markers[i % len(self.markers)]
                    # Show markers only at common epoch positions
                    marker_indices = [idx for idx, ep in enumerate(epochs) if ep in common_marker_epochs]
                    ax_miou_super_43.plot(epochs, mious, color=color, linewidth=12.0,
                                      linestyle=linestyle, marker=marker, markersize=45,
                                      markevery=marker_indices, alpha=0.9)
        
        self._beautify_axis_no_numbers(ax_miou_super_43)
        ax_miou_super_43.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        plt.tight_layout()
        miou_super_43_path = f"{base_path}_miou_supersmooth_4x3.{ext}"
        plt.savefig(miou_super_43_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved mIoU subplot 4:3 (super smooth): {miou_super_43_path}")
        plt.close()
        
        # Collect Y-axis limits from supersmooth plots for consistent scaling
        y_min_all, y_max_all = float('inf'), float('-inf')
        for i, (config_name, data) in enumerate(config_data.items()):
            frames_per_iter = 1
            if frames_info and config_name in frames_info:
                frames_per_iter = frames_info[config_name]['frames_per_iter']
            
            losses_original = np.array(data['train']['Loss'])
            if len(losses_original) > 10:
                from scipy.ndimage import uniform_filter1d
                window_size = max(1, len(losses_original) // 20)
                losses_smooth = uniform_filter1d(losses_original, size=window_size)
                y_min_all = min(y_min_all, losses_smooth.min())
                y_max_all = max(y_max_all, losses_smooth.max())
        
        # Save loss subplot with SUPER SMOOTH + UNCERTAINTY (no numbers, 1:1)
        fig7 = plt.figure(figsize=(figsize[0]//2, figsize[1]))
        ax_loss_uncertainty = fig7.add_subplot(111)
        fig7.patch.set_facecolor('white')
        ax_loss_uncertainty.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            frames_per_iter = 1
            if frames_info and config_name in frames_info:
                frames_per_iter = frames_info[config_name]['frames_per_iter']
            
            iters = np.array(data['train']['total_iter'])
            losses_original = np.array(data['train']['Loss'])
            iters_scaled = iters * frames_per_iter
            
            # Calculate uncertainty bounds using local statistics
            if len(losses_original) > 10:
                from scipy.ndimage import uniform_filter1d
                
                # Super smooth for main line
                window_size_super = max(1, len(losses_original) // 20)
                losses_smooth = uniform_filter1d(losses_original, size=window_size_super)
                
                # Calculate local standard deviation for uncertainty bounds
                window_size_std = max(10, len(losses_original) // 30)
                losses_upper = []
                losses_lower = []
                
                for i in range(len(losses_original)):
                    start = max(0, i - window_size_std // 2)
                    end = min(len(losses_original), i + window_size_std // 2)
                    window_data = losses_original[start:end]
                    
                    mean_val = losses_smooth[i]
                    std_val = np.std(window_data)
                    
                    losses_upper.append(mean_val + 0.5 * std_val)
                    losses_lower.append(mean_val - 0.5 * std_val)
                
                losses_upper = np.array(losses_upper)
                losses_lower = np.array(losses_lower)
                
                # Smooth the bounds for cleaner look
                losses_upper = uniform_filter1d(losses_upper, size=window_size_super)
                losses_lower = uniform_filter1d(losses_lower, size=window_size_super)
            else:
                losses_smooth = losses_original.copy()
                losses_upper = losses_original.copy()
                losses_lower = losses_original.copy()
            
            # Plot uncertainty region (statistical bounds)
            ax_loss_uncertainty.fill_between(iters_scaled, losses_lower, losses_upper,
                                            color=color, alpha=0.35, linewidth=0)
            
            # Plot smooth line on top
            ax_loss_uncertainty.plot(iters_scaled, losses_smooth, color=color, linewidth=12.0, 
                                    linestyle=linestyle, alpha=0.9)
        
        # Set Y-axis limits to match supersmooth plots
        ax_loss_uncertainty.set_ylim(y_min_all, y_max_all)
        self._beautify_axis_no_numbers(ax_loss_uncertainty)
        plt.tight_layout()
        loss_uncertainty_path = f"{base_path}_loss_supersmooth_uncertainty.{ext}"
        plt.savefig(loss_uncertainty_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved loss subplot with uncertainty (1:1): {loss_uncertainty_path}")
        plt.close()
        
        # Save loss subplot with SUPER SMOOTH + UNCERTAINTY (no numbers, 4:3)
        fig8 = plt.figure(figsize=(figsize[0]//2 * 4/3, figsize[1]))
        ax_loss_uncertainty_43 = fig8.add_subplot(111)
        fig8.patch.set_facecolor('white')
        ax_loss_uncertainty_43.set_facecolor('#E8EDF2')
        
        for i, (config_name, data) in enumerate(config_data.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            frames_per_iter = 1
            if frames_info and config_name in frames_info:
                frames_per_iter = frames_info[config_name]['frames_per_iter']
            
            iters = np.array(data['train']['total_iter'])
            losses_original = np.array(data['train']['Loss'])
            iters_scaled = iters * frames_per_iter
            
            # Calculate uncertainty bounds using local statistics
            if len(losses_original) > 10:
                from scipy.ndimage import uniform_filter1d
                
                # Super smooth for main line
                window_size = max(1, len(losses_original) // 20)  # 5x larger window
                losses_smooth = uniform_filter1d(losses_original, size=window_size)
                
                # Calculate local standard deviation for uncertainty bounds
                window_size_std = max(10, len(losses_original) // 30)
                losses_upper = []
                losses_lower = []
                
                for i in range(len(losses_original)):
                    start = max(0, i - window_size_std // 2)
                    end = min(len(losses_original), i + window_size_std // 2)
                    window_data = losses_original[start:end]
                    
                    mean_val = losses_smooth[i]
                    std_val = np.std(window_data)
                    
                    losses_upper.append(mean_val + 0.5 * std_val)
                    losses_lower.append(mean_val - 0.5 * std_val)
                
                losses_upper = np.array(losses_upper)
                losses_lower = np.array(losses_lower)
                
                # Smooth the bounds for cleaner look
                losses_upper = uniform_filter1d(losses_upper, size=window_size)
                losses_lower = uniform_filter1d(losses_lower, size=window_size)
            else:
                losses_smooth = losses_original.copy()
                losses_upper = losses_original.copy()
                losses_lower = losses_original.copy()
            
            # Plot uncertainty region (statistical bounds)
            ax_loss_uncertainty_43.fill_between(iters_scaled, losses_lower, losses_upper,
                                               color=color, alpha=0.35, linewidth=0)
            
            # Plot smooth line on top
            ax_loss_uncertainty_43.plot(iters_scaled, losses_smooth, color=color, linewidth=12.0, 
                                       linestyle=linestyle, alpha=0.9)
        
        # Set Y-axis limits to match supersmooth plots
        ax_loss_uncertainty_43.set_ylim(y_min_all, y_max_all)
        self._beautify_axis_no_numbers(ax_loss_uncertainty_43)
        plt.tight_layout()
        loss_uncertainty_43_path = f"{base_path}_loss_supersmooth_uncertainty_4x3.{ext}"
        plt.savefig(loss_uncertainty_43_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved loss subplot with uncertainty (4:3): {loss_uncertainty_43_path}")
        plt.close()
    
    def _beautify_axis(self, ax):
        """Beautify subplot axis - clean minimal style"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax.grid(True, color='white', linewidth=10.0, linestyle='-', alpha=1.0, axis='both')
        ax.set_axisbelow(True)
        
        # Reduce number of ticks (fewer grid lines)
        # X-axis (iterations): fewer grid lines for cleaner look
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        
        ax.tick_params(axis='both', length=10, width=2.5, labelsize=32)
    
    def _beautify_axis_no_numbers(self, ax):
        """Beautify subplot axis - no tick labels, fewer grid lines"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax.grid(True, color='white', linewidth=10.0, linestyle='-', alpha=1.0, axis='both')
        ax.set_axisbelow(True)
        
        # Reduce number of ticks (half of normal)
        # X-axis (iterations): fewer grid lines for cleaner look
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        
        # Hide tick labels and ticks
        ax.tick_params(axis='both', length=0, width=0, labelsize=0)

def main():
    parser = argparse.ArgumentParser(description='Training Loss and mIoU Visualization')
    parser.add_argument('--configs', nargs='+', required=True,
                       help='Config format: name1:path1 name2:path2')
    parser.add_argument('--output', default='loss_curves.png')
    parser.add_argument('--figsize', nargs=2, type=int, default=[20, 10])
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--smooth', action='store_true', default=True)
    parser.add_argument('--frames-per-iter', nargs='+', type=int,
                       help='Frames per iteration for each config')
    
    args = parser.parse_args()
    
    config_paths = {}
    for config_str in args.configs:
        name, path = config_str.split(':', 1)
        config_paths[name] = path
    
    loss_parser = LossParser()
    config_data = {}
    frames_info = {}
    
    for idx, (config_name, log_path) in enumerate(config_paths.items()):
        data = loss_parser.parse_log_file(log_path)
        config_data[config_name] = data
        
        frames_per_iter = 1
        if args.frames_per_iter and idx < len(args.frames_per_iter):
            frames_per_iter = args.frames_per_iter[idx]
        
        frames_info[config_name] = {'frames_per_iter': frames_per_iter}
        
        miou_str = f", mIoU: {len(data['val']['mIoU'])} epochs" if data['val']['mIoU'] else ""
        print(f"{config_name}: {len(data['train']['Loss'])} loss points, final loss: {data['train']['Loss'][-1]:.4f}{miou_str}")
    
    plotter = LossCurvePlotter()
    plotter.plot_multiple_configs(
        config_data, 
        save_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        smooth=args.smooth,
        frames_info=frames_info
    )

    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()

