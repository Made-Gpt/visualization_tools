import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
import json
import os


model_dict = {
    "blue_envelope": "geo", 
    "curve_envelope": "sem", 
}


def plot_multiple_curves(curves_config, save_mode="combined", save_path="./curves", figure_size=(8, 6), envelope_mode="none", envelope_config=None):
    """
    Plots multiple Gaussian curves based on configuration. 
    
    Parameters:
    - curves_config: List of dictionaries, each containing curve parameters
    - save_mode: "combined" for all curves in one plot, "separate" for individual plots
    - save_path: Base path for saving plots
    - figure_size: Tuple for figure size (width, height)
    - envelope_mode: "none", "blue_envelope", or "curve_envelope" for envelope drawing
    - envelope_config: Dictionary containing envelope configuration (color, line_width)
    """
    
    # Default envelope configuration
    if envelope_config is None:
        envelope_config = {
            'color': 'darkblue',
            'line_width': 12,
            'alpha': 0.8
        }
    
    if save_mode == "combined":
        # Plot all curves in one figure
        plt.figure(figsize=figure_size)
        
        # Store all curve data for envelope calculation
        all_curves_data = [] 
        
        for i, curve in enumerate(curves_config):
            x_min = curve.get('x_min', -3)
            x_max = curve.get('x_max', 3)
            mean = curve.get('mean', 0)
            std_dev = curve.get('std_dev', 1)
            height = curve.get('height', 1.0)  # Height scaling factor
            width = curve.get('width', 1.0)   # Width scaling factor
            color = curve.get('color', 'darkred')
            label = curve.get('label', f'Curve {i+1}')
            fill_alpha = curve.get('fill_alpha', 0.2)
            line_alpha = curve.get('line_alpha', 1.0)  # Line transparency
            line_width = curve.get('line_width', 2)
            
            # Generate x values and calculate PDF
            x = np.linspace(x_min, x_max, 1000)
            # Apply width scaling to std_dev and height scaling to PDF
            effective_std_dev = std_dev * width
            y = norm.pdf(x, mean, effective_std_dev) * height
            
            # Store curve data for envelope calculation
            all_curves_data.append({
                'x': x,
                'y': y,
                'color': color,
                'x_min': x_min,
                'x_max': x_max
            })
             
            # Plot curve
            plt.plot(x, y, color=color, linewidth=line_width, alpha=line_alpha)
            if fill_alpha > 0:
                plt.fill_between(x, y, color=color, alpha=fill_alpha)
        
        # Draw envelope if requested
        if envelope_mode != "none":
            draw_envelope(all_curves_data, envelope_mode, envelope_config)
        
        plt.axis('off')  # Hide all axes and labels
         
        # Save combined plot
        model_sign = model_dict[envelope_mode]
        save_file = f"{save_path}_combined_{model_sign}.png"
        plt.savefig(save_file, bbox_inches='tight', transparent=True, pad_inches=0, dpi=300)
        plt.close()
        print(f"Combined plot saved to {save_file}")
        
    elif save_mode == "separate":
        # Plot each curve separately
        for i, curve in enumerate(curves_config):
            plt.figure(figsize=figure_size)
            plt.gca().set_facecolor('white')  # Set background to white
            
            x_min = curve.get('x_min', -3)
            x_max = curve.get('x_max', 3)
            mean = curve.get('mean', 0)
            std_dev = curve.get('std_dev', 1)
            height = curve.get('height', 1.0)  # Height scaling factor
            width = curve.get('width', 1.0)   # Width scaling factor
            color = curve.get('color', 'darkred')
            label = curve.get('label', f'Curve {i+1}')
            fill_alpha = curve.get('fill_alpha', 0.2)
            line_alpha = curve.get('line_alpha', 1.0)  # Line transparency
            line_width = curve.get('line_width', 2)
            
            # Generate x values and calculate PDF
            x = np.linspace(x_min, x_max, 1000)
            # Apply width scaling to std_dev and height scaling to PDF
            effective_std_dev = std_dev * width
            y = norm.pdf(x, mean, effective_std_dev) * height
            
            # Plot curve
            plt.plot(x, y, color=color, linewidth=line_width, alpha=line_alpha)
            if fill_alpha > 0:
                plt.fill_between(x, y, color=color, alpha=fill_alpha)
            
            plt.axis('off')  # Hide all axes and labels
            
            # Save individual plot
            save_file = f"{save_path}_curve_{i+1}.png"
            plt.savefig(save_file, bbox_inches='tight', transparent=True, pad_inches=0, dpi=300)
            plt.close()
            print(f"Curve {i+1} saved to {save_file}")
    
    elif save_mode == "both":
        # Plot all curves in one figure first
        plt.figure(figsize=figure_size)
        
        # Store all curve data for envelope calculation
        all_curves_data = []
        
        for i, curve in enumerate(curves_config):
            x_min = curve.get('x_min', -3)
            x_max = curve.get('x_max', 3)
            mean = curve.get('mean', 0)
            std_dev = curve.get('std_dev', 1)
            height = curve.get('height', 1.0)  # Height scaling factor
            width = curve.get('width', 1.0)   # Width scaling factor
            color = curve.get('color', 'darkred')
            label = curve.get('label', f'Curve {i+1}')
            fill_alpha = curve.get('fill_alpha', 0.2)
            line_alpha = curve.get('line_alpha', 1.0)  # Line transparency
            line_width = curve.get('line_width', 2)
            
            # Generate x values and calculate PDF
            x = np.linspace(x_min, x_max, 1000)
            # Apply width scaling to std_dev and height scaling to PDF
            effective_std_dev = std_dev * width
            y = norm.pdf(x, mean, effective_std_dev) * height
            
            # Store curve data for envelope calculation
            all_curves_data.append({
                'x': x,
                'y': y,
                'color': color,
                'x_min': x_min,
                'x_max': x_max
            })
            
            # Plot curve
            plt.plot(x, y, color=color, linewidth=line_width, alpha=line_alpha)
            if fill_alpha > 0:
                plt.fill_between(x, y, color=color, alpha=fill_alpha)
        
        # Draw envelope if requested
        if envelope_mode != "none":
            draw_envelope(all_curves_data, envelope_mode, envelope_config)
        
        plt.axis('off')  # Hide all axes and labels
        
        # Save combined plot
        model_sign = model_dict[envelope_mode] 
        save_file = f"{save_path}_combined_{model_sign}.png"
        plt.savefig(save_file, bbox_inches='tight', transparent=True, pad_inches=0, dpi=300)
        plt.close()
        print(f"Combined plot saved to {save_file}")
        
        # Then plot each curve separately
        for i, curve in enumerate(curves_config):
            plt.figure(figsize=figure_size)
            plt.gca().set_facecolor('white')  # Set background to white
            
            x_min = curve.get('x_min', -3)
            x_max = curve.get('x_max', 3)
            mean = curve.get('mean', 0)
            std_dev = curve.get('std_dev', 1)
            height = curve.get('height', 1.0)  # Height scaling factor
            width = curve.get('width', 1.0)   # Width scaling factor
            color = curve.get('color', 'darkred')
            label = curve.get('label', f'Curve {i+1}')
            fill_alpha = curve.get('fill_alpha', 0.2)
            line_alpha = curve.get('line_alpha', 1.0)  # Line transparency
            line_width = curve.get('line_width', 2)
            
            # Generate x values and calculate PDF
            x = np.linspace(x_min, x_max, 1000)
            # Apply width scaling to std_dev and height scaling to PDF
            effective_std_dev = std_dev * width
            y = norm.pdf(x, mean, effective_std_dev) * height
            
            # Plot curve
            plt.plot(x, y, color=color, linewidth=line_width, alpha=line_alpha)
            if fill_alpha > 0:
                plt.fill_between(x, y, color=color, alpha=fill_alpha)
            
            plt.axis('off')  # Hide all axes and labels
            
            # Save individual plot
            save_file = f"{save_path}_curve_{i+1}.png"
            plt.savefig(save_file, bbox_inches='tight', transparent=True, pad_inches=0, dpi=500)
            plt.close()
            print(f"Curve {i+1} saved to {save_file}")

def draw_envelope(all_curves_data, envelope_mode, envelope_config):
    """
    Draw envelope based on maximum values across curves.
    
    Parameters:
    - all_curves_data: List of curve data dictionaries
    - envelope_mode: "blue_envelope" or "curve_envelope"
    - envelope_config: Dictionary containing envelope configuration
    """
    if not all_curves_data:
        return
    
    # Debug: print envelope configuration
    print(f"Drawing envelope with mode: {envelope_mode}")
    print(f"Envelope config: {envelope_config}")
    
    # Find the common x range
    x_min_global = min(curve['x_min'] for curve in all_curves_data)
    x_max_global = max(curve['x_max'] for curve in all_curves_data)
    
    # Create a common x grid for comparison
    x_common = np.linspace(x_min_global, x_max_global, 2000)
    
    # Interpolate all curves to the common x grid
    interpolated_curves = []
    for curve in all_curves_data:
        y_interp = np.interp(x_common, curve['x'], curve['y'])
        interpolated_curves.append({
            'y': y_interp,
            'color': curve['color']
        })
    
    # Find maximum values and corresponding curve indices
    max_values = []
    max_curve_indices = []
    
    for i in range(len(x_common)):
        values_at_x = [curve['y'][i] for curve in interpolated_curves]
        max_value = max(values_at_x)
        max_index = values_at_x.index(max_value)
        max_values.append(max_value)
        max_curve_indices.append(max_index)
    
    if envelope_mode == "blue_envelope":
        # Draw envelope in specified color with specified line width
        plt.plot(x_common, max_values, 
                color=envelope_config.get('color', 'darkblue'), 
                linewidth=envelope_config.get('line_width', 12), 
                alpha=envelope_config.get('alpha', 0.8))
        
    elif envelope_mode == "curve_envelope":
        # Draw envelope with colors changing based on which curve is maximum
        # Group consecutive points with the same maximum curve
        current_curve_idx = max_curve_indices[0]
        start_idx = 0 
        
        print(f"Drawing curve_envelope with line_width: {envelope_config.get('line_width', 12)}")
        
        for i in range(1, len(max_curve_indices) + 1):
            if i == len(max_curve_indices) or max_curve_indices[i] != current_curve_idx:
                # Draw segment
                end_idx = i - 1
                segment_x = x_common[start_idx:end_idx + 1]
                segment_y = max_values[start_idx:end_idx + 1]
                curve_color = all_curves_data[current_curve_idx]['color']

                # Make sure we're using the envelope_config line_width, not the original curve line_width
                line_width = envelope_config.get('line_width', 12) * 1.25
                alpha = envelope_config.get('alpha', 0.9)
                
                print(f"Plotting segment with color: {curve_color}, line_width: {line_width}, alpha: {alpha}")
                
                plt.plot(segment_x, segment_y, 
                        color=curve_color, 
                        linewidth=line_width, 
                        alpha=alpha)
                
                # Update for next segment 
                if i < len(max_curve_indices):
                    current_curve_idx = max_curve_indices[i]
                    start_idx = i

def main():
    parser = argparse.ArgumentParser(description='Plot Gaussian curves with flexible configuration')
    parser.add_argument('--config', type=str, required=True, help='JSON config file path')
    parser.add_argument('--save_mode', type=str, default='combined', choices=['combined', 'separate', 'both'], 
                       help='Save mode: combined, separate, or both')
    parser.add_argument('--save_path', type=str, default='./curves', help='Base save path')
    parser.add_argument('--figure_size', type=str, default='8,6', help='Figure size as width,height')
    parser.add_argument('--envelope_mode', type=str, default='none', choices=['none', 'blue_envelope', 'curve_envelope'],
                       help='Envelope mode: none, blue_envelope, or curve_envelope')
    parser.add_argument('--envelope_color', type=str, default='darkblue', help='Envelope color for blue_envelope mode')
    parser.add_argument('--envelope_width', type=float, default=12, help='Envelope line width')
    parser.add_argument('--envelope_alpha', type=float, default=0.8, help='Envelope transparency')
    
    args = parser.parse_args()
    
    # Parse figure size
    figure_size = tuple(map(float, args.figure_size.split(',')))
    
    # Create envelope configuration
    envelope_config = {
        'color': args.envelope_color,
        'line_width': args.envelope_width,
        'alpha': args.envelope_alpha
    }
    
    # Load configuration
    with open(args.config, 'r') as f:
        curves_config = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)
    
    # Plot curves
    plot_multiple_curves(curves_config, args.save_mode, args.save_path, figure_size, args.envelope_mode, envelope_config)

if __name__ == "__main__":
    main()