#!/usr/bin/env python3
"""
3D Gaussian Splatting Renderer using Mitsuba v3 - Global Bird's Eye View
Renders PLY files containing 3D Gaussians from top-down perspective
"""
import numpy as np
import mitsuba as mi
import os
import sys
from plyfile import PlyData
import argparse
import colorsys
from tqdm import tqdm

# Set Mitsuba variant - try CUDA first, fallback to LLVM
try:
    mi.set_variant('cuda_ad_rgb')
    print("Using CUDA variant")
except:
    try: 
        mi.set_variant('llvm_ad_rgb') 
        print("Using LLVM variant")
    except:
        mi.set_variant('scalar_rgb') 
        print("Using scalar variant")
 
def enhance_colors_hsv(colors, brightness_factor=1.5, saturation_factor=1.5): 
    """Enhance colors using HSV space transformation"""
    enhanced_colors = np.zeros_like(colors)
    
    for i in range(len(colors)):
        # Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(colors[i, 0], colors[i, 1], colors[i, 2])
        
        # Enhance saturation and value (brightness)
        s = min(1.0, s * saturation_factor)  # Increase saturation by factor
        v = min(1.0, v * brightness_factor)  # Increase brightness
        
        # Convert back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        enhanced_colors[i] = [r, g, b]
    
    return enhanced_colors

class GaussianVisualizer:
    def __init__(self, ply_file_path):
        self.ply_file_path = ply_file_path
        self.xyz = None
        self.colors = None
        self.scales = None
        self.rotations = None
        self.opacity = None
    
    def load_ply_data(self):
        """Load Gaussian ellipsoid data from .ply file"""
        print(f"Loading PLY file: {self.ply_file_path}")
        plydata = PlyData.read(self.ply_file_path)
        vertex = plydata['vertex']
        
        # Extract attributes and apply coordinate transformation
        xyz = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
        self.xyz = xyz[:, [0, 2, 1]] * -1  # [-1, 1, -1] 
        # self.xyz[:, 2] += 0.10

        self.opacity = vertex['opacity']
        self.colors = np.column_stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']])
        self.scales = np.column_stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']])
        self.rotations = np.column_stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']])
        
        # Convert log scale back to original scale
        self.scales = np.exp(self.scales)
        # Convert logit opacity back to original opacity
        self.opacity = 1 / (1 + np.exp(-self.opacity)) 

        # Normalize colors to [0,1] range 
        self.colors = np.abs(self.colors) / (np.sqrt(4 * np.pi)) 
        self.colors = np.clip(self.colors, 0, 1)
        
        self.colors = enhance_colors_hsv(self.colors, brightness_factor=1.5, saturation_factor=1.5)
        # Ensure colors remain in valid range after enhancement
        self.colors = np.clip(self.colors, 0.0, 1.0)
        
        print(f"Successfully loaded {len(self.xyz)} Gaussian ellipsoids")
        print(f"Position range: X[{self.xyz[:,0].min():.2f}, {self.xyz[:,0].max():.2f}], "
              f"Y[{self.xyz[:,1].min():.2f}, {self.xyz[:,1].max():.2f}], "
              f"Z[{self.xyz[:,2].min():.2f}, {self.xyz[:,2].max():.2f}]")

def quaternion_to_matrix(q):
    """Convert quaternion (w,x,y,z) to 4x4 transformation matrix"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Create rotation matrix from quaternion
    rot_matrix = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y), 0],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x), 0],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y), 0],
        [0, 0, 0, 1]
    ])
    
    return rot_matrix

def create_ellipsoid_mesh(n_theta=12, n_phi=6):
    """Create vertices and faces for a unit ellipsoid mesh with proper normals
    
    Args:
        n_theta: Number of longitude divisions (horizontal, around Z-axis)
        n_phi: Number of latitude divisions (vertical, from north to south pole)
    
    Returns:
        vertices: Vertex positions
        normals: Vertex normals for proper lighting
        faces: Triangle face indices
    """
    vertices = [] 
    normals = []  # Add normals for proper lighting
    faces = [] 
    
    # Add top pole
    vertices.append([0, 0, 1])
    normals.append([0, 0, 1])  # Normal pointing outward
    
    # Create vertices for latitude bands
    for i in range(1, n_phi):
        phi = np.pi * i / n_phi  # from 0 to π (latitude)
        for j in range(n_theta):
            theta = 2 * np.pi * j / n_theta  # from 0 to 2π (longitude)
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta) 
            z = np.cos(phi)
            vertices.append([x, y, z])
            
            # For a unit sphere, the normal is the same as the position
            normals.append([x, y, z])
    
    # Add bottom pole
    vertices.append([0, 0, -1])
    normals.append([0, 0, -1])  # Normal pointing outward
    
    # Create faces 
    # Top cap
    for j in range(n_theta):
        next_j = (j + 1) % n_theta
        faces.append([0, j + 1, next_j + 1])
    
    # Middle bands
    for i in range(n_phi - 2):
        for j in range(n_theta): 
            next_j = (j + 1) % n_theta 
            
            # Current band vertices
            v0 = 1 + i * n_theta + j
            v1 = 1 + i * n_theta + next_j
            v2 = 1 + (i + 1) * n_theta + next_j
            v3 = 1 + (i + 1) * n_theta + j
            
            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    
    # Bottom cap
    bottom_vertex = len(vertices) - 1
    for j in range(n_theta):
        next_j = (j + 1) % n_theta
        v0 = 1 + (n_phi - 2) * n_theta + j
        v1 = 1 + (n_phi - 2) * n_theta + next_j
        faces.append([bottom_vertex, v1, v0])
    
    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(faces, dtype=np.uint32)

def create_ply_string(vertices, normals, faces):
    """Create a PLY format string for the mesh with normals"""
    ply_header = f"""ply 
        format ascii 1.0 
        element vertex {len(vertices)}
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        element face {len(faces)}
        property list uchar int vertex_indices
        end_header
    """
    
    # Add vertices with normals
    vertex_lines = []
    for i in range(len(vertices)):
        v = vertices[i]
        n = normals[i]
        vertex_lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    
    # Add faces
    face_lines = []
    for f in faces:
        face_lines.append(f"3 {f[0]} {f[1]} {f[2]}")
    
    return ply_header + '\n'.join(vertex_lines) + '\n' + '\n'.join(face_lines)



def create_mitsuba_scene(gaussians, max_gaussians=5000, camera_params=None, render_params=None, render_mode='enhanced', 
                        n_theta=24, n_phi=16, ambient_light=0.4, main_light=3.0, fill_light=2.0, top_light=1.5):
    """Create Mitsuba scene with Gaussian ellipsoids - Bird's Eye View
    
    Args:
        render_mode: 'basic', 'enhanced' (with opacity), or 'volume' (full 3D gaussian)
    """
    
    # Default camera parameters - BIRD'S EYE VIEW (top-down)
    if camera_params is None:
        bbox_min = gaussians.xyz.min(axis=0)
        bbox_max = gaussians.xyz.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2 
        bbox_size = np.linalg.norm(bbox_max - bbox_min)
         
        # Bird's eye view: camera directly above the scene
        # After coordinate transformation [0,2,1]*-1, Y-axis is vertical (negative Y is up)
        camera_params = {   
            'origin': bbox_center + np.array([0, -bbox_size * 1.5, 0]),  # Above scene (negative Y direction)
            'target': bbox_center,   
            'up': [0.0, 0.0, -1.0],  # Z-axis points "up" in screen space
            'fov': 45.0  
        }
        
        print(f"[INFO] Bird's Eye View Camera:")
        print(f"  Origin: {camera_params['origin']}")
        print(f"  Target: {camera_params['target']}")
        print(f"  Looking: Top-down (negative Y-axis)")
    
    # Default render parameters
    if render_params is None:
        render_params = {
            'width': 1024,
            'height': 1024,
            'spp': 128
        }
    
    # Create scene dictionary
    scene_dict = { 
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8, 
            'hide_emitters': True  # add this 
        }, 
        'sensor': {
            'type': 'perspective',
            'fov': camera_params['fov'],
            'to_world': mi.ScalarTransform4f().look_at(
                origin=camera_params['origin'], 
                target=camera_params['target'], 
                up=camera_params['up']
            ),
            'film': {
                'type': 'hdrfilm',
                'width': render_params['width'],
                'height': render_params['height'], 
                'pixel_format': 'rgba', 
                'component_format': 'float32', 
            }, 
            'sampler': {
                'type': 'independent',
                'sample_count': render_params['spp']
            }
        }
    }
     
    # Add environment lighting - configurable for academic quality
    scene_dict['env'] = {
        'type': 'constant',
        'radiance': {
            'type': 'rgb',
            # 'value': [0, 0, 0], 
            'value': [ambient_light, ambient_light, ambient_light]
        }  
    }   
    
    # Add multiple directional lights for better ellipsoid illumination
    # For bird's eye view, lighting should come from various angles
    scene_dict['light1'] = {
        'type': 'directional',
        'direction': [1, 1, -1],
        'irradiance': { 
            'type': 'rgb',
            'value': [main_light, main_light, main_light]
        }
    }
    
    scene_dict['light2'] = {
        'type': 'directional', 
        'direction': [-1, -1, -1],
        'irradiance': {
            'type': 'rgb',
            'value': [fill_light, fill_light, fill_light]
        }
    }
    
    # Add a third light from above for better shape definition
    scene_dict['light3'] = {
        'type': 'directional',
        'direction': [0, 0, -1],
        'irradiance': {
            'type': 'rgb',
            'value': [top_light, top_light, top_light]
        }
    }
    
    # Sort gaussians by opacity (highest first) and limit count
    opacity_indices = np.argsort(-gaussians.opacity)
    num_gaussians = min(max_gaussians, len(gaussians.xyz))
    selected_indices = opacity_indices[:num_gaussians]
    
    print(f"Rendering {num_gaussians} gaussians (sorted by opacity)") 
    
    # Add Gaussian ellipsoids to scene
    for i, idx in enumerate(tqdm(selected_indices, desc="Creating ellipsoids")):
        pos = gaussians.xyz[idx]
        scale = gaussians.scales[idx] 
        rot = gaussians.rotations[idx]
        opacity = gaussians.opacity[idx]
        color = np.clip(gaussians.colors[idx], 0.0, 1.0)
        
        # Skip very transparent or very small gaussians
        if opacity < 0.05 or np.max(scale) < 0.005:
            continue
        
        # Create transformation matrices using individual transforms
        pos_list = [float(x) for x in pos]
        scale_list = [float(x) for x in scale]
        
        # Create basic transforms and combine them 
        translation = mi.ScalarTransform4f().translate(v=pos_list)
        
        # Convert quaternion to rotation matrix
        rot_matrix = quaternion_to_matrix(rot)
        rotation = mi.ScalarTransform4f([[float(x) for x in row] for row in rot_matrix])
        
        # Create scaling
        scaling = mi.ScalarTransform4f().scale(v=scale_list)
        
        # Combine transformations: T * R * S
        transform = translation @ rotation @ scaling
    
        # Enhanced material with opacity via transparency
        base_color = [float(np.clip(color[0], 0.0, 1.0)), 
                     float(np.clip(color[1], 0.0, 1.0)), 
                     float(np.clip(color[2], 0.0, 1.0))]
        if opacity < 0.95:
            # Use mask material for transparency
            material = {
                'type': 'mask',
                'opacity': {
                    'type': 'rgb',
                    'value': [float(opacity)] * 3
                },
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': base_color
                    }
                }
            }
        else:
            material = {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': base_color
                }
            }
            
        # Create gaussian as true ellipsoid mesh
        vertices, normals, faces = create_ellipsoid_mesh(n_theta=n_theta, n_phi=n_phi)
        
        # Apply scaling, rotation, and translation manually
        scaled_vertices = vertices * scale[np.newaxis, :]
    
        # Transform normals properly for ellipsoids
        # For ellipsoids, normals need inverse transpose transformation
        scale_matrix = np.diag(1.0 / scale)  # Inverse scaling for normals
        transformed_normals = (scale_matrix @ normals.T).T 
        # Normalize the transformed normals
        norm_lengths = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
        transformed_normals = transformed_normals / (norm_lengths + 1e-8)
        
        # Get rotation matrix from quaternion
        rot_matrix_4x4 = quaternion_to_matrix(rot)
        rot_matrix_3x3 = rot_matrix_4x4[:3, :3]
        
        # Apply rotation to vertices and normals
        rotated_vertices = (rot_matrix_3x3 @ scaled_vertices.T).T
        rotated_normals = (rot_matrix_3x3 @ transformed_normals.T).T
        
        # Apply translation (only affects vertices, not normals)
        final_vertices = rotated_vertices + pos[np.newaxis, :]
        final_normals = rotated_normals
        
        # Create temporary PLY file for the ellipsoid
        import tempfile
        temp_ply_file = tempfile.NamedTemporaryFile(mode='w', suffix='.ply', delete=False)
        ply_content = create_ply_string(final_vertices, final_normals, faces)
        temp_ply_file.write(ply_content)
        temp_ply_file.flush()
        temp_ply_file.close()
        
        shape = {
            'type': 'ply',
            'filename': temp_ply_file.name,  
            'bsdf': material  
        }
        
        scene_dict[f'gaussian_{i}'] = shape
    
    return mi.load_dict(scene_dict)

def render_gaussian_scene(ply_path, output_path, render_params=None, camera_params=None, max_gaussians=5000, render_mode='enhanced',
                         n_theta=24, n_phi=16, ambient_light=0.4, main_light=3.0, fill_light=2.0, top_light=1.5):
    """Render a single Gaussian PLY file with bird's eye view"""
    
    print(f"Processing: {os.path.basename(ply_path)}") 
    
    # Load Gaussian data
    gaussians = GaussianVisualizer(ply_path)
    gaussians.load_ply_data()
    
    # Create and render scene
    print("Creating Mitsuba scene (Bird's Eye View)...")
    scene = create_mitsuba_scene(gaussians, max_gaussians, camera_params, render_params, render_mode,
                                n_theta, n_phi, ambient_light, main_light, fill_light, top_light)
    
    print("Rendering...")
    image = mi.render(scene, spp=render_params['spp'] if render_params else 128) 

    # Save the output in multiple formats 
    # Save as EXR (high dynamic range) 
    mi.util.write_bitmap(output_path, image)  
    print(f"Saved EXR: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Render 3D Gaussians with Mitsuba v3 - Bird\'s Eye View')
    
    # Two modes: batch processing or single file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ply_root', type=str, help='Root directory containing PLY files (batch mode)')
    group.add_argument('--input_file', type=str, help='Single PLY file to render')
    
    # Batch mode parameters
    parser.add_argument('--ply_fold', type=str, help='Folder name containing Gaussian PLY files (batch mode)')
    parser.add_argument('--output_folder', type=str, help='Output folder for rendered images (batch mode)')
    parser.add_argument('--ply_ext', type=str, default='.ply', help='PLY file extension (batch mode)')
    
    # Single file mode parameter
    parser.add_argument('--output_file', type=str, help='Output file path (single file mode)')
    
    # Common parameters
    parser.add_argument('--width', type=int, default=1024, help='Render width')
    parser.add_argument('--height', type=int, default=1024, help='Render height')
    parser.add_argument('--spp', type=int, default=128, help='Samples per pixel')
    parser.add_argument('--max_gaussians', type=int, default=5000, help='Maximum number of gaussians to render')
    parser.add_argument('--camera_distance', type=float, default=None, help='Camera distance from scene center')
    parser.add_argument('--render_mode', type=str, default='enhanced', choices=['basic', 'enhanced', 'volume'], 
                       help='Rendering mode: basic (fast), enhanced (opacity), volume (full 3D gaussian)')
    
    # Ellipsoid mesh quality parameters
    parser.add_argument('--n_theta', type=int, default=24, help='Longitude divisions for ellipsoid mesh (default: 24)')
    parser.add_argument('--n_phi', type=int, default=16, help='Latitude divisions for ellipsoid mesh (default: 16)')
    
    # Lighting parameters 
    parser.add_argument('--ambient_light', type=float, default=0.4, help='Ambient lighting strength (default: 0.4)')
    parser.add_argument('--main_light', type=float, default=3.0, help='Main directional light strength (default: 3.0)')
    parser.add_argument('--fill_light', type=float, default=2.0, help='Fill light strength (default: 2.0)')
    parser.add_argument('--top_light', type=float, default=1.5, help='Top light strength (default: 1.5)')
    
    args = parser.parse_args()
    
    # Render parameters
    render_params = {
        'width': args.width,
        'height': args.height,
        'spp': args.spp
    }
    
    # Single file mode
    if args.input_file:
        if not args.output_file:
            parser.error("--output_file is required when using --input_file")
        
        if not os.path.exists(args.input_file):
            print(f"Error: Input file {args.input_file} does not exist!")
            return
        
        # Create output directory if needed
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Rendering single file: {args.input_file}")
        try:
            render_gaussian_scene(
                args.input_file,
                args.output_file,
                render_params=render_params,
                max_gaussians=args.max_gaussians,
                render_mode=args.render_mode,
                n_theta=args.n_theta,
                n_phi=args.n_phi,
                ambient_light=args.ambient_light,
                main_light=args.main_light,
                fill_light=args.fill_light,
                top_light=args.top_light
            )
            print(f"✅ Rendering complete: {args.output_file}")
        except Exception as e:
            print(f"❌ Error rendering {args.input_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    
    # Batch mode
    else:
        if not args.ply_fold or not args.output_folder:
            parser.error("--ply_fold and --output_folder are required for batch mode")
        
        batch_process(args, render_params)

def batch_process(args, render_params):
    """Process multiple PLY files in batch mode"""
    # Setup paths
    input_folder = os.path.join(args.ply_root, args.ply_fold)
    output_folder = os.path.join(args.ply_root, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all PLY files
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist!")
        return
        
    ply_files = [f for f in os.listdir(input_folder) if f.endswith(args.ply_ext)]
    ply_files.sort()
    
    if not ply_files:
        print(f"No PLY files found in {input_folder}")
        return
    
    print(f"Found {len(ply_files)} PLY files in {input_folder}")
    
    # Process each PLY file
    for ply_file in ply_files:
        ply_path = os.path.join(input_folder, ply_file)
        base_name = os.path.splitext(ply_file)[0] 
        output_path = os.path.join(output_folder, f"{base_name}_mitsuba_glob.exr")
        
        if os.path.exists(output_path):
            print(f"Skipping {ply_file} (output already exists)")
            continue
        
        try:
            render_gaussian_scene(
                ply_path, 
                output_path, 
                render_params=render_params,
                max_gaussians=args.max_gaussians,
                render_mode=args.render_mode,
                n_theta=args.n_theta,
                n_phi=args.n_phi,
                ambient_light=args.ambient_light,
                main_light=args.main_light,
                fill_light=args.fill_light,
                top_light=args.top_light
            )
        except Exception as e:
            print(f"Error processing {ply_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nBatch rendering complete! Images saved to: {output_folder}")

if __name__ == "__main__":
    main()
