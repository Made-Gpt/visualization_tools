import sys
import numpy as np
import open3d as o3d 
from pathlib import Path

def load_file(filename):
    if 'npy' in filename:
        pcl = np.load(filename)
        color = None 
    elif 'ply' in filename:
        pcd = o3d.io.read_point_cloud(filename)
        pcl = np.array(pcd.points)
        colors = np.array(pcd.colors)
    print(f"Loaded {filename} with shape {pcl.shape}")
    return pcl, colors


def standardize_bbox(pcl, points_per_object):
    if pcl.shape[0] < points_per_object:
        print(f"Warning: point cloud only has {pcl.shape[0]} points, less than {points_per_object}, using all points with replacement.")
        indices = np.random.choice(pcl.shape[0], points_per_object, replace=True)
    else:
        indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(indices)
    pcl_sampled = pcl[indices] # n by 3 
    mins = np.amin(pcl_sampled, axis=0) 
    maxs = np.amax(pcl_sampled, axis=0) 
    center = (mins + maxs) / 2. 
    scale = np.amax(maxs - mins) 
    print("Center: {}, Scale: {}".format(center, scale)) 
    result = ((pcl_sampled - center) / scale).astype(np.float32)  # [-0.5, 0.5] 
    return result, indices 
 
# "1.20, 3.60, 1.50"
# "1.08, 3.24, 1.35" 
# "1.00, 2.80, 1.20" 
# "0.80, 2.40, 1.00" 
# "0.80, 3.00, 0.20"

xml_head = \
""" 
<scene version="0.6.0"> 
    <integrator type="path"> 
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>  
        <float name="nearClip" value="0.1"/>  
        <transform name="toWorld">
            <lookat origin="0.00, 2.80, 0.45" target="0, 0, 0.3" up="0, 0, 1"/>
        </transform> 
        <float name="fov" value="25"/> 
 
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/> 
            <boolean name="banner" value="false"/>
        </film>
    </sensor>

    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
""" 

# <float name="radius" value="0.016"/>
# <float name="radius" value="0.012"/>  

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.012"/>  
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/> 
        </bsdf>
    </shape>
"""


xml_tail = \
"""
    <shape type="rectangle"> 
        <ref name="bsdf" id="surfaceMaterial"/> 
        <transform name="toWorld"> 
            <scale x="10" y="10" z="1"/> 
            <translate x="0" y="0" z="-0.5"/> 
        </transform> 
    </shape> 
    
    <emitter type="constant"> 
        <rgb name="radiance" value="1.2, 1.2, 1.2"/>  
    </emitter>

</scene>
"""

# """
#     <shape type="rectangle">
#         <ref name="bsdf" id="surfaceMaterial"/>
#         <transform name="toWorld">
#             <scale x="10" y="10" z="1"/>
#             <translate x="0" y="0" z="-0.5"/>
#         </transform>
#     </shape>
#     
#     <shape type="rectangle">
#         <transform name="toWorld">
#             <scale x="10" y="10" z="1"/>
#             <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
#         </transform>
#         <emitter type="area">
#             <rgb name="radiance" value="6,6,6"/>
#         </emitter>
#     </shape>
# </scene>
# """

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]
xml_segments = [xml_head]


if len(sys.argv) > 4:
    root = Path(sys.argv[1])
    fold = sys.argv[2]
    scene = sys.argv[3]
    name = sys.argv[4]
else:
    root = Path('path/to/your/folder')  # absolute path
    fold = 'your_data_foler'  
    scene = 'scene0000_00'  # FIXME
    name = 'pcd_00005'  # FIXME 
 
ext = '.ply'
name_ext = name + ext 
file = str(root / fold / scene / name_ext)

pcl, colors = load_file(file) 
pcl, indices = standardize_bbox(pcl, 15360)  # 1024, 15360  # FIXME
 
# convert to Mitsuba coordinate 
# pcl = pcl[:,[2,0,1]]  
# pcl[:,0] *= -1  
pcl = pcl[:,[0, 2, 1]]  
pcl *= -1  
pcl[:,2] += 0.35  

for i in range(pcl.shape[0]):  
    if colors.shape[0] > 1:  
        colors_ = colors[indices]
        color = colors_[i] 
    else: 
        color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
    xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
xml_segments.append(xml_tail)

xml_content = str.join('', xml_segments)

with open('mitsuba_scene.xml', 'w') as f:
    f.write(xml_content)


