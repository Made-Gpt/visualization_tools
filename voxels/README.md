# Voxel Visualization Tools

Tools for 3D voxel occupancy visualization using Mayavi.

## Environment

Use the `mayavi_clean` environment:

```bash
conda activate mayavi_clean
```

**Important:** For offscreen rendering, ensure `xvfb` is installed:
```bash
sudo apt-get install xvfb
```

## Tools

### 1. Static Voxel Visualization

Render single-frame 3D voxel occupancy with customizable camera parameters. 

**Usage:**
```bash
bash vis_occ.sh
```

**Interactive Steps:**
1. Select method (Label, SplatSSC, EmbodiedOcc, Ours variants)
2. Select scene from list
3. Select frame/timestamp
4. Choose zoom mode (optional)

**Features:**
- Parallel projection rendering
- Per-scene/frame camera configuration
- Transparent background removal
- Multiple method support

**Configuration:** `camera_config.json` - Per-scene camera parameters (elevation, azimuth, scale)

---

### 2. Rotating Voxel Animation

Generate rotating 360° voxel animation with automatic GIF creation.

**Usage:**
```bash
bash vis_occ_rot.sh
```

**Quick Mode:**
Input `y` when prompted to enable quick mode - only select method/scene/frame, all other settings use defaults.

**Interactive Mode Steps:**
1. Select quick mode (y/n)
2. Select method
3. Select scene
4. Select frame
5. Configure rotation (frames, angles, resolution) - *auto in quick mode*
6. Choose window display (y/n) - *defaults to offscreen in quick mode*
7. GIF creation settings - *auto-enabled in quick mode*

**Features:**
- 36 frames (10° increments) by default
- Automatic GIF generation with white background
- Configurable frame rate (default: 10 fps)
- Optional scaling (50% for smaller files)
- Auto-generated filenames: `{scene}_{frame}_rotation.gif`
- Offscreen rendering (no window popup)

**Output Structure:**
```
method_output/
└── scene_name/
    └── frame_name_rotation/
        ├── frame_0000.png
        ├── frame_0000_t.png  (transparent bg)
        ├── frame_0001.png
        ├── ...
        └── scene_frame_rotation.gif
```

---

### 3. Global/Top-Down Voxel View

Render bird's eye view (top-down) of voxel occupancy for global scene understanding.

**Usage:**
```bash
bash vis_occ_glob.sh
```

**Interactive Steps:**
1. Select method (Label, SplatSSC, EmbodiedOcc, Ours)
2. Select scene from list
3. Select frame/timestamp

**Features:**
- Fixed top-down camera perspective
- Global scene overview
- Pre-configured camera parameters for bird's eye view
- Same transparent background removal as other tools

**Configuration:** `method_global_config.json` - Global view camera settings

---

## Configuration Files

- **method_config.json**: Method-specific paths (pcd_root, pcd_fold, output_folder)
- **method_global_config.json**: Global/top-down view camera parameters
- **camera_config.json**: Per-scene camera parameters
  ```json
  {
    "scene0000_00": {
      "pcd_00005": {
        "elevation": 65,
        "parallel_scale_factor": 0.55,
        "center_offset": [0.0, 0.0, 0.0]
      }
    }
  }
  ```

## Tips

- **Quick Mode**: Best for batch processing multiple scenes with consistent settings
- **Custom Camera**: Edit `camera_config.json` for specific viewpoints
- **GIF Size**: Enable scaling (50%) to reduce file size significantly
- **Offscreen**: Default mode avoids window popups, uses `xvfb-run` internally
