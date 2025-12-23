# Gaussian Splatting Visualization

Tools for rendering 3D Gaussian Splatting results.

## Environments

Two rendering backends are supported:

### 1. Mitsuba Renderer (Recommended)

High-quality physically-based rendering.

**Setup:**
```bash
conda create -n renderpy python=3.9
conda activate renderpy
pip install matplotlib numpy Pillow seaborn pandas open3d plyfile scipy tqdm mitsuba
```

**Scripts:**
- `vis_gs.py` - Single scene Gaussian rendering
- `vis_gs.sh` - Interactive wrapper for single scene rendering
- `vis_gs_glob.sh` - Batch rendering for multiple scenes (bird's eye view)

**Usage (Single Scene):**
```bash
bash vis_gs.sh
```

**Usage (Batch/Global View):**
```bash
bash vis_gs_glob.sh
```

**Features:**
- ✅ Physically-based rendering with Mitsuba v3
- ✅ Bird's eye view for global scene understanding
- ✅ Batch processing (render all scenes or selected scenes)
- ✅ Configurable quality settings (SPP, mesh quality)
- ✅ Academic lighting setup (ambient, directional, fill, top lights)

**Configuration Parameters (vis_gs_glob.sh):**
- `WIDTH=1200`, `HEIGHT=900` - Output resolution
- `SPP=128` - Samples per pixel (quality)
- `MAX_GAUSSIANS=15000` - Max Gaussians per scene
- `N_THETA=24`, `N_PHI=16` - Ellipsoid mesh quality
- Lighting: `AMBIENT_LIGHT=0.1`, `MAIN_LIGHT=3.0`, `FILL_LIGHT=2.5`, `TOP_LIGHT=1.0`

---

### 2. Matplotlib Renderer (Lightweight)

Fast 2D/3D visualization without GPU requirements.

**Setup:**
Use the `renderpy` environment (same as above).

**Script:** `matplotlib/vis_gs.py`, `matplotlib/vis_gs.sh`

**Usage:**
```bash
cd matplotlib
bash vis_gs.sh
```

---

## HTML Viewers

Pre-built HTML templates for interactive Gaussian visualization:

- **2d_gs.html** - 2D Gaussian visualization viewer
- **3d_gs.html** - 3D interactive Gaussian viewer

Open in a web browser to use.

---

## Features

### Mitsuba Backend
- ✅ Physically-based rendering
- ✅ High-quality output
- ✅ Batch processing support (vis_gs_glob.sh)
- ✅ Bird's eye view for global scene analysis
- ✅ Configurable quality and lighting parameters
- ✅ Multiple camera angles support

### Matplotlib Backend
- ✅ No GPU required
- ✅ Fast rendering
- ✅ 2D projection support
- ✅ Ellipse visualization
- ✅ Interactive shell script (vis_gs.sh)

---

## Configuration

### Mitsuba Rendering

**vis_gs_glob.sh parameters:**
Edit the script to adjust:
- `PLY_ROOT`, `PLY_FOLD`, `OUTPUT_FOLDER` - Input/output paths
- `WIDTH`, `HEIGHT` - Output resolution
- `SPP` - Samples per pixel (quality vs speed)
- `MAX_GAUSSIANS` - Maximum Gaussians to render
- `N_THETA`, `N_PHI` - Ellipsoid mesh quality
- `AMBIENT_LIGHT`, `MAIN_LIGHT`, `FILL_LIGHT`, `TOP_LIGHT` - Lighting setup

**Camera Configuration:**
- Global view: Fixed bird's eye view camera position
- Custom views: Edit camera parameters in Python script

### Matplotlib Rendering

Edit rendering parameters directly in `matplotlib/vis_gs.py`:
- Camera positions
- Resolution
- Output formats
- Gaussian appearance (scale, opacity)

## Notes

- Mitsuba backend requires GPU for best performance
- Matplotlib backend is CPU-only, suitable for quick previews
- Input format: PLY files with Gaussian parameters
