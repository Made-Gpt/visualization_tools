# Picture Generation Tools

Miscellaneous visualization utilities for figures and illustrations.

## Environment

Use the `renderpy` environment:

```bash
conda activate renderpy
```

## Tools

### 1. 3D Gaussian Noise Visualization (`3d_gaussian_noise.py`)

Generate 3D Gaussian distribution visualization.

**Dependencies:** matplotlib, numpy

---

### 2. Bar Chart Generator (`bar_fig.py`)

Create styled bar charts for paper figures.

**Dependencies:** matplotlib, seaborn, numpy, pandas

---

### 3. Color Bar Generator (`color_bar.py`)

Generate color legends for semantic categories.

**Dependencies:** matplotlib, numpy, Pillow

**Usage:**
```bash
python color_bar.py
```

---

### 4. Exponential Curve Plotter (`exp_curve.py`)

Plot exponential growth/decay curves.

**Dependencies:** matplotlib, numpy

---

### 5. Point Flow Visualization (`pointflow_fig_colorful.py`)

Visualize 3D point cloud flows with colors.

**Dependencies:** numpy, open3d

**Usage:**
```bash
python pointflow_fig_colorful.py <input_ply>
```

---

## Notes

- Scripts are standalone utilities for generating specific figures
- Customize parameters directly in each script
- Output paths are typically hardcoded - modify as needed
