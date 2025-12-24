# Tools for Visualization

This repository contains visualization tools for 3D semantic scene completion and training analysis.

## 🦉 Environment Setup

### Base Environment (renderpy)

Create a base environment for general visualization tasks:

```bash
# Create environment
conda create -n renderpy python=3.9
conda activate renderpy

# Install required packages
pip install matplotlib numpy Pillow seaborn pandas open3d plyfile scipy tqdm mitsuba
```

### Mayavi Environment (mayavi_clean)

Create a dedicated environment for 3D voxel visualization with Mayavi:

```bash
# Create clean environment
conda create -n mayavi_clean python=3.9 -c conda-forge
conda activate mayavi_clean

# Install Mayavi and dependencies
conda install -c conda-forge mayavi
conda install -c conda-forge open3d

# Install additional packages
pip install Pillow
```

---

## 🐦 Testing Your Installation

### Test Mayavi Environment

Validate your Mayavi installation with these tests:

**(a) Test Qt Backend**

```python
from PyQt5.QtWidgets import QApplication, QLabel
import sys
app = QApplication(sys.argv)
label = QLabel("Hello from Qt")
label.show()
app.exec_()
```

**(b) Test TraitsUI**

```python
import os
os.environ["ETS_TOOLKIT"] = "qt"
os.environ["QT_API"] = "pyqt5"

from traits.api import HasTraits, Str
from traitsui.api import View, Item

class Person(HasTraits):
    name = Str() 
    traits_view = View(Item('name')) 
 
p = Person(name="ChatGPT") 
p.configure_traits() 
```

**(c) Test Mayavi**

```python
import os
os.environ["ETS_TOOLKIT"] = "qt"
os.environ["QT_API"] = "pyqt5"

from mayavi import mlab 
mlab.test_plot3d() 
mlab.show()
```

---

## 🦚 Usage

### Training & Efficiency Analysis ([experiment/](experiment/README.md))

Visualize training metrics, efficiency comparisons, and latency analysis.

```bash
cd experiment
conda activate renderpy
bash vis_train.sh      # Training loss & mIoU curves
bash vis_effect.sh     # Efficiency metrics (anchors, features, memory)
bash vis_latency.sh    # Latency and parameter scatter plots
```

**[→ See detailed documentation](experiment/README.md)**

---

### 3D Voxel Visualization ([voxels/](voxels/README.md))

Render static and rotating 3D voxel occupancy with Mayavi.

```bash
cd voxels
conda activate mayavi_clean
bash vis_occ.sh        # Static visualization
bash vis_occ_rot.sh    # Rotating animation + GIF
```

**Quick mode:** Auto-generate with defaults - only select method/scene/frame.

**[→ See detailed documentation](voxels/README.md)**

---

### Gaussian Splatting Rendering ([gaussian/](gaussian/README.md))

High-quality Gaussian splatting visualization with Mitsuba or matplotlib.

**Mitsuba (recommended):**
```bash 
conda activate renderpy
bash vis_gs.sh
bash vis_gs_glob.sh
``` 

**Matplotlib (lightweight):**
```bash
cd gaussian/matplotlib
conda activate renderpy
bash vis_gs.sh 
```

**[→ See detailed documentation](gaussian/README.md)**

---

### Miscellaneous Figures ([pictures/](pictures/README.md))

Utilities for generating paper figures, color bars, and point flow visualizations.

```bash
cd pictures
conda activate renderpy
bash <script_name>.sh 
```

**[→ See detailed documentation](pictures/README.md)**

---

## 📝 Important Notes

> 💡 **When using Mayavi**, you **must set the ETS environment variables** at the beginning of your scripts to ensure it correctly uses the Qt backend:

```python
import os
os.environ['ETS_TOOLKIT'] = 'qt'
os.environ['QT_API'] = 'pyqt5'

# Import other modules after setting environment
import open3d as o3d
import numpy as np
from mayavi import mlab
from pathlib import Path
# ...
```

> ⚠️ **Offscreen Rendering**: The rotation tool uses `xvfb-run` for offscreen rendering. Install if needed:
> ```bash
> sudo apt-get install xvfb
> ```












