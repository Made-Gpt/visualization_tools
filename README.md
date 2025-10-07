# Tools for visualization

## Install

#### 🚀 Mayavi Setup and Testing Guide 

We highly recommend building a clean, dedicated environment for **Mayavi**.

```bash
# install
conda create -n mayavi_clean python=3.9 -c conda-forge
conda activate mayavi_clean
conda install -c conda-forge mayavi
conda install -c conda-forge open3d
```

You can validate if Mayavi and its dependencies (like the **Qt backend** and **TraitsUI**) run successfully using the following tests. 

(a) Test Qt backend

```python
from PyQt5.QtWidgets import QApplication, QLabel
import sys
app = QApplication(sys.argv)
label = QLabel("Hello from Qt")
label.show()
app.exec_()
```

(b) Test TraitsUI

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

 (c) Test Mayavi

```python
import os
os.environ["ETS_TOOLKIT"] = "qt"
os.environ["QT_API"] = "pyqt5"

from mayavi import mlab 
mlab.test_plot3d() 
mlab.show()
```

> 💡When using Mayavi within your scripts, you **must set the ETS environment variables** at the beginning to ensure it correctly uses the Qt backend. 
>
> ```python
> import os
> os.environ['ETS_TOOLKIT'] = 'qt'
> os.environ['QT_API'] = 'pyqt5'
> 
> # 🔻 import other models
> import open3d as o3d
> import numpy as np
> from mayavi import mlab
> from pathlib import Path
> # ... ...
> ```









