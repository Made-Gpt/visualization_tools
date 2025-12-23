# Experiment Visualization Tools

Tools for visualizing training metrics and efficiency analysis.

## Environment

Use the `renderpy` environment:

```bash
conda activate renderpy
```

## Tools

### 1. Training Loss & mIoU Visualization

Visualize training loss curves and mIoU tracking for multiple models.

**Usage:**
```bash
bash vis_train.sh
```

**Features:**
- Multi-model loss comparison
- mIoU curve tracking
- Interactive model selection
- Multiple plot variants (4:3, 16:9 aspect ratios)
- Clean mode (no labels) and numbered mode

**Configuration:** Edit `train_config.json` to add/modify models and log paths.

---

### 2. Efficiency Metrics Analysis

Generate comparison plots for efficiency metrics:
- Accumulated anchors
- Instance features
- Gaussian memory usage

**Usage:**
```bash
bash vis_effect.sh
```

**Features:**
- Mean curves with standard deviation bands
- Optional std band control (`--no-std-band`)
- Multiple aspect ratios
- Automatic fallback (features → anchors if missing)

**Configuration:** Edit `effect_config.json` to configure methods and metrics.

---

### 3. Latency & Parameter Analysis

Create scatter plots comparing:
- Latency vs mIoU
- Parameters vs mIoU

**Usage:**
```bash
bash vis_latency.sh
```

**Features:**
- Interactive scatter plots with guide lines
- Marker size optimization
- Clean mode (no scientific notation)
- Automatic margin adjustment

**Input:** Reads from `effect/*.txt` logs (parses inference time, mIoU, parameter count)

---

### 4. Scene Efficiency Visualization

Visualize accumulated anchors across frames for different methods.

**Usage:**
```bash
bash vis_scene_eff.sh
```

**Features:**
- Per-scene comparison
- Anchor accumulation tracking
- Optional std band control

---

## Configuration Files

- **train_config.json**: Model configurations, log paths, loss definitions
- **effect_config.json**: Method settings, metric configurations
- **scene_config.json**: Scene-specific settings (if used)

## Output Structure

All visualizations are saved to method-specific output folders:
```
experiment/
├── outputs/
│   ├── model_name/
│   │   ├── loss_4_3.png
│   │   ├── loss_16_9.png
│   │   ├── miou_4_3.png
│   │   └── ...
```
