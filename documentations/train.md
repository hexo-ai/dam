# YOLO Training Guide

## Basic Usage

Train a YOLO model using either Python or command line:

```python
from ultralytics import YOLO

# Load and train model
model = YOLO("yolo11n.pt")  # load pretrained model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

```bash
# CLI command
yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
```

## Key Training Arguments

### Essential Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `model` | `None` | Path to model (.pt or .yaml) |
| `data` | `None` | Dataset config file (e.g., coco8.yaml) |
| `epochs` | `100` | Number of training epochs |
| `imgsz` | `640` | Input image size |
| `batch` | `16` | Batch size (use -1 for auto-batch) |

### Training Control
| Argument | Default | Description |
|----------|---------|-------------|
| `device` | `None` | Training device (cpu, 0, 0,1, mps) |
| `workers` | `8` | Number of data loading workers |
| `resume` | `False` | Resume training from last checkpoint |
| `pretrained` | `True` | Use pretrained weights |
| `patience` | `100` | Epochs to wait for no improvement |
| `save` | `True` | Save training checkpoints |
| `cache` | `False` | Cache images in RAM/disk |

### Learning Rate & Optimization
| Argument | Default | Description |
|----------|---------|-------------|
| `lr0` | `0.01` | Initial learning rate |
| `lrf` | `0.01` | Final learning rate fraction |
| `momentum` | `0.937` | SGD momentum/Adam beta1 |
| `weight_decay` | `0.0005` | Optimizer weight decay |
| `warmup_epochs` | `3.0` | Warmup epochs |
| `optimizer` | `'auto'` | Optimizer (SGD, Adam, AdamW, etc.) |

### Loss Components
| Argument | Default | Description |
|----------|---------|-------------|
| `box` | `7.5` | Box loss weight |
| `cls` | `0.5` | Class loss weight |
| `dfl` | `1.5` | Distribution focal loss weight |
| `pose` | `12.0` | Pose loss weight |

### Advanced Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `rect` | `False` | Rectangular training |
| `cos_lr` | `False` | Cosine LR scheduler |
| `close_mosaic` | `10` | Disable mosaic in last N epochs |
| `amp` | `True` | Automatic mixed precision |
| `fraction` | `1.0` | Dataset fraction to use |
| `freeze` | `None` | Freeze layers |
| `seed` | `0` | Random seed |
| `deterministic` | `True` | Deterministic mode |

### Output & Logging
| Argument | Default | Description |
|----------|---------|-------------|
| `project` | `None` | Project directory name |
| `name` | `None` | Run name within project |
| `exist_ok` | `False` | Allow overwriting existing run |
| `plots` | `False` | Generate training plots |
| `save_period` | `-1` | Checkpoint save frequency |
| `val` | `True` | Include validation |

## Special Features

1. **Auto-batch sizing**:
   - Use `batch=-1` for 60% GPU memory
   - Use `batch=0.70` for custom GPU utilization

2. **Multi-GPU Training**:
   ```python
   model.train(device=[0, 1])  # Use GPUs 0 and 1
   ```

3. **Apple Silicon Support**:
   ```python
   model.train(device="mps")  # Train on M1/M2/M3
   ```

4. **Resume Training**:
   ```python
   model.train(resume=True)  # Continue from last checkpoint
   ```
