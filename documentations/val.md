# YOLO Validation Guide

## Basic Usage

Validate a YOLO model using either Python or command line:

```python
from ultralytics import YOLO

# Load and validate model
model = YOLO("yolo11n.pt")  # load model
metrics = model.val()  # validate model
print(metrics.box.map)  # print mAP50-95
```

```bash
# CLI command
yolo detect val model=yolo11n.pt
```

## Validation Arguments

### Essential Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `data` | `None` | Dataset config file path (e.g., coco8.yaml) |
| `imgsz` | `640` | Input image size |
| `batch` | `16` | Batch size for validation |
| `conf` | `0.001` | Minimum confidence threshold |
| `iou` | `0.7` | NMS IoU threshold |

### Processing Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `device` | `None` | Device to run on (cpu, cuda:0, etc.) |
| `workers` | `8` | Number of data loading workers |
| `half` | `True` | Use half precision (FP16) |
| `dnn` | `False` | Use OpenCV DNN for ONNX inference |
| `rect` | `True` | Use rectangular inference |

### Output Controls
| Argument | Default | Description |
|----------|---------|-------------|
| `project` | `None` | Project directory for outputs |
| `name` | `None` | Run name within project |
| `save_txt` | `False` | Save results to text files |
| `save_json` | `False` | Save results to JSON file |
| `save_conf` | `False` | Save confidences in txt labels |
| `plots` | `False` | Generate plots of results |
| `verbose` | `False` | Print detailed information |

### Detection Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `max_det` | `300` | Maximum detections per image |
| `classes` | `None` | Filter by class (list of IDs) |
| `agnostic_nms` | `False` | Class-agnostic NMS |
| `single_cls` | `False` | Treat as single-class dataset |
| `augment` | `False` | Test-time augmentation |

## Special Features

1. **Accessing Metrics**:
   ```python
   metrics = model.val()
   print(metrics.box.map)    # mAP50-95
   print(metrics.box.map50)  # mAP50
   print(metrics.box.map75)  # mAP75
   ```

2. **Export Results**:
   ```python
   # Save results in different formats
   results = model.val(plots=True)
   results.confusion_matrix.to_df()    # DataFrame
   results.confusion_matrix.to_json()  # JSON
   results.confusion_matrix.to_csv()   # CSV
   ```

3. **Custom Dataset Validation**:
   ```python
   # Validate on custom dataset
   metrics = model.val(data="path/to/custom.yaml")
   ```

4. **Test-Time Augmentation**:
   ```python
   # Enable TTA for better accuracy
   metrics = model.val(augment=True)
   ```
