# YOLO Prediction Guide

## Basic Usage

Run predictions using either Python or command line:

```python
from ultralytics import YOLO

# Load and predict
model = YOLO("yolo11n.pt")  # load model
results = model.predict("path/to/image.jpg")  # predict on image
```

```bash
# CLI command
yolo predict model=yolo11n.pt source="path/to/image.jpg"
```

## Prediction Arguments

### Essential Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `source` | `'ultralytics/assets'` | Path to image/video/directory |
| `conf` | `0.25` | Confidence threshold |
| `iou` | `0.7` | NMS IoU threshold |
| `imgsz` | `640` | Input image size |
| `device` | `None` | Device to run on (cpu, 0, 0,1) |

### Processing Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `half` | `False` | Use FP16 half-precision |
| `batch` | `1` | Batch size for processing |
| `rect` | `True` | Use rectangular inference |
| `max_det` | `300` | Maximum detections per image |
| `vid_stride` | `1` | Video frame-rate stride |

### Performance Options
| Argument | Default | Description |
|----------|---------|-------------|
| `stream` | `False` | Use memory-efficient generator |
| `stream_buffer` | `False` | Buffer frames in video streams |
| `augment` | `False` | Apply test-time augmentation |
| `agnostic_nms` | `False` | Class-agnostic NMS |
| `retina_masks` | `False` | High-resolution segmentation masks |

### Output Controls
| Argument | Default | Description |
|----------|---------|-------------|
| `project` | `None` | Project directory name |
| `name` | `None` | Run name within project |
| `verbose` | `True` | Print detailed information |
| `visualize` | `False` | Visualize model features |
| `classes` | `None` | Filter by class (list of IDs) |
| `embed` | `None` | Extract feature embeddings |

## Special Features

1. **Video Processing**:
   ```python
   # Process video with frame skipping
   model.predict("video.mp4", vid_stride=2)
   ```

2. **Memory-Efficient Processing**:
   ```python
   # Use generator for large datasets
   results = model.predict("folder/*.jpg", stream=True)
   ```

3. **Batch Processing**:
   ```python
   # Process multiple images in batches
   model.predict(["img1.jpg", "img2.jpg"], batch=2)
   ```

4. **Real-time Streaming**:
   ```python
   # Process webcam feed
   model.predict(source=0, stream_buffer=False)
   ```

## Advanced Usage Examples

1. **High-Precision Detection**:
   ```python
   model.predict(
       source="images/",
       conf=0.5,        # Higher confidence threshold
       iou=0.6,         # Stricter NMS
       max_det=100      # Limit detections
   )
   ```

2. **Fast Inference Setup**:
   ```python
   model.predict(
       source="video.mp4",
       half=True,       # FP16 inference
       vid_stride=2,    # Skip frames
       imgsz=384       # Smaller image size
   )
   ```

3. **Feature Extraction**:
   ```python
   model.predict(
       source="images/",
       embed=[1,2,3],   # Extract embeddings
       retina_masks=True,  # High-res masks
       augment=True       # Test-time augmentation
   )
   ```
