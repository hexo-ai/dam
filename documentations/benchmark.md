# YOLO Benchmarking Guide

## Basic Usage

Run benchmarks using either Python or command line:

```python
from ultralytics.utils.benchmarks import benchmark

# Run benchmark on GPU
benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, device=0)

# Benchmark specific format
benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, format="onnx")
```

```bash
# CLI command
yolo benchmark model=yolo11n.pt data=coco8.yaml imgsz=640 device=0
```

## Benchmark Arguments

### Essential Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `model` | `None` | Path to model file (.pt or .yaml) |
| `data` | `None` | Path to data configuration file |
| `imgsz` | `640` | Input image size |
| `device` | `None` | Device to run on (cpu, 0, 0,1) |

### Performance Options
| Argument | Default | Description |
|----------|---------|-------------|
| `half` | `False` | Use FP16 half-precision |
| `int8` | `False` | Use INT8 quantization |
| `verbose` | `False` | Print detailed information |
| `format` | `''` | Specific format to benchmark |

## Export Formats

### CPU Optimized
| Format | Description | Speed Improvement |
|--------|-------------|-------------------|
| ONNX | Optimal CPU performance | Up to 3x speedup |
| OpenVINO | Intel hardware optimization | Up to 3x speedup |

### GPU Optimized
| Format | Description | Speed Improvement |
|--------|-------------|-------------------|
| TensorRT | Maximum GPU efficiency | Up to 5x speedup |

### Platform Specific
| Format | Description |
|--------|-------------|
| CoreML | iOS deployment |
| TensorFlow | General ML applications |

## Key Metrics

1. **Performance Metrics**:
   - mAP50-95: For detection/segmentation
   - accuracy_top5: For classification
   - Inference Time: ms per image

2. **Resource Usage**:
   - Memory consumption
   - Hardware utilization
   - Power efficiency

## Advanced Usage Examples

1. **Full GPU Benchmark**:
   ```python
   benchmark(
       model="yolo11n.pt",
       data="coco8.yaml",
       imgsz=640,
       half=True,    # Enable FP16
       device=0      # Use GPU
   )
   ```

2. **Format-Specific Test**:
   ```python
   benchmark(
       model="yolo11n.pt",
       data="coco8.yaml",
       imgsz=640,
       format="onnx"  # Test only ONNX
   )
   ```

3. **Edge Device Testing**:
   ```python
   benchmark(
       model="yolo11n.pt",
       data="coco8.yaml",
       imgsz=320,    # Smaller size
       int8=True,    # INT8 quantization
       device="cpu"
   )
   ```

## Benefits of Benchmarking

1. **Decision Making**:
   - Compare speed vs accuracy
   - Evaluate hardware options
   - Optimize resource allocation

2. **Optimization**:
   - Identify best export format
   - Fine-tune performance
   - Reduce costs

3. **Deployment Planning**:
   - Hardware requirements
   - Resource allocation
   - Cost estimation
