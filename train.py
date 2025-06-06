#!/usr/bin/env python3
"""
YOLO-11n Fine-tuning on LVIS Dataset
Fine-tune YOLO-11n model for 2 epochs on LVIS dataset with 1,203 classes
"""

import os
from ultralytics import YOLO

def main():
    print("YOLO-11n Fine-tuning on LVIS Dataset")
    print("=" * 50)
    
    # Check if CUDA is available
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Dataset configuration path
    data_config = "data/lvis/lvis.yaml"
    
    # Verify data configuration exists
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Data configuration file not found: {data_config}")
    
    # Load pre-trained YOLO-11n model
    print("\nLoading pre-trained YOLO-11n model...")
    model = YOLO("yolo11n.pt")  # This will download if not already present
    
    # Print model info
    print(f"Model: {model.model}")
    
    # Training configuration with advanced improvements
    training_config = {
        "data": data_config,
        "epochs": 30,  # Increased epochs for better learning (max requested)
        "imgsz": 640,  # Standard size for better feature learning
        "batch": 6,   # Reduced batch size for 640px images
        "device": device,
        "project": "models",
        "name": "lvis_yolo11n_v3",
        "exist_ok": True,
        "patience": 15,  # Early stopping patience
        "save": True,
        "plots": True,
        "verbose": True,
        "val": True,
        "cache": False,  # Disable caching to avoid memory issues
        "workers": 4,    # Increased workers
        "pretrained": True,
        "optimizer": "AdamW",  # Explicit optimizer choice
        "lr0": 0.002,    # Slightly higher initial learning rate
        "lrf": 0.01,     # Lower final learning rate for better convergence
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,  # More warmup epochs
        "warmup_momentum": 0.8,  # Warmup momentum
        "warmup_bias_lr": 0.1,   # Warmup bias learning rate
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "amp": True,  # Automatic mixed precision
        "cos_lr": True,  # Cosine learning rate scheduler
        "label_smoothing": 0.1,  # Label smoothing for better generalization
        "hsv_h": 0.015,  # HSV-Hue augmentation
        "hsv_s": 0.7,    # HSV-Saturation augmentation
        "hsv_v": 0.4,    # HSV-Value augmentation
        "mosaic": 1.0,  # Enable mosaic augmentation
        "mixup": 0.15,   # Increased mixup augmentation
        "copy_paste": 0.3,  # Increased copy-paste augmentation
        "degrees": 15.0,     # Increased rotation augmentation
        "translate": 0.2,    # Increased translation augmentation
        "scale": 0.9,        # Increased scale augmentation
        "shear": 3.0,        # Increased shear augmentation
        "perspective": 0.0003,  # Perspective augmentation
        "flipud": 0.5,       # Vertical flip
        "fliplr": 0.5,       # Horizontal flip
        "erasing": 0.4,      # Random erasing augmentation
        "crop_fraction": 1.0,  # Crop fraction
        "auto_augment": "randaugment",  # Advanced augmentation
        "single_cls": False,  # Multi-class detection
        "rect": False,        # Disable rectangular training for better augmentation
        "close_mosaic": 10,   # Disable mosaic in last 10 epochs
    }
    
    print("\nTraining Configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training...")
    print("This may take a while depending on your hardware...")
    
    try:
        # Start training
        results = model.train(**training_config)
        
        print("\nTraining completed successfully!")
        print(f"Results: {results}")
        
        # Save the best model to models directory
        best_model_path = os.path.join("models", "lvis_yolo11n_v3", "weights", "best.pt")
        if os.path.exists(best_model_path):
            print(f"Best model saved at: {best_model_path}")
        else:
            print("Warning: Best model file not found at expected location")
            
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

if __name__ == "__main__":
    main()