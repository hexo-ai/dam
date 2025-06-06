#!/usr/bin/env python3
"""
YOLO-11n Model Evaluation on LVIS Dataset
Evaluate the fine-tuned YOLO-11n model on LVIS validation set
"""

import os
import json
from ultralytics import YOLO

def main():
    print("YOLO-11n Model Evaluation on LVIS Dataset")
    print("=" * 50)
    
    # Check for trained model (prioritize latest version)
    model_path = "models/lvis_yolo11n_v3/weights/best.pt"
    
    # Fallback to previous models if new one doesn't exist
    if not os.path.exists(model_path):
        model_path = "models/lvis_yolo11n_v2/weights/best.pt"
    
    if not os.path.exists(model_path):
        model_path = "models/lvis_yolo11n/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        print("Please run train.py first to train the model.")
        return
    
    # Load the trained model
    print(f"Loading trained model from: {model_path}")
    model = YOLO(model_path)
    
    # Dataset configuration
    data_config = "data/lvis/lvis.yaml"
    
    print(f"Using data configuration: {data_config}")
    
    # Validation configuration with improved settings
    val_config = {
        "data": data_config,
        "imgsz": 640,   # Match training image size
        "batch": 4,     # Reduced batch size to avoid memory issues
        "conf": 0.001,  # Low confidence threshold for evaluation
        "iou": 0.6,     # Lower IoU threshold for better recall
        "device": "cuda" if hasattr(model.model, 'cuda') else "cpu",
        "half": True,
        "plots": True,
        "save_json": True,  # Save results in JSON format
        "verbose": True,
        "project": "outputs",
        "name": "validation_v3",
        "exist_ok": True,
        "augment": True,  # Test-time augmentation for better performance
        "agnostic_nms": False,  # Class-specific NMS
        "max_det": 300,  # Maximum detections per image
    }
    
    print("\nValidation Configuration:")
    for key, value in val_config.items():
        print(f"  {key}: {value}")
    
    print("\nRunning validation...")
    
    try:
        # Run validation
        metrics = model.val(**val_config)
        
        print("\nValidation completed successfully!")
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        
        # Print key metrics
        if hasattr(metrics, 'box'):
            print(f"mAP50-95 (IoU=0.5:0.95): {metrics.box.map:.4f}")
            print(f"mAP50 (IoU=0.5):         {metrics.box.map50:.4f}")
            print(f"mAP75 (IoU=0.75):        {metrics.box.map75:.4f}")
            print(f"mAP (small):              {metrics.box.maps[0]:.4f}")
            print(f"mAP (medium):             {metrics.box.maps[1]:.4f}")
            print(f"mAP (large):              {metrics.box.maps[2]:.4f}")
        else:
            print("Warning: Box metrics not available")
        
        # Print detailed metrics
        print("\nDetailed Results:")
        print(f"Results object: {type(metrics)}")
        print(f"Available attributes: {dir(metrics)}")
        
        # Try to get the main evaluation metric
        main_score = 0.0
        if hasattr(metrics, 'box') and hasattr(metrics.box, 'map'):
            main_score = float(metrics.box.map)
            print(f"\nFinal mAP50-95 Score: {main_score:.4f}")
        else:
            print("Warning: Could not extract mAP score")
        
        # Save results summary
        results_file = "outputs/evaluation_results.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        results_summary = {
            "model_path": model_path,
            "data_config": data_config,
            "mAP50-95": main_score,
            "evaluation_status": "completed"
        }
        
        if hasattr(metrics, 'box'):
            results_summary.update({
                "mAP50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
                "mAP75": float(metrics.box.map75) if hasattr(metrics.box, 'map75') else 0.0,
            })
        
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        return main_score
        
    except Exception as e:
        print(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def generate_test_predictions():
    """Generate predictions on test set for submission"""
    print("\n" + "=" * 60)
    print("GENERATING TEST SET PREDICTIONS")
    print("=" * 60)
    
    model_path = "models/lvis_yolo11n_v3/weights/best.pt"
    
    # Fallback to previous models if new one doesn't exist
    if not os.path.exists(model_path):
        model_path = "models/lvis_yolo11n_v2/weights/best.pt"
        
    if not os.path.exists(model_path):
        model_path = "models/lvis_yolo11n/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        return
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Test images path - from test.txt
    test_txt_file = "data/lvis/test.txt"
    
    if not os.path.exists(test_txt_file):
        print(f"Error: Test file not found at {test_txt_file}")
        return
    
    # Read test image paths
    with open(test_txt_file, 'r') as f:
        test_image_paths = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(test_image_paths)} test images")
    
    # Process test images in smaller chunks to avoid OOM
    chunk_size = 50  # Process 50 images at a time
    total_images = len(test_image_paths)
    all_results = []
    
    print(f"Processing {total_images} images in chunks of {chunk_size}")
    
    # Clear GPU cache first
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for i in range(0, total_images, chunk_size):
        chunk_paths = test_image_paths[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (total_images + chunk_size - 1) // chunk_size
        
        print(f"\nProcessing chunk {chunk_num}/{total_chunks} ({len(chunk_paths)} images)")
        
        # Predict on current chunk with improved settings
        predict_config = {
            "source": chunk_paths,
            "conf": 0.25,  # Confidence threshold for test predictions
            "iou": 0.6,    # Lower IoU threshold for better recall
            "imgsz": 640,  # Match training image size for better performance
            "batch": 1,    # Process one image at a time
            "device": "cuda" if hasattr(model.model, 'cuda') else "cpu",
            "half": True,
            "save_json": False,  # Don't save individual chunks
            "project": "outputs",
            "name": f"test_chunk_{chunk_num}",
            "exist_ok": True,
            "verbose": False,  # Reduce verbosity
            "stream": True,   # Use streaming mode for memory efficiency
            "augment": True,  # Test-time augmentation for better performance
            "agnostic_nms": False,  # Class-specific NMS
            "max_det": 300,  # Maximum detections per image
        }
    
        try:
            results = model.predict(**predict_config)
            
            # Collect results from this chunk
            for result in results:
                all_results.append(result)
            
            # Clear GPU cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"Chunk {chunk_num} completed successfully")
            
        except Exception as e:
            print(f"Chunk {chunk_num} failed with error: {e}")
            # Continue with next chunk
            continue
    
    print(f"\nTest predictions completed! Processed {len(all_results)} image results")
    
    # Save all results to a combined JSON file
    try:
        import json
        combined_predictions = []
        
        for result in all_results:
            # Extract predictions from each result
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if len(boxes) > 0:
                    # Convert to COCO format
                    for i in range(len(boxes)):
                        prediction = {
                            "image_id": int(os.path.splitext(os.path.basename(result.path))[0]),
                            "category_id": int(boxes.cls[i].item()),
                            "bbox": boxes.xywh[i].tolist(),  # COCO format: [x, y, width, height]
                            "score": float(boxes.conf[i].item())
                        }
                        combined_predictions.append(prediction)
        
        # Save combined predictions
        os.makedirs("outputs", exist_ok=True)
        predictions_file = "outputs/test_predictions_combined_v3.json"
        
        with open(predictions_file, 'w') as f:
            json.dump(combined_predictions, f)
            
        print(f"Combined predictions saved to: {predictions_file}")
        print(f"Total predictions: {len(combined_predictions)}")
        
    except Exception as e:
        print(f"Error saving combined predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run validation
    score = main()
    
    # Generate test predictions
    generate_test_predictions()
    
    print(f"\nFinal evaluation score: {score:.4f}")