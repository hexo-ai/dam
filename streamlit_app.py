import streamlit as st
import os
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import random
import glob
import cv2
import tempfile

# Set paths
DATA_DIR = "/home/ubuntu/eureka/bugman/services/sandboxes/68415b7e634f0aa3adf601f0/data"
MODELS_DIR = "/home/ubuntu/eureka/bugman/services/sandboxes/68415b7e634f0aa3adf601f0/snapshots/393d73210dab69509b15ca5216a1af924f7611a2/models"

# Set page config
st.set_page_config(
    page_title="YOLO-11n Object Detection",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("YOLO-11n Object Detection")
st.markdown("This app uses a fine-tuned YOLO-11n model to detect objects in images from the LVIS dataset.")

# Load model
@st.cache_resource
def load_model():
    # Find the best model in the models directory
    model_path = os.path.join(MODELS_DIR, "lvis_yolo11n_v3", "weights", "best.pt")
    
    # If best model not found, use the base model
    if not os.path.exists(model_path):
        st.warning("Best model not found. Using base YOLO-11n model.")
        model = YOLO("yolo11n.pt")
    else:
        st.success(f"Loaded fine-tuned model from: {model_path}")
        model = YOLO(model_path)
    
    return model

# Load class names
@st.cache_data
def load_class_names():
    try:
        lvis_yaml = os.path.join(DATA_DIR, "lvis", "lvis.yaml")
        if os.path.exists(lvis_yaml):
            import yaml
            with open(lvis_yaml, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    return data['names']
        
        # Fallback to looking for names.txt
        names_path = os.path.join(DATA_DIR, "lvis", "names.txt")
        if os.path.exists(names_path):
            with open(names_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Error loading class names: {e}")
    
    # Return placeholder if no class names found
    return [f"Class {i}" for i in range(1203)]  # LVIS has 1203 classes

# Function to run prediction
def predict(model, img, conf_threshold=0.25):
    results = model.predict(img, conf=conf_threshold)
    return results[0]

# Function to display results
def display_results(results, class_names):
    # Display the image with bounding boxes
    res_plotted = results.plot()
    st.image(res_plotted, caption='Detected Objects', use_column_width=True)
    
    # Display detection details
    if len(results.boxes) > 0:
        st.subheader("Detection Results:")
        
        # Create a table of results
        data = []
        for i, box in enumerate(results.boxes):
            class_id = int(box.cls.item())
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            confidence = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            data.append({
                "ID": i+1,
                "Class": class_name,
                "Confidence": f"{confidence:.2f}",
                "Bounding Box": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
            })
        
        st.table(data)
    else:
        st.info("No objects detected in the image.")

# Function to get test images
def get_test_images():
    # Look for test images in common locations
    test_dirs = [
        os.path.join(DATA_DIR, "lvis", "test", "images"),
        os.path.join(DATA_DIR, "lvis", "images", "test2017"),
        os.path.join(DATA_DIR, "lvis", "test"),
        os.path.join(DATA_DIR, "lvis", "val", "images"),
        os.path.join(DATA_DIR, "lvis", "images", "val2017"),
        os.path.join(DATA_DIR, "lvis", "val"),
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(test_dir, ext)))
            
            if image_files:
                return image_files, test_dir
    
    # If no test directory found, return empty list
    return [], ""

# Main app
def main():
    # Load model and class names
    model = load_model()
    class_names = load_class_names()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Single Image Prediction", "Batch Prediction", "Test Set Prediction"])
    
    # Tab 1: Single Image Prediction
    with tab1:
        st.header("Single Image Prediction")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        # Confidence threshold
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction when button is clicked
            if st.button("Detect Objects"):
                with st.spinner("Detecting objects..."):
                    results = predict(model, image, conf_threshold)
                    display_results(results, class_names)
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Prediction")
        
        # Multiple image upload
        uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        # Confidence threshold
        batch_conf_threshold = st.slider("Confidence Threshold (Batch)", 0.0, 1.0, 0.25, 0.05)
        
        if uploaded_files:
            if st.button("Process Batch"):
                for uploaded_file in uploaded_files:
                    st.write(f"Processing: {uploaded_file.name}")
                    
                    # Create a column layout
                    col1, col2 = st.columns(2)
                    
                    # Display original image
                    image = Image.open(uploaded_file)
                    with col1:
                        st.image(image, caption="Original Image", use_column_width=True)
                    
                    # Make prediction and display results
                    with col2:
                        with st.spinner("Detecting objects..."):
                            results = predict(model, image, batch_conf_threshold)
                            res_plotted = results.plot()
                            st.image(res_plotted, caption="Detected Objects", use_column_width=True)
                    
                    # Display detection details
                    if len(results.boxes) > 0:
                        with st.expander(f"Detection details for {uploaded_file.name}"):
                            data = []
                            for i, box in enumerate(results.boxes):
                                class_id = int(box.cls.item())
                                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                                confidence = box.conf.item()
                                
                                data.append({
                                    "ID": i+1,
                                    "Class": class_name,
                                    "Confidence": f"{confidence:.2f}"
                                })
                            
                            st.table(data)
                    else:
                        st.info(f"No objects detected in {uploaded_file.name}")
                    
                    st.markdown("---")
    
    # Tab 3: Test Set Prediction
    with tab3:
        st.header("Test Set Prediction")
        
        # Get test images
        test_images, test_dir = get_test_images()
        
        if not test_images:
            st.warning("No test images found. Please check the data directory structure.")
        else:
            st.success(f"Found {len(test_images)} test images in {test_dir}")
            
            # Confidence threshold
            test_conf_threshold = st.slider("Confidence Threshold (Test)", 0.0, 1.0, 0.25, 0.05)
            
            # Number of random images to display
            num_images = st.number_input("Number of random images to process", min_value=1, max_value=min(10, len(test_images)), value=3)
            
            if st.button("Process Random Test Images"):
                # Select random images
                selected_images = random.sample(test_images, int(num_images))
                
                for img_path in selected_images:
                    st.write(f"Processing: {os.path.basename(img_path)}")
                    
                    # Create a column layout
                    col1, col2 = st.columns(2)
                    
                    # Display original image
                    image = Image.open(img_path)
                    with col1:
                        st.image(image, caption="Original Image", use_column_width=True)
                    
                    # Make prediction and display results
                    with col2:
                        with st.spinner("Detecting objects..."):
                            results = predict(model, img_path, test_conf_threshold)
                            res_plotted = results.plot()
                            st.image(res_plotted, caption="Detected Objects", use_column_width=True)
                    
                    # Display detection details
                    if len(results.boxes) > 0:
                        with st.expander(f"Detection details for {os.path.basename(img_path)}"):
                            data = []
                            for i, box in enumerate(results.boxes):
                                class_id = int(box.cls.item())
                                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                                confidence = box.conf.item()
                                
                                data.append({
                                    "ID": i+1,
                                    "Class": class_name,
                                    "Confidence": f"{confidence:.2f}"
                                })
                            
                            st.table(data)
                    else:
                        st.info(f"No objects detected in {os.path.basename(img_path)}")
                    
                    st.markdown("---")

if __name__ == "__main__":
    main()