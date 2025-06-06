# YOLO-Wild

A powerful object detection system built on YOLO (You Only Look Once) architecture, specifically fine-tuned for the LVIS (Large Vocabulary Instance Segmentation) dataset. This project provides both training and inference capabilities, along with a user-friendly Streamlit web interface for real-time object detection.

## Features

- 🎯 Fine-tuned YOLO-11n model for LVIS dataset
- 📊 Interactive web interface using Streamlit
- 🔄 Support for single image and batch processing
- 📈 Comprehensive evaluation metrics
- 🖼️ Real-time object detection with confidence scores
- 📝 Detailed detection results with bounding boxes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolo-wild.git
cd yolo-wild
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

Launch the Streamlit web interface:
```bash
streamlit run streamlit_app.py
```

The web interface provides three main features:
1. Single Image Prediction
2. Batch Prediction
3. Test Set Prediction

### Training

To train the model on your dataset:
```bash
python train.py
```

### Evaluation

To evaluate the model's performance:
```bash
python evaluate.py
```

## Project Structure

```
yolo-wild/
├── streamlit_app.py    # Web interface for object detection
├── train.py           # Training script
├── evaluate.py        # Evaluation script
├── requirements.txt   # Project dependencies
└── documentations/    # Additional documentation
```

## Dependencies

Key dependencies include:
- PyTorch
- Ultralytics YOLO
- Streamlit
- OpenCV
- NumPy
- Pandas

For a complete list of dependencies, see `requirements.txt`.

## Model Details

The project uses YOLO-11n, a state-of-the-art object detection model, fine-tuned on the LVIS dataset. The model can detect objects from 1203 different classes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- Ultralytics for the YOLO implementation
- LVIS dataset team
- Streamlit team for the web framework