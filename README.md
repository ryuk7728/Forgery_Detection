# Document Forgery Detection System

A deep learning-based system that detects and localizes forged regions in digital documents, developed for a hackathon project.

## Overview 
The system combines Convolutional Neural Networks (CNN) and YOLOv8 to detect document forgeries through a multi-stage process:

1. Error Level Analysis (ELA) preprocessing to highlight potential tampering
2. CNN classification to identify forged documents with confidence scores
3. YOLOv8 object detection to place bounding boxes around suspected forged regions

## Features
- Real-time forgery detection using deep learning models
- Training mode for continuous model improvement
- Visual output with highlighted forgery regions
- Confidence scoring for forgery classification

## Prerequisites
- Python 3.8+
- TensorFlow 
- OpenCV
- PIL
- Ultralytics YOLOv8

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/Forgery_Detection.git
cd Forgery_Detection
```

## Testing with Sample Images

### Running Inference
The system includes test images to verify the forgery detection pipeline. Here's how to test:

```bash
# Test on authentic document
python src/inference.py --input test_data/authentic/2.jpg

# Test on forged document 
python src/inference.py --input test_data/forged/1t.jpg
```

# Output Format and Image Saving

## Output Structure
When running inference, the system saves the output in the current directory as 'output_image_with_boxes.jpg'

