# Smart-Waste-Segregation-System-using-YOLOv8
This project implements an AI-based smart waste classification system using YOLOv8
The model detects and classifies waste objects such as cardboard, plastic, metal, and other categories to support automated waste management systems.

##  Model
- Architecture: YOLOv8n
- Framework: Ultralytics YOLOv8
- Task: Object Detection
- Dataset: Custom annotated waste dataset
- Training Epochs: 120

##  Classes
The model is trained to detect:
- Cardboard
- Plastic
- Metal
- Glass
- Paper
- Other waste types (depending on dataset)


## Results

The model outputs bounding boxes with confidence scores for each detected waste object.

## Example Output
Input: Waste image
Output: Labeled image with detected objects

 
 ## Applications
Smart waste management systems
Recycling automation
Environmental monitoring
Smart city solutions


## Requirements
Python 3.10+
PyTorch
Ultralytics YOLOv8
OpenCV

 
 ## Author
Wasan Al-Salhi


# 📄 requirements.txt

```txt
ultralytics
torch
torchvision
opencv-python
numpy
matplotlib
