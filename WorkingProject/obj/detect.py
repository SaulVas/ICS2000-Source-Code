import torch
import cv2
from pathlib import Path

def load_model(weights_path):
    # Load the YOLOv5 model from the specified weights
    model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local')  # Use 'source=local' if the model is stored locally
    return model

def detect_car(image_path, model):
    # Read the image using OpenCV
    img = cv2.imread(str(image_path))  # Convert Path object to string
    
    # Perform the detection
    results = model(img)
    
    # Extract detection results
    detections = results.pred[0]
    
    # Check if any of the detected objects is a car
    for det in detections:
        class_id = int(det[5])  # The class ID
        if class_id == 0:  # Assuming 'car' is class 2, change this as per your training classes
            return True
    return False

if __name__ == '__main__':
    # Path to the weights file
    weights_path = Path('yolov5/runs/train/car_detection_drone/weights/best.pt')

    # Directory containing the images
    image_dir = Path('new_images/')

    # Load the model once
    model = load_model(weights_path)

    # Iterate through all images in the directory
    for image_path in image_dir.glob('*.JPG'):  # You can adjust the glob pattern to match your image file types, e.g., '*.jpg' or '*.png'
        if detect_car(image_path, model):
            print(f"Car detected in image: {image_path}")
        else:
            print(f"No car detected in image: {image_path}")
