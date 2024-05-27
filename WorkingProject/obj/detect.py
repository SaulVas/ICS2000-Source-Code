import torch
import cv2
from pathlib import Path

def load_model(weights_path):
    model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local')
    return model

def detect_car(image_path, model):
    img = cv2.imread(str(image_path))

    results = model(img)

    detections = results.pred[0]

    for det in detections:
        class_id = int(det[5])
        if class_id == 0:
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
    for image_path in image_dir.glob('*.JPG'):
        if detect_car(image_path, model):
            print(f"Car detected in image: {image_path}")
        else:
            print(f"No car detected in image: {image_path}")
