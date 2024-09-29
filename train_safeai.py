import os
import torch
from yolov5 import train

# Define paths and settings
dataset_yaml = 'dataset.yaml'  # Path to your dataset.yaml file
weights = 'yolov5s.pt'  # Path to pre-trained YOLOv5 weights
img_size = 640  # Image size for YOLO training
batch_size = 16  # Batch size for training
epochs = 50  # Number of training epochs

# Run the training
def run_training():
    train.run(
        data=dataset_yaml,
        imgsz=img_size,
        batch_size=batch_size,
        epochs=epochs,
        weights=weights,
        device='0'  # Use '0' for GPU, 'cpu' for CPU
    )

if __name__ == '__main__':
    run_training()
