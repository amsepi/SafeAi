import torch
import torchvision
import cv2
import yaml
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F


def load_config(yaml_path):
    # Load the dataset configuration from the YAML file
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


def load_model(num_classes, model_path, device):
    # Load the pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def infer_image(model, img_path, class_names, device, threshold=0.5):
    # Load and preprocess the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    # Filter out low-confidence detections
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    high_conf_indices = scores > threshold
    boxes = boxes[high_conf_indices]
    labels = labels[high_conf_indices]
    scores = scores[high_conf_indices]

    # Draw bounding boxes and labels on the image
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        label = f"{class_names[labels[i]]}: {scores[i]:.2f}"
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def main():
    # Load configuration and class names
    config_path = r'C:\Users\user\Downloads\archive2\hit-uav\dataset.yaml'  # Use raw string
    config = load_config(config_path)
    class_names = config['names']
    num_classes = config['nc'] + 1  # Classes + background
    
    # Set device and load model
    device = torch.device('cpu') 
    # device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')
    model_path = 'faster_rcnn_hit_uav.pth'
    model = load_model(num_classes, model_path, device)

    # Path to a single test image
    img_path = r'C:\Users\user\Downloads\archive2\hit-uav\images\test\1_130_60_0_03936.jpg'

    # Perform inference and visualize the result
    infer_image(model, img_path, class_names, device)


if __name__ == "__main__":
    main()
