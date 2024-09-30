import torch
import yaml
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import cv2

# Define your dataset class
class HITUAVDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(img_dir)))
        self.labels = list(sorted(os.listdir(label_dir)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load corresponding label
        label_path = os.path.join(self.label_dir, self.labels[idx])
        boxes = []
        labels = []

        # Read label file and parse it
        with open(label_path, 'r') as f:
            for line in f.readlines():
                label_data = line.strip().split()
                class_id = int(label_data[0])
                # Assuming your label format is [class_id, xmin, ymin, width, height]
                xmin = float(label_data[1])
                ymin = float(label_data[2])
                width = float(label_data[3])
                height = float(label_data[4])
                xmax = xmin + width
                ymax = ymin + height

                # Ensure width and height are positive
                if width <= 0 or height <= 0:
                    print(f"Invalid box found: {label_data}, skipping...")
                    continue

                labels.append(class_id)
                boxes.append([xmin, ymin, xmax, ymax])  # Convert to [xmin, ymin, xmax, ymax]

        # Handle cases where no valid boxes are found
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        img = transforms.ToTensor()(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


def load_config(yaml_path):
    # Load the dataset configuration from the YAML file
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    # Load the dataset configuration
    dataset_yaml = r'C:\Users\user\Downloads\archive2\hit-uav\dataset.yaml'  # Use raw string
    config = load_config(dataset_yaml)

    # Set paths from the YAML file
    train_img_dir = config['train']
    val_img_dir = config['val']
    train_label_dir = train_img_dir.replace('images', 'labels')
    val_label_dir = val_img_dir.replace('images', 'labels')

    # Parameters
    epochs = 1
    batch_size = 16
    device = torch.device('cpu')
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define datasets and dataloaders
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = HITUAVDataset(train_img_dir, train_label_dir, transforms=transform)
    val_dataset = HITUAVDataset(val_img_dir, val_label_dir, transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Load Faster R-CNN model with updated weights parameter
    model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = config['nc'] + 1  # Number of classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Create a new box predictor using FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to the device
    model.to(device)

    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass and calculate loss
            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backpropagation
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                print(f"Epoch [{epoch+1}], Loss: {losses.item()}")
            except Exception as e:
                print(f"Error during training: {e}")
                continue

    # Save the trained model
    torch.save(model.state_dict(), 'faster_rcnn_hit_uav.pth')

    # Plot training results (loss curve, mAP, etc.)
    plot_results()

    print("Training complete!")


def plot_results():
    # Placeholder: Create a dummy plot for the results (adjust based on what you log during training)
    plt.figure(figsize=(10, 5))
    plt.plot([0, 1, 2, 3], [1.0, 0.9, 0.8, 0.7], label='Loss')  # Replace with actual loss values during training
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results.png')
    plt.show()

    # Optionally, display the final plot in the terminal
    img = plt.imread('results.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
