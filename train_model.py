# Jiaqi Liu/Pingqi An
# CS 5330 Lab3
# 10/28/2024

import os
import torch
import torchvision
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from lego_dataset import LegoDataset
from torchvision.transforms import v2 as T
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Set dataset path
base_path = os.getenv("DATASET_PATH", "CS5330_lab3/")
train_dir = os.path.join(base_path, 'train')
val_dir = os.path.join(base_path, 'validation')

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10  # Number of training epochs
batch_size = 4  # Batch size
learning_rate = 0.001  # Learning rate

# Data augmentation and transformation
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))  # Random horizontal flip
    transforms.append(T.ToDtype(torch.float, scale=True))  # Convert data type
    transforms.append(T.ToPureTensor())  # Convert to pure Tensor format
    return T.Compose(transforms)

# Initialize the training dataset with transformations
train_dataset = LegoDataset(
    train_dir,
    transforms=get_transform(train=True)
)

# Create the DataLoader with specified batch size and collate function
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

# Configure Mask R-CNN model
def get_model_instance_segmentation(num_classes):
    # Initialize Mask R-CNN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None) 
    # Load pre-trained weights
    model.load_state_dict(
        torch.load(
            "CS5330_lab3/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth", 
            weights_only=True
        )
    )  

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Set classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    # Configure Mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

model = get_model_instance_segmentation(num_classes=2)  # Create detection model with 2 classes
model.to(device)  # Load model to device

# Configure optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, 
    lr=learning_rate, 
    momentum=0.9, 
    weight_decay=0.0005
)

# Training loop
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]  # Load images to device
        # Load targets to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Compute loss
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()

        # Backpropagation
        optimizer.zero_grad()  # Clear gradients
        losses.backward()  # Compute gradients
        optimizer.step()  # Update weights

    # Calculate average loss
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "lego_detector_best.pth") # Save model weights
        print(f"Best model saved with loss {best_loss:.4f}")

print("Training complete.")
