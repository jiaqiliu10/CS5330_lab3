# Jiaqi Liu/Pingqi An
# CS 5330 Lab3
# 10/28/2024

import torch
import torch.nn as nn
from lego_dataset import LegoDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from engine import evaluate_map
from torch.utils.data import DataLoader
import utils
import os

# Custom FastRCNNPredictor to adapt to our specific class count
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        # Classification layer for class scores
        self.cls_score = nn.Linear(in_channels, num_classes)
        # Regression layer for bounding box predictions
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)  # Compute class scores
        bbox_deltas = self.bbox_pred(x)  # Compute bounding box deltas
        return scores, bbox_deltas

# Set dataset path from environment variable or use default path
base_path = os.getenv("DATASET_PATH", "CS5330_lab3/")
test_dir = os.path.join(base_path, 'test')  # Test dataset directory

# Load the test dataset
test_dataset = LegoDataset(test_dir, transforms=None)
test_loader = DataLoader(
    test_dataset, 
    batch_size=2, 
    shuffle=False, 
    collate_fn=utils.collate_fn
)

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the existing classifier to match the new class count 
# (2 classes: background and LEGO)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

# Use GPU if available; otherwise, use CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Evaluate the model and calculate mAP at different IoU thresholds
iou_thresholds = [0.4, 0.5, 0.6]
map_scores = {}

for iou_threshold in iou_thresholds:
    print(f"Evaluating mAP at IoU threshold {iou_threshold}")
    # Compute mAP for the given IoU threshold
    map_score = evaluate_map(
        model, 
        test_loader, 
        device=device, 
        iou_threshold=iou_threshold
    )
    map_scores[iou_threshold] = map_score
    print(f"mAP at IoU threshold {iou_threshold}: {map_score:.4f}")

# Output all mAP results
print("mAP scores at different IoU thresholds:")
for iou, score in map_scores.items():
    print(f"IoU={iou}: mAP={score:.4f}")
