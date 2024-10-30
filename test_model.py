# Jiaqi Liu/Pingqi An
# CS 5330 Lab3
# 10/28/2024

import os
import torch
import torchvision
from torchvision.transforms import functional as F
from lego_dataset import LegoDataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.ops import nms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "lego_detector_best.pth"  # Path to the trained model

# Load the Mask R-CNN model
model = maskrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
# Replace classifier for 2 classes (background and LEGO)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes=2
)

# Replace mask predictor
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
    in_features_mask, hidden_layer, num_classes=2
)

# Load the trained model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

print("Model loaded successfully")

# Set dataset path from environment variable or use default path
base_path = os.getenv("DATASET_PATH", "CS5330_lab3/")  # Base dataset directory
test_dir = os.path.join(base_path, 'test/images')  # Test dataset directory within base path

# Function to make predictions on a single image
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")  # Open and convert image to RGB
    image_tensor = F.to_tensor(image).to(device)  # Convert image to tensor and load to device
    with torch.no_grad():
        prediction = model([image_tensor])[0]  # Get predictions from the model
    
    # Apply Non-Maximum Suppression (NMS) with IoU threshold of 0.5
    boxes = prediction['boxes']
    scores = prediction['scores']
    masks = prediction['masks']
    keep = nms(boxes, scores, iou_threshold=0.5)  # Filter overlapping boxes based on IoU
    
    # Keep only boxes, scores, and masks after NMS
    prediction['boxes'] = boxes[keep]
    prediction['scores'] = scores[keep]
    prediction['masks'] = masks[keep]
    
    return image, prediction

# Function to display image with bounding boxes
def display_image_with_boxes(image, prediction, threshold=0.5):  # Set threshold for display
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for idx, score in enumerate(prediction['scores']):
        if score > threshold:  # Display boxes with score above threshold
            box = prediction['boxes'][idx].cpu().numpy()
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)  # Draw bounding box on the image
            print(f"Detected box with score: {score}, location: {box}")

    plt.show()

# Loop through test images and display the results
for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    image, prediction = predict_image(img_path)  # Predict for each image
    display_image_with_boxes(image, prediction)  # Display image with bounding boxes

