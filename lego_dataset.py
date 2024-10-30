# Jiaqi Liu/Pingqi An
# CS 5330 Lab3
# 10/28/2024

import os
import torch
import xml.etree.ElementTree as ET
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

# Custom dataset class for loading LEGO images and annotations
class LegoDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # List and sort all image files in the "images" directory
        self.imgs = sorted([
            f for f in os.listdir(os.path.join(root, "images")) 
            if f.endswith(".jpg") or f.endswith(".png")
        ])
        # List and sort all annotation files in the "annotations" directory
        self.annotations = sorted([
            f for f in os.listdir(os.path.join(root, "annotations")) 
            if f.endswith(".xml")
        ])
        
        # Ensure the number of images matches the number of annotations
        assert len(self.imgs) == len(self.annotations), (
            "Images and annotations count do not match!"
        )

    def __len__(self):
        # Return the total number of images
        return len(self.imgs)

    def __getitem__(self, idx):
        # Get paths for the image and its corresponding annotation
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annotation_path = os.path.join(self.root, "annotations", self.annotations[idx])
        # Read and normalize the image to [0, 1] range
        img = read_image(img_path).float().div(255)

        # Load annotation data
        boxes, labels, areas, masks = self.load_annotations(
            annotation_path, 
            img.shape[1], 
            img.shape[2]
        )
        
        # Prepare the target dictionary with bounding boxes, labels, and other information
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor(boxes, dtype=torch.float32), 
                format="XYXY", 
                canvas_size=F.get_size(img)
            ),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(areas, dtype=torch.float32),
            # Indicating that all objects are not crowd
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64), 
            "masks": torch.stack(masks)  # Stack masks to create a 3D tensor
        }

        # Apply any transformations if provided
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target

    # Function to load bounding boxes, labels, areas, and masks from XML annotation
    def load_annotations(self, annotation_path, img_height, img_width):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []
        areas = []
        masks = []
        
        # Extract information for each object in the annotation
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            # Append the bounding box coordinates
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Assume all objects are labeled as 1
            # Calculate area of the bounding box
            areas.append((xmax - xmin) * (ymax - ymin))

            # Create a binary mask for the object
            mask = torch.zeros((img_height, img_width), dtype=torch.uint8)
            mask[ymin:ymax, xmin:xmax] = 1  # Set the object region to 1, background is 0
            masks.append(mask)

        return boxes, labels, areas, masks
