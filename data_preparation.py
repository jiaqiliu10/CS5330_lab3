# Jiaqi Liu/Pingqi An
# CS 5330 Lab3
# 10/28/2024
# Regarding setting environment variables, the code first checks the 
# DATASET_PATH environment variable. If it exists, it will use that path 
# as base_path; otherwise, it will use the default CS5330_lab3/ directory.
# CS5330_lab3/ include 50 images and 50 annotations.

import os
import shutil
import random

# Function to set paths, defining the base path for the dataset
def setup_paths(base_path):
    img_dir = os.path.join(base_path, 'images/')# Path for image files
    ann_dir = os.path.join(base_path, 'annotations/')# Path for annotation files
    train_dir = os.path.join(base_path, 'train/')# Path for training set folder
    val_dir = os.path.join(base_path, 'validation/')# Path for validation set folder
    test_dir = os.path.join(base_path, 'test/')# Path for test set folder
    return img_dir, ann_dir, train_dir, val_dir, test_dir

# Create dataset folder
def create_directories(train_dir, val_dir, test_dir):
    for split_dir in [train_dir, val_dir, test_dir]:
        # Create image folder
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        # Create annotation folder
        os.makedirs(os.path.join(split_dir, 'annotations'), exist_ok=True)

# Split the dataset
def split_dataset(image_files, train_ratio=0.7, val_ratio=0.15):
    total = len(image_files) # Total number of images
    train_split = int(train_ratio * total)  # Number of training images
    val_split = int(val_ratio * total)  # Number of validation images
    # Split image files according to the ratio
    train_files = image_files[:train_split]
    val_files = image_files[train_split:train_split + val_split]
    test_files = image_files[train_split + val_split:]
    return train_files, val_files, test_files

# Function to transfer files
def transfer_files(file_list, source_img_dir, source_ann_dir, target_img_dir, target_ann_dir):
    for img_file in file_list:
        ann_file = img_file.replace('.jpg', '.xml')
        shutil.copy(os.path.join(source_img_dir, img_file), target_img_dir)
        shutil.copy(os.path.join(source_ann_dir, ann_file), target_ann_dir)

# Main function for dataset initialization and splitting
def main():
    # Read base path from environment variable, or use default if not set
    base_path = os.getenv("DATASET_PATH", "CS5330_lab3/")
    img_dir, ann_dir, train_dir, val_dir, test_dir = setup_paths(base_path)
    
    # Create necessary directories
    create_directories(train_dir, val_dir, test_dir)
    
    # Load and shuffle image files
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    random.shuffle(image_files)
    
    # Split dataset into train, val, and test
    train_files, val_files, test_files = split_dataset(image_files)
    
    # Transfer files for the train dataset
    transfer_files(
        train_files, 
        img_dir, ann_dir, 
        os.path.join(train_dir, 'images'), 
        os.path.join(train_dir, 'annotations')
    )
    # Transfer files for the validation dataset
    transfer_files(
        val_files, 
        img_dir, 
        ann_dir, 
        os.path.join(val_dir, 'images'), 
        os.path.join(val_dir, 'annotations')
    )
    # Transfer files for the test dataset
    transfer_files(
        test_files, 
        img_dir, 
        ann_dir, 
        os.path.join(test_dir, 'images'), 
        os.path.join(test_dir, 'annotations')
    )
    
    print("Dataset preparation and splitting completed.")

if __name__ == "__main__":
    main()