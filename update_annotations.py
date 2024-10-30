# Jiaqi Liu/Pingqi An
# CS 5330 Lab3
# 10/28/2024

import os
import xml.etree.ElementTree as ET

# Get the dataset path, prioritizing the environment variable DATASET_PATH. 
# If not set, use the default path 'CS5330_lab3/'
base_path = os.getenv('DATASET_PATH', 'CS5330_lab3/')

# Dynamically set the directories for annotation files
annotation_dirs = [
    os.path.join(base_path, 'train/annotations'),
    os.path.join(base_path, 'validation/annotations'),
    os.path.join(base_path, 'test/annotations')
]

# Function to update labels to 'lego'
def update_annotations(annotation_dir):
    for filename in os.listdir(annotation_dir):
        if filename.endswith('.xml'):
            file_path = os.path.join(annotation_dir, filename)
            tree = ET.parse(file_path)  # Parse the XML file
            root = tree.getroot()

            # Update label in each XML file
            for obj in root.findall('object'):
                name = obj.find('name')
                name.text = 'lego'  # Change label to 'lego'

            # Save the updated XML file
            tree.write(file_path)

# Main function to execute label updates
def main():
    for annotation_dir in annotation_dirs:
        update_annotations(annotation_dir)
    print("Labels in all annotation files have been changed to 'lego'")

if __name__ == "__main__":
    main()
