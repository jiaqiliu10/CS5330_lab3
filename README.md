# CS5330 Lab 3 - LEGO Detection

## File Download and Placement Instructions

To facilitate the project setup, we have stored the larger model files separately on Google Drive. Please follow the steps below to download and place the files to ensure proper code execution.

### Download Link

All required model files are available in the following Google Drive folder:
https://drive.google.com/drive/folders/1BfF1X2G9VI1xftn4HQavyKo_ZjWzOx2L?usp=drive_link

### File Placement

1. **lego_detector_best.pth**
   - **Purpose**: This is the best model weight file for the LEGO detection task. This file was obtained as the best result from previous training epochs using the `train_model` script.
   - **Placement**: After downloading, place the `lego_detector_best.pth` file in the `CS5330_lab3_LEGO` folder.
   - **Reason**: This file contains the trained weights specifically for LEGO detection. Placing it in this folder ensures that `test_model.py` and other related scripts can correctly load this model for LEGO recognition and bounding box drawing.

2. **fasterrcnn_resnet50_fpn_coco-258fb6c6.pth** and **maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth**
   - **Purpose**: These two files are pre-trained weights for the Faster R-CNN and Mask R-CNN models, respectively, used for other computer vision tasks.
   - **Placement**: Place these files in the `CS5330_lab3` folder.
   - **Reason**: These files serve as reference models for additional testing. Keeping them in the `CS5330_lab3` folder helps avoid conflicts between different model files and organizes them for separate tasks within the project.

### Important Note
Under the CS5330_lab3 folder, the annotations and images can be switched to your preferred images and annotations for testing.

### Running the Code
Follow these steps to run the code in the correct order:

1. **Data Preparation**
    - **Run python3 data_preparation.py**
    The purpose of this file is to prepare and split the dataset. The code reads the dataset path from the environment variable ‘DATASET PATH’. If it’s not set, a default path is used. Additionally, the code creates ‘images’ and ‘annotations’ folders under the paths for the training, validation, and test sets.

2. **Update Annotations**
    - **Run python3 update_annotations.py**
    This file is used to update the labels in the annotations to “lego”. The code iterates through all XML annotation files and replaces the labels with “lego”.

3. **Train Model**
    - **Run python3 train_model.py**
    This file is used for model training, selecting the Mask R-CNN model and loading pre-trained weights (maskr-cnn resnet50 fpn)

4. **Test Model**
    - **Run python3 test_model.py**
    This file is used for testing a single image. It loads the MaskR-CNN model, successfully draws the prediction boxes, and sets the score threshold to 0.5.

5. **Evaluate Model**
    - **Run python3 evaluate_model.py**
    For model performance evaluation, outputs the mAP scores at different IoU thresholds and draws detection bounding boxes.
