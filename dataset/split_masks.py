import cv2
import numpy as np

import os

# Define the paths for input and output folders
input_folder = 'masks'  # original mask
output_left_folder = 'masks_left'  # replace with the path to the output folder
output_right_folder = 'masks_right'  # replace with the path to the output folder

# Get the list of all files in the input folder
file_list = os.listdir(input_folder)

# Process each file in the list
for filename in file_list:
    # Construct the full path of the input file
    input_filepath = os.path.join(input_folder, filename)
    
    # Open the image file
    lung_image = cv2.imread(input_filepath, cv2.IMREAD_GRAYSCALE)
    
    # Use connected components to label the disconnected areas
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(lung_image)

    # Get the centroids of the left and right connected components
    left_centroid = centroids[1] if stats[1, cv2.CC_STAT_AREA] > 0 else None
    right_centroid = centroids[2] if stats[2, cv2.CC_STAT_AREA] > 0 else None

    # Create blank images
    height, width = lung_image.shape
    left_component = np.zeros((height, width), dtype=np.uint8)
    right_component = np.zeros((height, width), dtype=np.uint8)

    if left_centroid is not None and right_centroid is not None:
        left_component_pixels = np.where(labels == 1, 255, 0).astype(np.uint8)
        right_component_pixels = np.where(labels == 2, 255, 0).astype(np.uint8)
        
        # Determine left and right based on the centroid coordinates
        if left_centroid[0] < right_centroid[0]:
            left_component = left_component_pixels
            right_component = right_component_pixels
        else:
            left_component = right_component_pixels
            right_component = left_component_pixels

    # Construct the full output file paths, keeping the filename unchanged
    output_filepath_left = os.path.join(output_left_folder, filename)
    output_filepath_right = os.path.join(output_right_folder, filename)
    # Save the left and right connected components as binary images
    cv2.imwrite(output_filepath_left, left_component)
    cv2.imwrite(output_filepath_right, right_component)

print("Left and right lung components saved successfully.")
