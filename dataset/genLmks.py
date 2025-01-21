import cv2
import numpy as np

def move_element_to_front(lst, element):
    if element in lst:
        index = lst.index(element)
        lst[:] = lst[index:] + lst[:index]

def sample_collection(collection, num_points):
    if num_points <= 0:
        return []

    # Calculate the step size
    step_size = len(collection) / num_points

    # Sample points from the collection based on the step size
    sampled_points = [collection[int(i * step_size)] for i in range(num_points)]

    return sampled_points

def genLandmarks(mask_path, set_num_points):
    # Read the image
    image = cv2.imread(mask_path, 0)
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]  # Use OpenCV to extract contours

    lmk = []

    for i, contour in enumerate(contours):

        contour = contour[:, 0, :]

        sampled_points = sample_collection(contour, set_num_points)

        sorted_contour = [] 

        for x, y in sampled_points:
            sorted_contour.append((x, y))

        # Specify the topmost point
        topmost_point = min(sorted_contour, key=lambda p: p[1])

        # Move the specified element to the front of the list, and place the previous part at the end of the list
        move_element_to_front(sorted_contour, topmost_point)
        for i in sorted_contour:
            lmk.append(i)
        
    return lmk
