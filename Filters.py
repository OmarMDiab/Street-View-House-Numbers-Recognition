""" 
This file is used for:- 
1) Internal program Computations and logic. 
2) Filters we tried in the project.
"""
import numpy as np
import cv2
from PIL import Image
import os


# Our main Logic
def detect_alternating_pixels(imo,height,width):
    
    # Set the directory for saving the image
    directory = r'EdgeDetectionRoom'

    # Change the working directory to the specified directory for saving the image
    os.chdir(directory) 
    cv2.imwrite("temp.png", imo)
    image = Image.open("temp.png") 
    # Get the pixel data
    pixels = image.load()
    
    # Initialize variables
    XMid_alternations = 0
    X1_alternations=0
    X3_alternations=0
    Y_alternations = 0
    D1_alternations=0
    D2_alternations=0
    StartPixel=0
    prev_pixel = None
    
    half_width=int(width/2)
    half_height=int(height/2)
    firstquarter=int(height/4)
    thirdquarter=int((height*3)/4)

# Check for alternations in the first quarter of the image
    for x in range(width):
        pixel = pixels[x, firstquarter]
        if pixel != 255 and pixel != 0:
            continue
        # Check for an alternation
        if prev_pixel is not None:
                if (prev_pixel == 0 and pixel == 255) or (prev_pixel == 255 and pixel == 0):
                        X1_alternations+= 1

        prev_pixel = pixel

    prev_pixel=None

# Check for alternations in the third quarter of the image
    for x in range(width):
        pixel = pixels[x, thirdquarter]
        if pixel != 255 and pixel != 0:
            continue
        # Check for an alternation
        if prev_pixel is not None:
                if (prev_pixel == 0 and pixel == 255) or (prev_pixel == 255 and pixel == 0):
                    X3_alternations += 1
        prev_pixel = pixel

    prev_pixel=None

# Check for alternations in the middle horizontal line of the image
    for x in range(width):
        pixel = pixels[x, half_height]
        if pixel != 255 and pixel != 0:
             pixel=255
        if x==0:
             StartPixel=pixel
        if x== width-1:
             if pixel!=StartPixel:
                  XMid_alternations += 1

        # Check for an alternation
        if prev_pixel is not None:
                if (prev_pixel == 0 and pixel == 255) or (prev_pixel == 255 and pixel == 0):
                    XMid_alternations += 1

        prev_pixel = pixel
    StartPixel=None
    prev_pixel=None

# Check for alternations in the middle vertical line of the image
    for y in range(height):
        pixel = pixels[half_width, y]
        if pixel != 255 and pixel != 0:
             pixel=255
        if y==0:
             StartPixel=pixel
        if y== height-1:
             if pixel!=StartPixel:
                   Y_alternations+= 1

        if prev_pixel is not None:
                if (prev_pixel == 0 and pixel == 255) or (prev_pixel == 255 and pixel == 0):
                    Y_alternations += 1

        prev_pixel = pixel

# Check for Diagonals Alternations
    for y in range(height):
        x = y
        if x >= width:
            break
        pixel = pixels[x, y]
        if pixel != 255 and pixel != 0:
            pixel = 255
        # Check for an alternation
        if prev_pixel is not None:
            if (prev_pixel == 0 and pixel == 255) or (prev_pixel == 255 and pixel == 0):
                D1_alternations += 1

        prev_pixel = pixel

    x = width - 1
    for y in range(height):
        if x < 0:
            break
        pixel = pixels[x, y]
        if pixel != 255 and pixel != 0:
            pixel = 255
        # Check for an alternation
        if prev_pixel is not None:
            if (prev_pixel == 0 and pixel == 255) or (prev_pixel == 255 and pixel == 0):
                D2_alternations += 1

        x -= 1

        prev_pixel = pixel


    # Change the working directory to the specified directory
    directory = r'..'
    os.chdir(directory) 
    
    return X1_alternations,XMid_alternations,X3_alternations,Y_alternations,D1_alternations,D2_alternations

    


# Filters >>>>>>>>>>>>>>>>>>>>>>>>
# 1) highpass Filter
def high_pass_filter(image, kernel_size):
    # Initialize a kernel matrix with all elements set to -1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) * (-1)
    # Calculate the index of the center of the kernel
    center = kernel_size // 2

    # Set the center value to the sum of all kernel elements minus one.
    # This ensures that the sum of the kernel coefficients remains zero.
    kernel[center, center] = kernel_size * kernel_size - 1

    # Convolve the image with the defined kernel
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image  # Return the resulting filtered image

# 2) Sobel
def sobel_edge_detection(image):  # both
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Apply the Sobel operator
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of the gradient
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    # Normalize the gradient magnitude to 0-255
    gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Apply thresholding to obtain binary edges
    _, edges = cv2.threshold(gradient_normalized, 60, 255, cv2.THRESH_BINARY)
    
    return edges

# 3) HoughCircles: used for counting circles in an image
def CountCircles(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=5, maxRadius=100)
    if circles is not None:
        return len(circles)
    else:
        return 0

# note: we used alot of filters but high_pass_filter was the best Filter we used