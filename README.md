
# SVHN Recognition Project - Computer Vision
This project aims to develop object recognition algorithms to identify digits and numbers in natural scene images using the Street View House Numbers (SVHN) dataset with computer Vision.

## About SVHN Dataset

SVHN is a real-world image dataset obtained from house numbers in Google Street View images. It contains over 600,000 digit images with minimal preprocessing requirements, making it suitable for training and evaluating object recognition algorithms.

- **Classes:** There are 10 classes, one for each digit from 0 to 9.Note that digit '0' has label 10.
- **Data Split:** The dataset consists of 73,257 digits for training, 26,032 digits for testing, and an additional 531,131 samples that can be used as extra training data.
- **Formats:** SVHN comes in two formats: original images with character-level bounding boxes and MNIST-like 32-by-32 images centered around a single character.
However, we used the **MNIST-like 32-by-32 images** format

You can download the dataset from [Stanford University](http://ufldl.stanford.edu/housenumbers/)


## Computer Vision Approach

The core of this project lies in the application of computer vision methodologies for digit detection and recognition. Due to the nature of our dataset, which lacks a "100%" closed region, traditional methods like Hough Transform are not suitable. Instead, we employ the **Alternating Edge Concept** for edge detection, which proves to be effective in identifying digit boundaries amidst complex backgrounds.



## Usage

### Alternating Edge Detection

The `detect_alternating_pixels` function implements the alternating edge detection logic. It takes the input image, height, and width as parameters and returns the number of alternations detected in different regions of the image.
<p float="left">
   <img src="https://github.com/OmarMDiab/Street-View-House-Numbers-Recognition/raw/main/Detection%20Technique/no1.png" width="400"  />
  <img src="https://github.com/OmarMDiab/Street-View-House-Numbers-Recognition/raw/main/Detection%20Technique/no6.png" width="400" height="440" />  
</p>


### Filters

The `Filters.py` module contains custom filter functions used in the project, including:

- **High Pass Filter**: Enhances high-frequency components in the image.
- **Sobel Edge Detection**: Detects edges in the image using the Sobel operator.
- **HoughCircles**: Used for counting circles in an image.
Among these, the **high_pass_filter** emerged as the most effective filter. Its ability to highlight high-frequency components in the image significantly contributed to the accuracy of our recognition system.
### Main Functionality

- It reads the image using OpenCV's cv2.imread().
- Converts the image to grayscale using cv2.cvtColor().
- Applies a high-pass filter to enhance image features.
- Detects alternating pixels in the filtered image using detect_alternating_pixels().
- Based on the detected pixel patterns, it attempts to recognize the digit using detectNumber().
- It compares the detected digit with the ground truth label and updates the correctness counter.

**Digit Detection Logic:**

The detectNumber() function implements the logic for recognizing digits based on detected pixel patterns.
It considers various combinations of alternating pixels in horizontal, vertical, and diagonal directions to determine the digit.

**Observation Function:**

The obs() function is a utility function used to visualize the grayscale image with its corresponding class label.


## Constraints

- This project **was restricted** from utilizing AI or machine learning techniques.
- **Only computer vision** methods are used for low-level number detection using detected edges.









