""" 
Project: The Street View House Numbers (SVHN) Recognition
        By: -
            Omar Diab     [20p3176]
            Mostafa Essam [20p2134]
            Judy Khaled   [20p6630]

Under Supervision of Dr Mahmoud Khalil & Eng Marwa Shams

Constraints:-
In this project we didnt use Ai or Machine learning 
We only used Computer Vision as a training for low level Numbers Detection 
using only the detected edges!

                                Accuracy ~= 18%
"""
import scipy.io                   # to read .mat files
import matplotlib.pyplot as plt   # for image Observing
import cv2                        # for filters
from Filters import *             # Importing custom filter functions
import os                         # for file system operations

folder_path = "DataSet"           # Path to the dataset folder

def main():
    trdata = scipy.io.loadmat('train_32x32.mat')    # Loading the training dataset
    labels = trdata['y']                            # Loading the labels of the training dataset
    correct = 0                                     # Counter for correct predictions
    data = 0                                        # Counter for processed data

    # Iterate through the files in the dataset folder
    for filename in os.listdir(folder_path):
        i = int(os.path.splitext(filename)[0])
        c = int(labels[i])

        label = 0
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        
        # Get the dimensions of the image
        height, width, _ = image.shape
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply high pass filter to the gray image
        high9 = high_pass_filter(gray_image, 9)
        
        # Detect alternating pixels in the filtered image
        x1, xmid, x3, y, D1, D2 = detect_alternating_pixels(high9, height, width)
        
        # Skip the current image if any of the Alternation values exceeds 8
        if x1 > 8 or xmid > 8 or x3 > 8 or y > 8:
            continue

        data += 1
        
        # Detect the number based on the alternating pixel values
        label = detectNumber(x1, xmid, x3, y, D1, D2)
    
        # Check if the detected label matches the ground truth label
        if label == c:
            correct += 1

    # Calculate and print accuracy percentage
    perc = float((float(correct) / float(data)) * 100)
    print(f"Accuracy = {perc}%")





def detectNumber(x1, xmid, x3, Y_alternations, D1_alternations, D2_alternations):
    """
    Detects and returns a number based on the Detected Edges.
    Parameters:
      - x1, xmid, x3: The values of three Horizontal Edge detection Lines.
      - Y_alternations: The number of alternations in the Vertical Edge detection Line.
      - D1_alternations: The number of alternations in the diagonal direction TopLeft => BottomRight.
      - D2_alternations: The number of alternations in the diagonal direction TopRight => BottomLeft.

    Returns:
      - The detected number based on the Detected Edges. Returns -1 if no number is detected.
    """
    if Y_alternations==4 and xmid==4:
                return 10
    elif Y_alternations==6:
            if x1==4 and x3==4:
                return 8
            elif (x1==2 or x1==3) and x3 ==4:
                    return 6
            elif (x3==2 or x3==3) and x1==4:
                    return 9   
              
    if Y_alternations == 2:
        if D1_alternations == 2 or D2_alternations == 2:
            return 1
    elif D1_alternations == 2 and D2_alternations == 2:
        return 1

    if Y_alternations == 4:
        if D1_alternations == 3:
            return 2
    elif D1_alternations == 3 and D2_alternations == 2:
        return 2

    if Y_alternations == 6:
        if D1_alternations == 6 or D2_alternations == 4:
            return 3
    elif D1_alternations == 6 and D2_alternations == 4:
        return 3

    if Y_alternations == 4:
        if D2_alternations == 4:
            return 4
    elif D1_alternations == 4 and D2_alternations == 4:
        return 4

    if Y_alternations == 6:
        if D1_alternations == 4 or D2_alternations == 6:
            return 5
    elif D1_alternations == 4 and D2_alternations == 6:
        return 5

    if Y_alternations == 4:
        if D1_alternations == 4 or D2_alternations == 2:
            return 7
    elif D1_alternations == 4 and D2_alternations == 2:
        return 7
    
    else:
        # No Number is Detected!
        return -1


def obs(gray_image,c):
    plt.imshow(gray_image, cmap='gray')
    plt.title(f"Class: {c}")
    plt.show()

# Call the main code
if __name__ == '__main__':
    main()