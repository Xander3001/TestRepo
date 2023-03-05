# File: digit_recognizer.py
# Description: Captures video input and recognizes digits using logistic regression, shallow network,
#              and deep network models.
# Author: [Your Name]

import cv2
import numpy as np
from digit_recognizer_LR import Digit_Recognizer_LR
from digit_recognizer_NN import Digit_Recognizer_NN
from digit_recognizer_DL import Digit_Recognizer_DL

# Define function to get image contours and threshold
# Parameters:
# - img: input image
# Returns:
# - img: processed image
# - contours: contours detected in the processed image
# - thresh: threshold image
def get_img_contour_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours, thresh

# Define video capture device
cap = cv2.VideoCapture(0)

# Load models
w_LR, b_LR = Digit_Recognizer_LR.load_model()
d2 = Digit_Recognizer_NN.load_model()
d3 = Digit_Recognizer_DL.load_model()

# Loop for capturing video and processing frames
while (cap.isOpened()):
    # Read video frame
    ret, img = cap.read()

    # Process image and get contours and threshold
    img, contours, thresh = get_img_contour_thresh(img)
    
    # Initialize variables for storing model predictions
    ans1 = ''
    ans2 = ''
    ans3 = ''
    
    # If any contours are detected
    if len(contours) > 0:
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # If the contour area is greater than 2500 (i.e. likely a digit)
        if cv2.contourArea(contour) > 2500:
            # Get bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Crop image to bounding box and resize to 28x28 pixels
            newImage = thresh[y:y + h, x:x + w]
            newImage = cv2.resize(newImage, (28, 28))
            
            # Flatten and reshape image for model input
            newImage = np.array(newImage)
            newImage = newImage.flatten()
            newImage = newImage.reshape(newImage.shape[0], 1)
            
            # Get predictions from each model
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

    # Draw rectangle and display predictions on video
    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display video and threshold images
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)

    # Check for escape key to exit loop
    k = cv2.waitKey(10)
    if k == 27:
        break

# Release video capture device and destroy windows
cap.release()
cv2.destroyAllWindows()