# Imports the necessary packages and modules
import cv2
import numpy as np
from Digit_Recognizer_LR import Digit_Recognizer_LR
from Digit_Recognizer_NN import Digit_Recognizer_NN
from Digit_Recognizer_DL import Digit_Recognizer_DL
from get_img_contour_thresh import get_img_contour_thresh

# This code captures a video feed and detects digits using three different methods. The methods used are Logistic Regression,
# Shallow Network, and Deep Network.

# Define the video capture device 
cap = cv2.VideoCapture(0)

# Loop through each frame of the video feed while it is open
while (cap.isOpened()):
    # Read in the current frame
    ret, img = cap.read()
    
    # Get the image, contours, and threshold values
    img, contours, thresh = get_img_contour_thresh(img)
    
    # Initialize answer variables for each digit recognition method
    ans1 = ''
    ans2 = ''
    ans3 = ''
    
    # If there are contours in the image
    if len(contours) > 0:
        # Select the contour with the largest area
        contour = max(contours, key=cv2.contourArea)
        
        # If the contour area is greater than 2500
        if cv2.contourArea(contour) > 2500:
            
            # Get the x, y, width, and height of the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create a new image within the bounding rectangle of the contour, resize it to 28x28 pixels,
            # flatten the image pixel values, and reshape them
            newImage = thresh[y:y + h, x:x + w]
            newImage = cv2.resize(newImage, (28, 28))
            newImage = np.array(newImage)
            newImage = newImage.flatten()
            newImage = newImage.reshape(newImage.shape[0], 1)
            
            # Use each digit recognition method to predict the value of the digit in the new image
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

    # Draw a rectangle around the entire image and display the predicted digit values for each method
    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the current frame and the threshold image
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    
    # Wait for a key press and break if the 'Esc' key is pressed
    k = cv2.waitKey(10)
    if k == 27:
        break