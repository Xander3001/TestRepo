# This code captures frames from a video input and identifies digits in the frame using three different models: Logistic Regression, Shallow Network, and Deep Network.

import cv2
import numpy as np
from digit_recognizer_LR import Digit_Recognizer_LR
from digit_recognizer_NN import Digit_Recognizer_NN
from digit_recognizer_DL import Digit_Recognizer_DL
from get_img_contour_thresh import get_img_contour_thresh   # non-function, import of another file

# initialize video capture from default camera
cap = cv2.VideoCapture(0)

# start loop for capturing frames
while (cap.isOpened()):
    # read frame from video capture device
    ret, img = cap.read()

    # process image to extract contours and threshold image
    img, contours, thresh = get_img_contour_thresh(img)

    # initialize placeholder variables for predictions
    ans1 = ''
    ans2 = ''
    ans3 = ''

    # if any contours were found
    if len(contours) > 0:
        # find the contour with the largest area
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 2500:
            # extract bounding box for contour
            x, y, w, h = cv2.boundingRect(contour)

            # crop image to bounding box, resize to 28x28, flatten and reshape to column vector
            newImage = thresh[y:y + h, x:x + w]
            newImage = cv2.resize(newImage, (28, 28))
            newImage = np.array(newImage)
            newImage = newImage.flatten()
            newImage = newImage.reshape(newImage.shape[0], 1)

            # predict digit using Logistic Regression model
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)

            # predict digit using Shallow Network model
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)

            # predict digit using Deep Network model
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

    # draw bounding box on original image and display predictions for each model
    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # display both original image and thresholded image with contours
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)

    # wait for key press to exit
    k = cv2.waitKey(10)
    if k == 27:
        break