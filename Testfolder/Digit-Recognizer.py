# This program uses three different models to recognize digits in a live video feed from a camera.

import cv2
import numpy as np
from digit_recognition import DigitRecognizerLR, DigitRecognizerNN, DigitRecognizerDL

# initialization of model objects
Digit_Recognizer_LR = DigitRecognizerLR()
Digit_Recognizer_NN = DigitRecognizerNN()
Digit_Recognizer_DL = DigitRecognizerDL()

# function to get image contours and threshold
def get_img_contour_thresh(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(
        img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours, img_thresh

# open video capture
cap = cv2.VideoCapture(0)

# while the capture is open, read the frame and get predicted outputs from three models
while (cap.isOpened()):
    ret, img = cap.read()
    img, contours, thresh = get_img_contour_thresh(img)
    ans1 = ''
    ans2 = ''
    ans3 = ''
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 2500:
            # get prediction from logistic regression model
            x, y, w, h = cv2.boundingRect(contour)
            newImage = thresh[y:y + h, x:x + w]
            newImage = cv2.resize(newImage, (28, 28))
            newImage = np.array(newImage)
            newImage = newImage.flatten()
            newImage = newImage.reshape(newImage.shape[0], 1)
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            
            # get prediction from shallow neural network model
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            
            # get prediction from deep neural network model
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

    # draw rectangle and text on image with predicted outputs
    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # display frame and contours
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    
    # break when escape is pressed
    k = cv2.waitKey(10)
    if k == 27:
        break

# release video capture and close all windows
cap.release()
cv2.destroyAllWindows()