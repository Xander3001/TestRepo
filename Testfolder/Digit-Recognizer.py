This code reads frames from a video stream and uses image processing to predict digits in the video stream. It uses three different models of increasing depth (logistic regression, shallow network, deep network), and displays the predictions for each model in a rectangle on the video frame.

```
# Code for image processing and digit prediction in live video stream
import cv2
import numpy as np
from digit_recognition import Digit_Recognizer_LR, Digit_Recognizer_NN, Digit_Recognizer_DL


def get_img_contour_thresh(img):
    """
    Function to process the image and find contours

    :param img: Numpy array of image
    :return: processed image, contours, and threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours, thresh


cap = cv2.VideoCapture(0)

# Load pre-trained models
w_LR, b_LR = np.load('weights/w_LR.npy'), np.load('weights/b_LR.npy')
d2 = Digit_Recognizer_NN.read_from_file('weights/nn.json', 'weights/nn.h5')
d3 = Digit_Recognizer_DL.read_from_file('weights/dl.json', 'weights/dl.h5')

while (cap.isOpened()):
    # Read frame from video stream
    ret, img = cap.read()

    # Process image to extract contours
    img, contours, thresh = get_img_contour_thresh(img)

    # Predict digits for each model if contour exists
    ans1, ans2, ans3 = '', '', ''
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 2500:
            x, y, w, h = cv2.boundingRect(contour)
            newImage = thresh[y:y + h, x:x + w]
            newImage = cv2.resize(newImage, (28, 28))
            newImage = np.array(newImage)
            newImage = newImage.flatten()
            newImage = newImage.reshape(newImage.shape[0], 1)
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

    # Display predicted digits on video stream
    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)

    # Wait for user input
    k = cv2.waitKey(10)
    if k == 27:
        break

# Release video stream resource
cap.release()
```