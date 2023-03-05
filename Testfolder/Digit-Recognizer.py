# This code takes input from the camera and uses various machine learning models to recognize hand written digits from the camera feed.
# It displays the recognized digits using three models named : Logistic Regression, Shallow Network, and Deep Network.

# This function takes an image and applies canny detection algorithm to extract edges from the image. 
# Then it finds contours from the edges detected, and draws them on the image, and returns the image with the contours and the thresholded image. 

def get_img_contour_thresh(img):
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 30, 200)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thresh = cv2.threshold(imgGray, 120, 255, cv2.THRESH_BINARY_INV)[1]
    return img, contours, thresh

# This code reads the video feed from the default camera of the system.
# It applies machine learning models trained earlier to identify handwritten digits in the camera feed.
# And then it displays the recognized digits using various models. 

while (cap.isOpened()):
    ret, img = cap.read()
    img, contours, thresh = get_img_contour_thresh(img)
    ans1 = ''
    ans2 = ''
    ans3 = ''
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 2500:
            # Applying the Logistic Regression Model on Contour. 
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            
            # Applying the Shallow Neural Network on Contour.
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            
            # Applying the Deep Neural Network on Contour.
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

            # Drawing Rectangle around the Detected Contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Pre-Processing the Contour.
            newImage = thresh[y:y + h, x:x + w]
            newImage = cv2.resize(newImage, (28, 28))
            newImage = np.array(newImage)
            newImage = newImage.flatten()
            newImage = newImage.reshape(newImage.shape[0], 1)

    # Displaying the Recognized Digits on Screen. 
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    k = cv2.waitKey(10)
    if k == 27:
        break