# This code captures live video feed from the default camera and performs digit recognition using logistic regression,
# shallow neural network, and deep neural network models. The recognized digit is printed on the video feed in real-time.

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    # Reading each frame of the video feed
    ret, img = cap.read()
    
    # Getting the contour and threshold of the image
    img, contours, thresh = get_img_contour_thresh(img)
    
    ans1 = ''
    ans2 = ''
    ans3 = ''
    if len(contours) > 0:
        # Choosing the contour with maximum area
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 2500:
            # Getting the coordinates and dimensions of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Cropping and resizing the image to 28x28 for model prediction
            newImage = thresh[y:y + h, x:x + w]
            newImage = cv2.resize(newImage, (28, 28))
            newImage = np.array(newImage)
            newImage = newImage.flatten()
            newImage = newImage.reshape(newImage.shape[0], 1)
            
            # Predicting the digit using logistic regression model
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            
            # Predicting the digit using shallow neural network model
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            
            # Predicting the digit using deep neural network model
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

    # Drawing a rectangle on the video feed for image segmentation
    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Printing the recognized digit on the video feed
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Displaying the video feed and contour image
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    
    # Waiting for user input to exit the program
    k = cv2.waitKey(10)
    if k == 27:
        break