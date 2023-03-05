# The while loop iterates as long as the object 'cap' is open.
while (cap.isOpened()):
    
    # Reads a frame from the input video.
    ret, img = cap.read()
    
    # Gets the image, contours and threshold values for the image.
    img, contours, thresh = get_img_contour_thresh(img)
    
    # Initializes empty strings for the answer for each prediction model.
    ans1 = ''
    ans2 = ''
    ans3 = ''
    
    # If there are contours detected in the image.
    if len(contours) > 0:
        
        # Gets the contour with the largest area.
        contour = max(contours, key=cv2.contourArea)
        
        # If the area is greater than 2500, it predicts the digit.
        if cv2.contourArea(contour) > 2500:
            
            # Gets the x, y, width, and height values of the bounding rectangle of the contour.
            x, y, w, h = cv2.boundingRect(contour)
            
            # Gets a new image of the digit to be predicted.
            newImage = thresh[y:y + h, x:x + w]
            
            # Resizes the new image to 28x28 and flattens it.
            newImage = cv2.resize(newImage, (28, 28))
            newImage = np.array(newImage)
            newImage = newImage.flatten()
            newImage = newImage.reshape(newImage.shape[0], 1)
            
            # Predicts the digit using Logistic Regression model.
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            
            # Predicts the digit using Shallow Network model.
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            
            # Predicts the digit using Deep Network model.
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

    # Draws a rectangle around the original image and puts the predicted digit values for each model on the frame.
    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Shows the frame with predictions and the thresholded image.
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    
    # Waits for a key input to exit the program.
    k = cv2.waitKey(10)
    if k == 27:
        break