# This loop runs as long as the video capture is available
while (cap.isOpened()):
    # This reads the image from the video capture
    ret, img = cap.read()
    # This gets the image, contours and threshold
    img, contours, thresh = get_img_contour_thresh(img)
    # These are the initialization of the three answer variables
    ans1 = ''
    ans2 = ''
    ans3 = ''
    # This checks if there are any contours available
    if len(contours) > 0:
        # This gets the contour with the maximum area
        contour = max(contours, key=cv2.contourArea)
        # This checks if the area of the contour is greater than 2500
        if cv2.contourArea(contour) > 2500:
            # This gets the x, y, width and height of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # This crops the image to get only the new image
            newImage = thresh[y:y + h, x:x + w]
            # This resizes the new image to 28x28
            newImage = cv2.resize(newImage, (28, 28))
            # This converts the new image to an numpy array
            newImage = np.array(newImage)
            # This flattens the new image
            newImage = newImage.flatten()
            # This reshapes the new image to  a 1D array
            newImage = newImage.reshape(newImage.shape[0], 1)
            # These three lines predict the answer for the three models
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

    # This sets the x, y, width and height of the rectangle to be drawn
    x, y, w, h = 0, 0, 300, 300
    # This draws the green rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # These put text on the image showing the prediction for the three models
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # This displays the original image and also the threshold image
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    # This waits for a key press to break out of the loop
    k = cv2.waitKey(10)
    if k == 27:
        break