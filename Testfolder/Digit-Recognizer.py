```python
# This function runs a loop while the camera is open
while (cap.isOpened()):
    # Reads the image from the camera
    ret, img = cap.read()
    
    # Gets the image, contours, and thresholded image
    img, contours, thresh = get_img_contour_thresh(img)
    
    # Initializes the answer strings
    ans1 = ''
    ans2 = ''
    ans3 = ''
    
    if len(contours) > 0:
        # Gets the contour with the largest area
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 2500:
            x, y, w, h = cv2.boundingRect(contour)
            # Crops the image to only include the digit contour
            newImage = thresh[y:y + h, x:x + w]
            # Resizes the image to 28x28
            newImage = cv2.resize(newImage, (28, 28))
            # Flattens the image to a 1D array
            newImage = np.array(newImage)
            newImage = newImage.flatten()
            newImage = newImage.reshape(newImage.shape[0], 1)
            # Gets predictions for the digit from each model
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

    # Draws a green rectangle on the image
    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Puts the predictions on the image
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Displays the images
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    # Waits 10ms for a key press, and breaks the loop if the key is ESC
    k = cv2.waitKey(10)
    if k == 27:
        break
```