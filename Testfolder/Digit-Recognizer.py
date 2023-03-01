
    while (cap.isOpened()):
        ret, img = cap.read()
        img, contours, thresh = get_img_contour_thresh(img)
        ans1 = ''
        ans2 = ''
        ans3 = ''
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                # print(predict(w_from_model,b_from_model,contour))
                x, y, w, h = cv2.boundingRect(contour)
                # newImage = thresh[y - 15:y + h + 15, x - 15:x + w +15]
                newImage = thresh[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (28, 28))
                newImage = np.array(newImage)
                newImage = newImage.flatten()
                newImage = newImage.reshape(newImage.shape[0], 1)
                ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
                ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
                ans3 = Digit_Recognizer_DL.predict(d3, newImage)

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
        k = cv2.waitKey(10)
        if k == 27:
            break

This code reads frames from a video capture object using the `isOpen()` method of it. Then the `get_img_contour_thresh()` function is called, which takes an image as input, and returns the image, the contours detected in the image, and a binary threshold image. If there are any contours detected in the image, the code selects the contour with the largest area, provided it is larger than 2500. The contour area is calculated using `cv2.contourArea()`, which calculates the area of the contour. 

If the contour area is larger than 2500, it is assumed to contain a digit, and the bounding rectangle is determined with `cv2.boundingRect()`. A new image is then created containing only the digit, and the image is resized to 28 x 28 pixels using `cv2.resize()`. The image is then flattened and reshaped to 784 x 1, to make it acceptable input to the three digit recognition models used. The three models are `Digit_Recognizer_LR` (logistic regression), `Digit_Recognizer_NN` (shallow neural network), and `Digit_Recognizer_DL` (deep neural network). 

The predicted digits from each of the models are stored in the variables `ans1`, `ans2`, and `ans3`. 

Finally, a rectangle is drawn on the original image around the area where the digit was detected, and the predicted digits are displayed on the image using `cv2.putText()`. The image and the contours are displayed in separate windows using `cv2.imshow()`. The code then waits for a key press, and if the key pressed is the 'Esc' key (key code 27), the loop is ended using a `break` statement.