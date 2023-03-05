# Open camera and capture video
while (cap.isOpened()):
    # Read the image frame from captured video
    ret, img = cap.read()

    # apply filter and get thresholded image
    img, contours, thresh = get_img_contour_thresh(img)

    # initialize variables for storing predicted values
    ans1 = ''
    ans2 = ''
    ans3 = ''

    # check if contours are found in image
    if len(contours) > 0:
        # select the largest contour
        contour = max(contours, key=cv2.contourArea)
        # check if contour size is larger than 2500
        if cv2.contourArea(contour) > 2500:
            # get the dimensions of bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            # crop the image to area containing number
            newImage = thresh[y:y + h, x:x + w]
            # resize cropped image to 28 x 28 pixels
            newImage = cv2.resize(newImage, (28, 28))
            # convert cropped image into an array
            newImage = np.array(newImage)
            # flatten the image into a single row
            newImage = newImage.flatten()
            # reshape the image
            newImage = newImage.reshape(newImage.shape[0], 1)
            # predict number using logistic regression model
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            # predict number using shallow neural network model
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            # predict number using deep neural network model
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

    # set the dimensions of rectangle to be drawn on image frame
    x, y, w, h = 0, 0, 300, 300
    # draw rectangle on image frame
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # write predicted numbers on image
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # display the image frame and thresholded image
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)

    # wait for key press event
    k = cv2.waitKey(10)
    # if ESC key is pressed, exit the loop and stop capturing
    if k == 27:
        break