# Open the camera for capturing video
# cap is a VideoCapture object
while (cap.isOpened()):
    
    # Read the video input
    # ret is a boolean value indicating if reading is successful
    # img is the video frame
    ret, img = cap.read()
    
    # Call function get_img_contour_thresh with img as argument
    # img is modified to separate the foreground from the background
    # contours are extracted from the foreground
    # thresh is a thresholded image after morphological transformations 
    img, contours, thresh = get_img_contour_thresh(img)
    
    # Initialize three empty strings for prediction results
    ans1 = ''
    ans2 = ''
    ans3 = ''
    
    # Check if any contours are detected
    if len(contours) > 0:
        
        # Find the contour with max area
        # contour is a numpy array
        contour = max(contours, key=cv2.contourArea)
        
        # If the contour area is greater than 2500 pixels
        if cv2.contourArea(contour) > 2500:
            
            # Find the coordinates of the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Crop the thresholded image using the bounding rectangle coordinates
            # newImage is a numpy array
            newImage = thresh[y:y + h, x:x + w]
            
            # Resize newImage to 28x28 pixels
            newImage = cv2.resize(newImage, (28, 28))
            
            # Flatten newImage to a 1D array
            newImage = np.array(newImage)
            newImage = newImage.flatten()
            newImage = newImage.reshape(newImage.shape[0], 1)
            
            # Call predict function of Digit_Recognizer_LR object with w_LR and b_LR as arguments
            ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
            
            # Call predict_nn function of Digit_Recognizer_NN object with d2 and newImage as arguments
            ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
            
            # Call predict function of Digit_Recognizer_DL object with d3 and newImage as arguments
            ans3 = Digit_Recognizer_DL.predict(d3, newImage)

    # Define the coordinates and size of a rectangle for drawing a border around the video frame
    x, y, w, h = 0, 0, 300, 300
    
    # Draw a green rectangle on the video frame
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add text labels to the video frame with prediction results
    cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the video frame and thresholded image
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    
    # Wait for a key press for 10 milliseconds
    k = cv2.waitKey(10)
    
    # If the Esc key is pressed, break from the while loop
    if k == 27:
        break