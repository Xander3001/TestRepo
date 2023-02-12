
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

# import cv2
# import numpy as np
# import Digit_Recognizer_LR
# import Digit_Recognizer_NN
# 
# 
# def mnist_train():
#     l = np.load('weight_biases_LR.npy')
#     b = np.load('weight_biases_NN.npy')
#     return l, b
# 
# 
# def get_img_contour_thresh(img):
#     x, y, w, h = 0, 50, 300, 300
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (35, 35), 0)
#     ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     thresh1 = thresh1[y:y + h, x:x + w]
#     contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
#     return img, contours, thresh1
# 
# 
# w, b = mnist_train()
# d2, d3 = Digit_Recognizer_NN.load_model('MNIST_CNN_deeper.h5'), Digit_Recognizer_NN.load_model('MNIST_CNN_deeper.h5')
# cap = cv2.VideoCapture(0)
# cap.set(3, 1208)
# cap.set(4, 720)
# 
# 
#     while (cap.isOpened()):
#         # get coordinates of the rectangular area, to be displayed in the screen.
#         ret, img = cap.read()
#         # initialization of all the 3 learners, with '' by default.
#         img, contours, thresh = get_img_contour_thresh(img)
#         ans1 = ''
#         ans2 = ''
#         ans3 = ''
#         if len(contours) > 0:
#             contour = max(contours, key=cv2.contourArea)
#             if cv2.contourArea(contour) > 2500:
#                 # print(predict(w_from_model,b_from_model,contour))
#                 x, y, w, h = cv2.boundingRect(contour)
#                 # newImage = thresh[y - 15:y + h + 15, x - 15:x + w +15]
#                 newImage = thresh[y:y + h, x:x + w]
#                 newImage = cv2.resize(newImage, (28, 28))
#                 newImage = np.array(newImage)
#                 newImage = newImage.flatten()
#                 newImage = newImage.reshape(newImage.shape[0], 1)
#                 ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
#                 ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
#                 ans3 = Digit_Recognizer_DL.predict(d3, newImage)
# 
#         x, y, w, h = 0, 0, 300, 300
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.imshow("Frame", img)
#         cv2.imshow("Contours", thresh)
#         k = cv2.waitKey(10)
#         if k == 27:
#             break