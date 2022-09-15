# import cv2
# import numpy as np
#
# # 얼굴부분 crop
# # haarcascade 불러오기
# from matplotlib import pyplot as plt
#
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # 이미지 불러오기
# img = cv2.imread('./imageSet/training_set/Heart/heart (2).jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 얼굴 찾기
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     cropped = img[y: y + h, x: x + w]
#     resize = cv2.resize(cropped, (200, 200))
#
#     img_yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
#
#     # equalize the histogram of the Y channel
#     img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
#
#     # convert the YUV image back to RGB format
#     img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#
#     # edge enhancement
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     image_sharp_c = cv2.filter2D(src=img_output, ddepth=-1, kernel=kernel)
#
#     # 이미지 저장하기
#     cv2.imwrite("./imageSet/croptest.jpg", resize)
#
#     # cv2.imshow("crop&resize", resize)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     fig = plt.figure()
#
#     ax1 = fig.add_subplot(2, 2, 1)
#     ax1.imshow(img)
#     ax1.set_title('img')
#     ax1.axis("off")
#
#     ax2 = fig.add_subplot(2, 2, 2)
#     ax2.imshow(cropped)
#     ax2.set_title('cropped')
#     ax2.axis("off")
#
#     ax1 = fig.add_subplot(2, 2, 3)
#     ax1.imshow(image_sharp_c)
#     ax1.set_title('image_sharp')
#     ax1.axis("off")
#
#     ax2 = fig.add_subplot(2, 2, 4)
#     ax2.imshow(img_output)
#     ax2.set_title('equalized')
#     ax2.axis("off")