import glob
import os

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


## 함수
# 디렉토리가 존재하지 않으면 생성
def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


# 경로 설정 변수
readErrorImageList = []
saveErrorImageList = []
noneFaceErrorImageList = []
labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
setTypes = ['testing', 'training']

# 얼굴부분 crop
# haarcascade 불러오기
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Main
for setType in setTypes:
    print("Processing " + setType)
    for label in labels:
        print("Processing " + label)
        IMAGE_DIR = './imageSet/' + setType + '_set/' + label + '/'
        grayAug_DIR = './imageSet/grayAug_' + setType + '_image/' + label + '/'
        makedirs(IMAGE_DIR)
        makedirs(grayAug_DIR)
        IMAGE_FILES = glob.glob(IMAGE_DIR + '*.jpg')

        # 표현되는 랜드마크의 굵기와 반경
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)

        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            for idx, file in enumerate(IMAGE_FILES):
                currentFileName = os.path.basename(file)

                # 이미지 불러오기
                image = cv2.imread(file)
                if image is None:
                    print("READ ERROR!!" + currentFileName)
                    readErrorImageList.append(currentFileName)
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # 얼굴 crop
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    print(faces)
                    color = (0, 0, 255)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cropped = gray[y: y + h, x: x + w]

                        equalized = cv2.equalizeHist(cropped)

                        # edge enhancement
                        kernel = np.array([[0, -1, 0],
                                           [-1, 5, -1],
                                           [0, -1, 0]])
                        image_sharp = cv2.filter2D(src=equalized, ddepth=-1, kernel=kernel)

                    writeReturn = cv2.imwrite(grayAug_DIR + currentFileName, image_sharp)
                    if writeReturn == True:
                        print("successfully saved image!!" + currentFileName)
                    else:
                        print("SAVE ERROR!!" + currentFileName)
                        saveErrorImageList.append(currentFileName)
                else:
                    print("SAVE ERROR!!" + currentFileName)
                    noneFaceErrorImageList.append(currentFileName)

        print("Done " + label)
    print("Done " + setType)

for idx, readErrorImage in enumerate(readErrorImageList):
    print(idx, "Read Error Image: ", readErrorImage)

for idx, saveErrorImage in enumerate(saveErrorImageList):
    print(idx, "Save Error Image: ", saveErrorImage)

for idx, noneFaceErrorImage in enumerate(noneFaceErrorImageList):
    print(idx, "None Face Error Image: ", noneFaceErrorImage)

