import glob
import os
from PIL import Image, ImageOps

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

# Main
for setType in setTypes:
    for label in labels:
        print("Processing " + setType + " " + label)
        colorAug_DIR = './imageSet/colorAug/colorAug_' + setType + '_image/' + label + '/'
        color_poster_DIR = './imageSet/color_poster_' + setType + '_image/' + label + '/'
        makedirs(color_poster_DIR)
        IMAGE_FILES = glob.glob(colorAug_DIR + '*.jpg')

        # 표현되는 랜드마크의 굵기와 반경
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)

        for idx, file in enumerate(IMAGE_FILES):
            currentFileName = os.path.basename(file)

            # 이미지 불러오기
            image = Image.open(file)
            if image is None:
                print("READ ERROR!!" + currentFileName)
                readErrorImageList.append(currentFileName)
                continue

            # Image posterize
            image2 = ImageOps.posterize(image, 2)
            image2.save(color_poster_DIR + currentFileName)

        print("Done " + label)
    print("Done " + setType)