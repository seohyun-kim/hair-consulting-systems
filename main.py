import cv2
import numpy as np
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ORIGIN_MODEL = "best_model_origin.pth"
ESENTIAL_MODEL = "faceshape_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EfficientNet.from_name('efficientnet-b5', num_classes=5)
model.load_state_dict(torch.load(ESENTIAL_MODEL, map_location=device), strict=False)
# model = torch.load(ESENTIAL_MODEL)
model.eval() # eval 모드로 설정
# print(model)

shape_class ={0: "heart", 1: "oblong", 2: "oval", 3: "round", 4: "square"}

IMG_HEART = 'data_set/Heart/heart (3).jpg' # 0 (맞음)
IMG_OBLONG = 'data_set/Oblong/oblong (6).jpg' # 4 (틀림)
IMG_OVAL = 'data_set/Oval/Oval (2).jpg' # 3 (맞음)
IMG_ROUND = 'data_set/Round/round (1).jpg' # 2 (틀림)
IMG_SQUARE = 'data_set/Square/square (8).jpg' # 1 (틀림)
IMG_YJ = 'data_set/yj.jpg' # 1 (유진이)
IMG_EB = 'data_set/eb2.jpg' # 3 (박은빈)
IMG_KHD = 'data_set/khd.jpg' # 2 Oblong (강호동)

# 여기에 타겟 이미지 작성
target_img = cv2.imread(IMG_YJ)

# 이미지 전처리
gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) # gray scale
faces = face_cascade.detectMultiScale(gray, 1.3, 5) # 얼굴 찾기
for (x, y, w, h) in faces:
    cv2.rectangle(target_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cropped = target_img[y: y + h, x: x + w]
    resized = cv2.resize(cropped, (200, 200))

    img_yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # edge enhancement
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    pre_processed_img = cv2.filter2D(src=img_output, ddepth=-1, kernel=kernel)

# 텐서화
convert_tensor = transforms.ToTensor()

processed_img = convert_tensor(pre_processed_img)

with torch.no_grad():
    model.eval()

    inputs =torch.FloatTensor(processed_img.unsqueeze(0))
    output = model(inputs)
    print(output)
    pred = output.argmax(dim=1, keepdim=True)
    print(shape_class[int(pred)])
