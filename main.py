import cv2
import numpy as np
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms
import torchvision.models as models


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ORIGIN_MODEL = "best_model_origin.pth"
ESENTIAL_MODEL = "./faceshape_model.pth"
ESENTIAL_MODEL_GRAY = "./faceshape_model_gray.pth"


# device = "cuda" if torch.cuda.is_available() else "cpu"

# #model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=5)
# model = EfficientNet.from_pretrained("efficientnet-b5", in_channels=3, advprop=True)
# model.load_state_dict(torch.load(ESENTIAL_MODEL, map_location=device), strict=False)
#     # strict : 엄격하게 적용할 지 여부


class EffNet(nn.Module):
    def __init__(self, num_classes=5):
        super(EffNet, self).__init__()
        self.eff = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes, in_channels=1)
    def forward(self, x):
        x = self.eff(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EffNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load(ESENTIAL_MODEL, map_location=device), strict=True)

model.eval()  # eval 모드로 설정

# for param in model.parameters():
#     print(param)

shape_class = {0: "heart", 1: "oblong", 2: "oval", 3: "round", 4: "square"}

IMG_HEART = 'data_set/Heart/heart (3).jpg'
IMG_OBLONG = 'data_set/Oblong/oblong (6).jpg'
IMG_OVAL = 'data_set/Oval/Oval (2).jpg'
IMG_ROUND = 'data_set/Round/round (1).jpg'
IMG_SQUARE = 'data_set/Square/square (8).jpg'
IMG_YJ = 'data_set/Round/round_yj.jpg'
IMG_EB = 'data_set/Oblong/oblong_eb.jpg'
IMG_KHD = 'data_set/Square/square_khd.jpg'
ING_JB = 'data_set/Oval/oval_jb.jpg'

# 여기에 타겟 이미지 작성
target_img = cv2.imread(IMG_YJ)

# 이미지 전처리
gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)  # gray scale

faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 얼굴 찾기
for (x, y, w, h) in faces:
    # cv2.rectangle(target_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cropped = target_img[y: y + h, x: x + w]
    # resized = cv2.resize(cropped, (200, 200))

    cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cropped = gray[y: y + h, x: x + w]
    equalized = cv2.equalizeHist(cropped)
    # img_yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
    # img_yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # edge enhancement
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    pre_processed_img = cv2.filter2D(src=equalized, ddepth=-1, kernel=kernel)
    # pre_processed_img = cv2.cvtColor(pre_processed_img, cv2.COLOR_BGR2GRAY)

# 텐서화
convert_tensor = transforms.ToTensor()

processed_img = convert_tensor(pre_processed_img)
# processed_img = convert_tensor(target_img)

with torch.no_grad():
    inputs = torch.FloatTensor(processed_img.unsqueeze(0))
    output = model(inputs)

    print(output)
    pred_output = output.argmax(dim=1, keepdim=True)
    print(pred_output)
   # print(shape_class[int(pred)])
