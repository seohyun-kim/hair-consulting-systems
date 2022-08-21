import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms


ORIGIN_MODEL = "best_model_origin.pth"
ESENTIAL_MODEL = "faceshape_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EfficientNet.from_name('efficientnet-b5', num_classes=5)
model.load_state_dict(torch.load(ESENTIAL_MODEL, map_location=device), strict=False)
model.eval()
# print(model)

convert_tensor = transforms.ToTensor()

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
target_img = IMG_HEART


processed_img = convert_tensor(Image.open(target_img).convert('RGB'))

with torch.no_grad():
    model.eval()

    inputs =torch.FloatTensor(processed_img.unsqueeze(0))
    output = model(inputs)
    print(output)
    pred = output.argmax(dim=1, keepdim=True)
    print(shape_class[int(pred)])
