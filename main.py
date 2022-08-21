import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from PIL import Image

import numpy as np
from torchvision import transforms


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x
    # pass

ORIGIN_MODEL = "best_model_origin.pth"
ESENTIAL_MODEL = "faceshape_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = torch.load(ORIGIN_MODEL, map_location=device)
# model.load_state_dict(model, strict=False)
#
model = EfficientNet.from_name('efficientnet-b5', num_classes=5)
model.load_state_dict(torch.load(ESENTIAL_MODEL, map_location=device), strict=False)
model.eval()

# print(model)


convert_tensor = transforms.ToTensor()

img = Image.open('data_set/Heart/heart (3).jpg').convert('RGB')
test_ds = convert_tensor(img)

# print(test_ds)
#
with torch.no_grad():
    model.eval()

    inputs =torch.FloatTensor(test_ds.unsqueeze(0))
    output = model(inputs)
    pred = output.argmax(dim=1, keepdim=True)
    print(pred)