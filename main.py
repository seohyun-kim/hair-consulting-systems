import torch
from torch import nn


class CustomModel(nn.Module):
    # def __init__(self):
    #     super(CustomModel, self).__init__()
    #     self.layer = nn.Linear(2, 1)
    #
    # def forward(self, x):
    #     x = self.layer(x)
    #     return x
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("faceshape_model.pth", map_location=device)
print(model)

# with torch.no_grad():
#     model.eval()
#     inputs = torch.FloatTensor([[1 ** 2, 1], [5 **2, 5], [11**2, 11]]).to(device)
#     outputs = model(inputs)
#     print(outputs)