import torch
from torch import nn
from torch.nn import functional as func


class Colorizer(nn.Module):
    def __init__(self, img_size):
        super(Colorizer, self).__init__()
        self.conv1 = nn.Conv2d(1,  3, 5)
        self.conv2 = nn.Conv2d(3,  6, 5)
        self.conv3 = nn.Conv2d(6, 12, 5)
        self.fc1   = nn.Linear(12*(img_size//8 - 4)**2, 1024)
        self.fc2   = nn.Linear(                   1024, 1024)
        self.fc3   = nn.Linear(                   1024, 3*img_size**2)

    def forward(self, x):
        img_size = x.shape[-1]

        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, kernel_size=2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, kernel_size=2)
        x = func.relu(self.conv3(x))
        x = func.max_pool2d(x, kernel_size=2)
        x = func.relu(self.fc1(x.view(-1, 12*(img_size//8 - 4)**2)))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)

        return x
