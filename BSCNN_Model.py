from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1_conv1 = nn.Conv2d(3, 64, 3)
        self.block1_conv2 = nn.Conv2d(64, 64, 3)
        self.block1_pool = nn.MaxPool2d(2, 2)

        self.block2_conv1 = nn.Conv2d(64, 128, 3)
        self.block2_conv2 = nn.Conv2d(128, 128, 3)
        self.block2_pool = nn.MaxPool2d(2, 2)

        self.block3_conv1 = nn.Conv2d(128, 256, 3)
        self.block3_conv2 = nn.Conv2d(256, 256, 3)
        self.block3_conv3 = nn.Conv2d(256, 256, 3)
        self.block3_pool = nn.MaxPool2d(2, 2)

        self.block4_conv1 = nn.Conv2d(256, 512, 3)
        self.block4_conv2 = nn.Conv2d(512, 512, 3)
        self.block4_conv3 = nn.Conv2d(512, 512, 3)
        self.block4_pool = nn.MaxPool2d(2, 2)

        self.block5_conv1 = nn.Conv2d(256, 512, 3)
        self.block5_conv2 = nn.Conv2d(512, 512, 3)
        self.block5_conv3 = nn.Conv2d(512, 1, 3)
        self.block5_pool = nn.MaxPool2d(2, 2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x