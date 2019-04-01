import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.block1_BN2d = nn.BatchNorm2d(64)
        self.block1_pool = nn.MaxPool2d(2, stride=2)

        self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.block2_BN2d = nn.BatchNorm2d(128)
        self.block2_pool = nn.MaxPool2d(2, stride=2)

        self.block3_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.block3_BN2d = nn.BatchNorm2d(256)
        self.block3_pool = nn.MaxPool2d(2, stride=2)

        self.block4_conv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, 3, padding=1)
        self.block4_BN2d = nn.BatchNorm2d(512)
        self.block4_pool = nn.MaxPool2d(2, stride=2)

        self.block5_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.block5_conv3 = nn.Conv2d(256, 1, 3, padding=1)
        self.block5_BN2d = nn.BatchNorm2d(1)
        self.block5_pool = nn.MaxPool2d(2, stride=2)


    def forward(self, x):
        # block1
        x = F.relu(self.block1_conv1(x))
        x = self.block1_pool(F.relu(self.block1_BN2d(self.block1_conv2(x))))

        # block2
        x = F.relu(self.block2_conv1(x))
        x = self.block2_pool(F.relu(self.block2_BN2d(self.block2_conv2(x))))

        # block3
        x = F.relu(self.block3_conv1(x))
        x = F.relu(self.block3_conv2(x))
        x = self.block3_pool(F.relu(self.block3_BN2d(self.block3_conv2(x))))

        # block4
        x = F.relu(self.block4_conv1(x))
        x = F.relu(self.block4_conv2(x))
        x = self.block4_pool(F.relu(self.block4_BN2d(self.block4_conv3(x))))

        # block5
        x = F.relu(self.block5_conv1(x))
        x = F.relu(self.block5_conv2(x))
        x = self.block5_pool(torch.sigmoid(self.block5_BN2d(self.block5_conv3(x))))

        return x
