import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class Inception(nn.Module):
    def __init__(self, in_channels, conv_1x1, conv_3x3_reduce, conv_3x3, conv_5x5_reduce, conv_5x5, pool):
        super(Inception, self).__init__()
        self.branch1 = ConvBlock(in_channels, conv_1x1, kernel_size = 1, stride = 1, padding = 0)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, conv_3x3_reduce, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(conv_3x3_reduce, conv_3x3, kernel_size= 3, stride = 1, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, conv_5x5_reduce, kernel_size = 1, stride = 1, padding = 0),
            ConvBlock(conv_5x5_reduce, conv_5x5, kernel_size= 5, stride = 1, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride = 1, padding = 1),
            ConvBlock(in_channels, pool, kernel_size= 1, stride = 1, padding=0)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes = 100, aux_logits = True):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1, ceil_mode=True),
            nn.Conv2d(64, 64, kernel_size = 1, stride = 1, padding = 0),
            nn.Conv2d(64, 192, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode=True),
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

    def forward(self, x):

        x = self.seq1(x)        

        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool3(x)
        x = self.a4(x)
        aux1: Optional[Tensor] = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        aux2: Optional[Tensor] = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.e4(x)
        x = self.maxpool4(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1) # x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = self.dropout(x)

        if self.aux_logits and self.training:
            return aux1, aux2
        else:
            return x
    
if __name__ =='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet(100).to(device)

    input_tensor = torch.randn((1, 3, 224, 224)).to(device)
    output_tensor = model(input_tensor)
    print('Googlenet')
    print("input tensor shape : ", input_tensor.shape)
    print("output tensor shape : ", output_tensor[1].shape)

