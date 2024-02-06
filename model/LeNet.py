import torch
import torch.nn as nn
import torch.nn.functional as F


"""
10개의 class를 분류하는 lenet 구현

input size :  32, 32, 1
output size : 10
"""

class LeNet(nn.Module):
    def __init__(self, num_class):
        super(LeNet, self).__init__()
        self.C1 = nn.Conv2d(1, 6, kernel_size=5)
        self.S2 = nn.AvgPool2d(2)
        self.C3 = nn.Conv2d(6, 16, kernel_size=5)
        self.S4 = nn.AvgPool2d(2)
        self.C5 = nn.Conv2d(16, 120, kernel_size=5)
        self.F6 = nn.Linear(120, 84)
        self.Out = nn.Linear(84, num_class)
            
    def forward(self, x):
        x = self.C1(x)
        x = F.sigmoid(self.S2(x))
        x = self.C3(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.F6(torch.squeeze(x))
        x = self.Out(x)
        
        return x


if __name__ =='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LeNet().to(device)

    input_tensor = torch.randn((1, 1, 32, 32)).to(device)
    output_tensor = model(input_tensor)
    print('LeNet')
    print("input tensor shape : ", input_tensor.shape)
    print("output tensor shape : ", output_tensor.shape)