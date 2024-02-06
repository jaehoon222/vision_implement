import torch
import torch.nn as nn
import torch.nn.functional as F

"""
1000개의 class를 분류하는 alexnet 구현

input size :  227, 227, 3
output size : 1000
"""

class AlexNet(nn.Module):
    def __init__(self, num_class = 1000):
        super().__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, 2),
        )

        self.seq2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, 2)
        )

        self.seq3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
        )

        self.seq4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.fclayer = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_class)
        )
        
    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)
        x = x.view(x.size(0), -1)
        x = self.fclayer(x)

        return x


if __name__ =='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AlexNet().to(device)

    input_tensor = torch.randn((1, 3, 227, 227)).to(device)
    output_tensor = model(input_tensor)

    print('AlexNet')
    print("input tensor shape : ", input_tensor.shape)
    print("output tensor shape : ", output_tensor.shape)