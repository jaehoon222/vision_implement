import torch
import torch.nn as nn
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2),
            nn.MaxPool2d(3, 2),
            
        )

    def forward(self, x):
        x = self.seq1(x)
        return x
    
if __name__ =='__main__':
    model = GoogLeNet()

    x = torch.rand((1,3, 224, 224))
    output = model(x)
    print(output.shape)