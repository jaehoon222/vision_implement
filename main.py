#여기서 모든놈들 꺼내오고 실행해서 train, test하는곳
from dataset import dataset
from train_test import train_test

from model.AlexNet import AlexNet
from model.LeNet import LeNet
from model.GoogLeNet import GoogLeNet
from model.VGGNet_16 import VGGNet_16
from model.GoogLeNet import GoogLeNet
import torch




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #choose model
    # model = AlexNet(100).to(device)
    #model = VGGNet_16(100).to(device)
    #googlenet 은 구현만
    model = GoogLeNet(100).to(device)
    
    #choose hyperparameter
    image_size = 224
    batch_size = 64
    epochs = 5
    lr = 0.001


    train_loader, test_loader = dataset(image_size, batch_size)
    train_model = train_test(model, train_loader, test_loader, epochs, lr)

    