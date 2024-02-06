#여기서 모든놈들 꺼내오고 실행해서 train, test하는곳
from dataset import dataset
from train_test import train_test

from model.AlexNet import AlexNet
from model.LeNet import LeNet
import torch




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #choose model
    model = AlexNet(100).to(device)

    #coose hyperparameter
    image_size = 227
    batch_size = 64
    epochs = 5
    lr = 0.0001


    train_loader, test_loader = dataset(image_size, batch_size)
    train_model = train_test(model, train_loader, test_loader, epochs, lr)

    