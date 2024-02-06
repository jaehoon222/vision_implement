import torch
import torch.nn as nn
import torch.optim as optim
#cifar10, cifar100 test

def train_test(model, train_loader,test_loader,  epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
        print('[{}] train loss : {:.3f}'.format(epoch+1, running_loss / len(train_loader)) )


        model.eval()  # 모델을 평가 모드로 설정
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print('[{}] test loss : {:.3f}, test accuracy: {:.3f}%'.format(epoch + 1,
                                                                    test_loss / len(test_loader),
                                                                    100 * correct / total))

    return model