import torchvision
import torchvision.transforms as transforms
import torch

def dataset(size, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(), # PIL 이미지나 numpy 배열을 PyTorch Tensor로 변환
        transforms.Resize((size, size), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 이미지를 정규화
    ])

    # CIFAR-100 데이터셋 다운로드 및 불러오기
    train_dataset = torchvision.datasets.CIFAR100(root='./cifar100', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=False, transform=transform)

    # DataLoader를 사용하여 데이터를 미니배치 형태로 로드
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader