import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.CIFAR100(root='./cifar100', train=True, download=True, transform=None)
test_dataset = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=None)