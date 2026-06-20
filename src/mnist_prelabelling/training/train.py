import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_prelabelling.models.custom_cnn import CustomCNN

my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="data", train=True, transform=my_transform, download=True)
test_dataset = datasets.MNIST(root="data", train=False, transform=my_transform, download=True)