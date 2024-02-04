import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self, image_size):
        super(SimpleCNN, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=128 * (self.image_size//4) * (self.image_size//4), out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Output: [batch_size, 64, image_size/2, image_size/2]
        x = self.pool(self.relu(self.conv2(x)))  # Output: [batch_size, 128, image_size/4, image_size/4]
        x = x.view(-1, 128 * (self.image_size//4) * (self.image_size//4))  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
