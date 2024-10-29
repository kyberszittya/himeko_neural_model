import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.input_layer = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1, groups=1)
        self.pool_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1)
        self.conv_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1, groups=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1)
        self.conv_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1, groups=1)
        self.fc = nn.Linear(in_features=128 * 18 * 18, out_features=10)  # Adjusting in_features after manual calculation of size

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.pool_0(x)
        x = F.relu(self.conv_0(x))
        x = self.pool_1(x)
        x = F.relu(self.conv_1(x))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc(x)
        return x

# Instantiate the model and print its architecture
model = MNISTConvNet()
print(model)