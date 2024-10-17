import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist_mlp_network(nn.Module):

    def __init__(self):
        super(Mnist_mlp_network, self).__init__()
        self.input_layer = nn.Linear(784, 128)
        self.hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x
