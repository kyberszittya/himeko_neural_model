import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlpminimal(nn.Module):

    def __init__(self):
        super(Mlpminimal, self).__init__()
        self.input_layer = nn.Linear(4, 16)
        self.hidden_layer = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 3)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x
