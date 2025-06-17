import typing

import torch
import torch.nn as nn

from foundations.engine.engine import NeuralModelGenerator


class CharLevelLstmMultiClass(nn.Module):

    def __init__(self, input_size, num_classes, num_layers=1, hidden_size=128):
        super(CharLevelLstmMultiClass, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, device):
        h0 = torch.randn(self.num_layers, 1, self.hidden_size).to(device)
        c0 = torch.randn(self.num_layers, 1, self.hidden_size).to(device)
        return h0, c0

    def forward(self, x, h0, c0):
        out, hidden = self.rnn(x, (h0, c0))
        h0, c0 = hidden
        out = out[:, -1, :]
        out = self.softmax(self.fc(out))
        return out, h0, c0


class CharLevelLstmMultiClassGenerator(NeuralModelGenerator):

    @classmethod
    def generate(cls, hyperparameters):
        return CharLevelLstmMultiClass(
            hyperparameters['char_number'],
            hyperparameters['num_classes'],
            hyperparameters['num_layers'],
            hyperparameters['hidden_size']
        )
