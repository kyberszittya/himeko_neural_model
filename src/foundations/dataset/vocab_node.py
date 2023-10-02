import torch

import numpy as np

import typing

from himeko_hypergraph.src.elements.vertex import HyperVertex


class VocabNode(HyperVertex):

    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str,
                 parent: typing.Optional = None):
        super().__init__(name, timestamp, serial, guid, suid, label, parent)
        self.__vocab = []
        self.__vocab_d = {}
        self.__vocab_inv = {}

    def load_vocabulary(self, path):
        self.__vocab = []
        with open(path, encoding='utf-8') as f:
            for l in f:
                self.__vocab.append(l[0])
        self.__vocab_d = {k: v for v,k in enumerate(self.__vocab)}
        self.__vocab_inv = {v: k for v,k in enumerate(self.__vocab)}



    # letters one hot encoding
    def ltr_one_hot(self, ltr):
        # num_cats: number of categories for the one hot encoding (usually len(vocab))
        def char_to_id(ch):
            return self.__vocab_d[ch]
        return torch.nn.functional.one_hot(torch.LongTensor([char_to_id(ltr)]), len(self.__vocab))

    # line one hot
    def line_to_one_hot_tensor(self, line):
        # num_cats: number of categories for the one hot encoding (usually len(vocab))
        line_one_hot = None

        for i, ltr in enumerate(line):
            if line_one_hot is None:
                line_one_hot = self.ltr_one_hot(ltr)
                # print(line_one_hot.shape)
            else:
                line_one_hot = torch.cat([line_one_hot, self.ltr_one_hot(ltr)])
        # return torch.unsqueeze(line_one_hot, 1)
        return line_one_hot.type(torch.FloatTensor)

    def one_hot_tensor_to_line(self, one_hot_tensor):
        line = ''
        for one_hot_ltr in one_hot_tensor:
            line += self.__vocab_inv[torch.argmax(one_hot_ltr).item()]
        return line

    @property
    def vocab(self):
        return self.__vocab_d.copy()

    @property
    def vocab_inv(self):
        return self.__vocab_inv.copy()

    @property
    def vocab_len(self):
        return len(self.__vocab)
