import typing

from pandas import DataFrame
import os
import re

import torch
from torch.utils.data import Dataset, DataLoader
import unidecode
import pandas as pd
import numpy as np

from himeko.hbcm.elements.executable.edge import ExecutableHyperEdge
from himeko.hbcm.elements.vertex import HyperVertex

DATA, LABEL = 'data', 'label'

class LambdaDataframeDataset(Dataset):

    def __init__(self, dataframe: DataFrame, feature_encoding: typing.Callable, label_encoding: typing.Callable):
        self.sequences = dataframe[DATA].apply(feature_encoding).tolist()
        self.labels = dataframe[LABEL].apply(label_encoding).tolist()

    def __getitem__(self, i):
        return self.sequences[i], self.labels[i]

    def __len__(self):
        return len(self.sequences)


class DatasetNode(HyperVertex):

    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str,
                 parent: typing.Optional = None):
        super().__init__(name, timestamp, serial, guid, suid, label, parent)
        self._train_dataset: Dataset|None = None
        self._test_dataset: Dataset|None = None
        self._train_dataloader: DataLoader|None = None
        self._test_dataloader: DataLoader|None = None
        # DF train, test
        self._df_train: DataFrame|None = None
        self._df_test: DataFrame|None = None

    def _create_dataset(self):
        raise NotImplementedError

    def _create_dataloader(self, batch_size=1, shuffle=True):
        if self._train_dataset is not None and self._test_dataset is not None:
            self._train_dataloader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            raise ReferenceError("Train dataloader or test dataset not loaded")


    def load_data(self, df: DataFrame, random_state=None):
        """
        Inputs the raw dataset and partitions to train, validation and test datasets
        :param random_state: random seed
        :param df:
        :return:
        """
        df_grouped = df.groupby(LABEL)
        if random_state is None:
            self._df_train = df_grouped.sample(frac=0.8)
        else:
            self._df_train = df_grouped.sample(frac=0.8, random_state=random_state)
        df_train_idx_set = set(self._df_train.index)
        self._df_test = df.loc[[idx for idx in df.index if idx not in df_train_idx_set], :]
        self._df_train.reset_index(drop=True, inplace=True)
        self._df_test.reset_index(drop=True, inplace=True)
        self._create_dataset()

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def test_dataset(self):
        return self._test_dataset


def dataset_load(data_dir):
    re_pattern = r"[+*/#&\'\"\x02]"
    data: typing.Dict = dict()
    lower_category = set()
    for f in os.listdir(data_dir):
        cat = f.replace('.txt', '')
        lower_category.add(cat)
        l = [unidecode.unidecode(
            re.sub(re_pattern, "", name.strip())) for name in open(data_dir + '/' + f, encoding='utf-8').readlines()
            if len(name) > 0
        ]
        if cat not in data:
            data[cat] = l
        else:
            data[cat].extend(l)

        # data[f.replace('.txt', '')] = [name.strip() for name in open(DATA_DIR / 'data' / f).readlines()]
    # Sort data, seems to be some kind of error
    data = dict(sorted(data.items()))
    DATA, LABEL = 'data', 'label'
    df0 = pd.DataFrame(data=[[name, cl] for cl in data.keys() for name in data[cl]], columns=[DATA, LABEL])
    df0 = df0.replace('', np.nan).dropna()
    return df0


class ClassLabelTransformer(ExecutableHyperEdge):

    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str,
                 parent: typing.Optional[HyperVertex]) -> None:
        super().__init__(name, timestamp, serial, guid, suid, label, parent)
        self.__labels = []
        self.__labels_mapping = dict()
        self.__cnt_label = 0

    def __update_labels(self):
        # If new relation has been added, reiterate on incoming labels
        if self.__cnt_label != self.cnt_in_relations:
            self.__labels = []
            self.__cnt_label = self.cnt_in_relations
            for e in self.in_relations():
                self.__labels.append(e.target.name)
            self.__labels = sorted(self.__labels)
            self.__labels_mapping = {v: k for k, v in enumerate(self.__labels)}

    def operate(self, label, one_hot=False, *args, **kwargs):
        self.__update_labels()
        if one_hot:
            out = torch.zeros(self.__cnt_label)
            out[self.__labels_mapping[label]] = 1
            return out
        else:
            return self.__labels_mapping[label]

    @property
    def labels(self):
        self.__update_labels()
        return [c for c in self.__labels]


