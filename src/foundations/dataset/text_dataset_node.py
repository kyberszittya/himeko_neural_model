import typing

import torch

from himeko_neural_model.src.foundations.dataset.dataset_node import DatasetNode, LambdaDataframeDataset
from himeko_neural_model.src.foundations.dataset.vocab_node import VocabNode


class CharDatasetNode(DatasetNode):

    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str,
                 parent: typing.Optional = None):
        super().__init__(name, timestamp, serial, guid, suid, label, parent)
        self._vocab: VocabNode|None = None
        self._classes: typing.Iterable|None = None
        # Mapping between label and class
        self.__map_class_label = dict()
        self.__map_label_class = dict()


    def load_classes(self, classes):
        """
        Load mappings between labels and integer IDs
        :param classes: sorted list of classes
        """
        self._classes = [c for c in classes]
        self.__map_class_label = {v: k for k,v in enumerate(self._classes)}
        self.__map_label_class = {k: v for k,v in enumerate(self._classes)}

    def __lbl_to_id(self, lbl: str):
        return self.__map_label_class[lbl]

    def __id_to_label(self, id: int):
        return self.__map_class_label[id]

    def _create_dataset(self):
        self.__collect_vocab()
        self._train_dataset = LambdaDataframeDataset(
            self._df_train, self._vocab.line_to_one_hot_tensor, self.__lbl_to_id
        )
        self._test_dataset = LambdaDataframeDataset(
            self._df_test, self._vocab.line_to_one_hot_tensor, self.__lbl_to_id
        )

    def __collect_vocab(self):
        self._vocab = list(self.get_subelements(lambda x: isinstance(x, VocabNode)))
