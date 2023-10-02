from datetime import datetime
import abc
import typing

import torch
import torch.nn as nn

from himeko_hypergraph.src.elements.attribute import HypergraphAttribute
from himeko_hypergraph.src.elements.edge import HyperEdge, ExecutableHyperEdge
from himeko_hypergraph.src.elements.element import HypergraphElement
from himeko_hypergraph.src.elements.vertex import HyperVertex
from himeko_neural_model.src.foundations.dataset.dataset_node import DatasetNode


class HyperParameterNode(HyperVertex):

    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str,
                 parent: typing.Optional = None):
        super().__init__(name, timestamp, serial, guid, suid, label, parent)

    @classmethod
    def check_hypergraph_attribute(cls, x: HypergraphElement, attributes: typing.Iterable[str]):
        if attributes is None:
            return isinstance(x, HypergraphAttribute)
        return isinstance(x, HypergraphAttribute) and x.name in attributes

    def get_hyperparameters(self, attribute_names: typing.Iterable[str]):
        return self.get_subelements(lambda x: self.check_hypergraph_attribute(x, attribute_names))


class HyperParameterEdge(ExecutableHyperEdge):

    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str,
                 parent: typing.Optional[HyperVertex]) -> None:
        super().__init__(name, timestamp, serial, guid, suid, label, parent)

    def collect_parameters(self, attribute_names):
        for e in filter(lambda x: isinstance(x, HyperParameterNode), self.in_relations()):
            yield e.target.get_hyperparameters(attribute_names)

    def collect_engines(self):
        for e in filter(lambda x: isinstance(x, AbstractInferenceNeuralEngine), self.out_relations()):
            yield e.target

    def operate(self, attribute_names: typing.Optional[typing.Iterable[str]] = None):
        hyperparameters = {}
        # Collect hyperparameters
        for p in self.collect_parameters(attribute_names):
            hyperparameters[p.name] = p.value
        # Propagate to abstract inference engines
        for e in self.collect_engines():
            e.update_hyperparameters(hyperparameters)


class NeuralModelGenerator():

    def generate(self, hyperparameters):
        raise NotImplementedError


class AbstractInferenceNeuralEngine(HyperVertex, abc.ABC):

    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str,
                 neural_model_generator: NeuralModelGenerator, parent: typing.Optional = None):
        super().__init__(name, timestamp, serial, guid, suid, label, parent)
        # Set neural model
        self._neural_model_generator = neural_model_generator
        self._neural_model: typing.Optional[nn.Module] = None
        self._hyperparameters: typing.Dict = {}

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test(self):
        raise NotImplementedError

    @abc.abstractmethod
    def online_learning(self):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def reflexive_train(self):
        raise NotImplementedError

    def save_weights(self):
        if self.__neural_model is not None:
            now = datetime.now()
            torch.save(
                self.__neural_model.state_dict(),
                f"{self.name}_{self.timestamp}__{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.pth"
            )

    def load_weights(self):
        self.recreate_synapse()
        self.__neural_model.load_state_dict(torch.load(self.__hyperparameters['weight_path']))

    def update_parameters(self, parameters):
        self.__hyperparameters = parameters

    def recreate_synapse(self):
        self.__neural_model = self.__neural_model_generator.generate(self.__hyperparameters)


class PredictionHyperEdge(ExecutableHyperEdge):

    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str,
                 parent: typing.Optional[HyperVertex]) -> None:
        super().__init__(name, timestamp, serial, guid, suid, label, parent)

    def operate(self, vector):
        for o in filter(lambda x: isinstance(x, AbstractInferenceNeuralEngine), self.out_vertices()):
            o: AbstractInferenceNeuralEngine
            pred, _, _ = o.predict(vector)
            yield pred.detach()


class TrainHyperEdge(ExecutableHyperEdge):

    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str,
                 parent: typing.Optional[HyperVertex]) -> None:
        super().__init__(name, timestamp, serial, guid, suid, label, parent)

    def operate(self, epochs):
        datanodes = []
        for _datanode in filter(lambda x: isinstance(DatasetNode, x), self.in_relations()):
            datanodes.append(_datanode)
        """
        for epoch in range(epochs):
            epoch_train_loss, epoch_validation_loss = 0.0
            for train_datanode in datanodes:

            #for name_tensor, label in trai
        """


