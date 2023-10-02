import typing

from himeko_hypergraph.src.elements.vertex import HyperVertex
from himeko_hypergraph.src.factories.creation_elements import FactoryHypergraphElements
from himeko_neural_model.src.foundations.dataset.vocab_node import VocabNode


class FactoryVocabularyNode(FactoryHypergraphElements):

    @classmethod
    def create_vertex_default(cls, name: str, timestamp: int, parent: typing.Optional[HyperVertex] = None):
        label, serial, guid, suid = cls.create_default_attributes(name, timestamp, parent)
        v0 = VocabNode(name, timestamp, serial, guid, suid, label, parent)
        return v0
