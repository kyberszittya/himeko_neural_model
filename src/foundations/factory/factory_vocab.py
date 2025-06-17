import typing

from foundations.dataset.vocab_node import VocabNode
from himeko.hbcm.elements.vertex import HyperVertex
from himeko.hbcm.factories.creation_elements import FactoryHypergraphElements


class FactoryVocabularyNode(FactoryHypergraphElements):

    @classmethod
    def create_vertex_default(cls, name: str, timestamp: int, parent: typing.Optional[HyperVertex] = None):
        label, serial, guid, suid = cls.create_default_attributes(name, timestamp, parent)
        v0 = VocabNode(name, timestamp, serial, guid, suid, label, parent)
        return v0
