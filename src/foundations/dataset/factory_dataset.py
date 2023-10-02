import typing

from himeko_hypergraph.src.elements.vertex import HyperVertex
from himeko_hypergraph.src.factories.creation_elements import FactoryHypergraphElements


class FactoryCharDatasetNode(FactoryHypergraphElements):

    @classmethod
    def create_vertex_default(cls, name: str, timestamp: int, parent: typing.Optional[HyperVertex] = None):
        return super().create_vertex_default(name, timestamp, parent)