import typing

from himeko_hypergraph.src.elements.vertex import ExecutableHyperVertex


class AggregationNode(ExecutableHyperVertex):

    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str,
                 parent: typing.Optional = None):
        super().__init__(name, timestamp, serial, guid, suid, label, parent)