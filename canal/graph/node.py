import abc
import dataclasses
import enum
from ordered_set import OrderedSet
from typing import Iterator, List


@enum.unique
class NodeType(enum.Enum):
    SWITCH_BOX = enum.auto()
    PORT = enum.auto()
    REGISTER = enum.auto()
    GENERIC = enum.auto()


@dataclasses.dataclass
class Node(abc.ABC):
    x: int
    y: int
    width: int

    def __post_init__(self):
        self._neighbors = OrderedSet()
        self._conn_ins = []
        self._edge_cost = {}

    def add_edge(self, node: "Node", delay: int=0, force_connect: bool=False):
        if not force_connect:
            assert self.width == node.width
        if node not in self._neighbors:
            self._neighbors.add(node)
            node._conn_ins.append(self)
            self._edge_cost[node] = delay

    def remove_edge(self, node: "Node"):
        if node in self._neighbors:
            self._edge_cost.pop(node)
            self._neighbors.remove(node)
            node._conn_ins.remove(self) # remove incoming connections as well

    def get_edge_cost(self, node: "Node") -> int:
        if node not in self._edge_cost:
            return MAX_DEFAULT_DELAY
        return self._edge_cost[node]

    def get_conn_in(self) -> List["Node"]:
        return self._conn_ins

    def __iter__(self) -> Iterator["Node"]:
        return iter(self._neighbors)

    def __len__(self):
        return len(self._neighbors)

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def node_str(self):
        raise NotImplementedError()

    def clear(self):
        self._neighbors.clear()
        self._edge_cost.clear()
        self._conn_ins.clear()

    def __contains__(self, item):
        return item in self._neighbors

    def __hash__(self):
        return hash(self.width) ^ hash(self.x) ^ hash(self.y)
