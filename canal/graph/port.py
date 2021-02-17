import dataclasses

from canal.graph.node import Node


@dataclasses.dataclass
class PortNode(Node):
    name: str

    def node_str(self):
        return f"PORT {self.name} ({self.x}, {self.y}, {self.width})"

    def __repr__(self):
        return f"CB_{self.name}"

    def __hash__(self):
        return super().__hash__() ^ hash(self.name)
