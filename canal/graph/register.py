import dataclasses

from canal.graph.node import Node


@dataclasses.dataclass
class RegisterNode(Node):
    name: str
    track: int

    def node_str(self):
        return (f"REG {self.name} ({self.track}, {self.x},"
                f" {self.y}, {self.width})")

    def __repr__(self):
        return f"REG_{self.name}_B{self.width}"

    def __hash__(self):
        return super().__hash__() ^ hash(self.name) ^ hash(self.track)
