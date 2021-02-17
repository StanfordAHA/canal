import dataclasses

from canal.graph.node import Node
from canal.graph.switch_box import SwitchBoxSide


@dataclasses.dataclass
class RegisterMuxNode(Node):
    track: int
    side: SwitchBoxSide

    def __post_init__(self):
        self.name = f"{int(self.side.value)}_{self.track}"

    def node_str(self):
        return (f"RMUX ({self.track}, {self.x}, {self.y}, "
                f"{self.side.value}, {self.width})")

    def __repr__(self):
        return f"RMUX_T{self.track}_{self.side.name}_B{self.width}"

    def __hash__(self):
        return super().__hash__() ^ hash(self.track) ^ hash(self.side)
