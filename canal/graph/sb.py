import dataclasses
import enum

from canal.graph.node import Node


@enum.unique
class SwitchBoxSide(enum.Enum):
    """
    Enum which represents the various sides:

       3
      ---
    2 | | 0
      ---
       1
    """
    NORTH = 3
    SOUTH = 1
    EAST = 0
    WEST = 2

    def get_opposite_side(self) -> "SwitchBoxSide":
        side = self
        if side == SwitchBoxSide.NORTH:
            return SwitchBoxSide.SOUTH
        elif side == SwitchBoxSide.SOUTH:
            return SwitchBoxSide.NORTH
        elif side == SwitchBoxSide.EAST:
            return SwitchBoxSide.WEST
        elif side == SwitchBoxSide.WEST:
            return SwitchBoxSide.EAST
        raise ValueError("Unknown value", side)


class SwitchBoxIO(enum.Enum):
    SB_IN = 0
    SB_OUT = 1


@dataclasses.dataclass(frozen=True)
class SwitchBoxConnectionType:
    side: SwitchBoxSide
    track: int
    io: SwitchBoxIO


@dataclasses.dataclass
class SwitchBoxNode(Node):
    track: int
    side: SwitchBoxSide
    io: SwitchBoxIO

    def node_str(self):
        return (f"SB ({self.track}, {self.x}, {self.y}, "
                f"{self.side.value}, {self.io.value}, {self.width})")

    def __repr__(self):
        return f"SB_T{self.track}_{self.side.name}_{self.io.name}_B{self.width}"

    def __hash__(self):
        return (super().__hash__() ^
                hash(self.track) ^ hash(self.side) ^ hash(self.io))
