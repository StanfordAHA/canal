import abc
import dataclasses
import enum
from typing import Dict, List, Optional, Tuple, Union

from canal.graph.node import Node
from canal.graph.port import PortNode
from canal.graph.sb import SwitchBoxSide, SwitchBoxIO, SwitchBoxConnectionType
from canal.graph.sb_container import SwitchBox
from canal.graph.tile import Tile


class InterconnectPolicy(enum.Enum):
    PASS_THROUGH = enum.auto()
    IGNORE = enum.auto()


class InterconnectCore(abc.ABC):
    @abc.abstractmethod
    def inputs(self) -> List[Tuple[int, str]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def outputs(self) -> List[Tuple[int, str]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_port_ref(self, port_name: str):
        raise NotImplementedError()

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()


@dataclasses.dataclass
class InterconnectGraph:
    bit_width: int

    def __post_init__(self):
        self._tiles: Dict[Tuple[int, int], Tile] = {}
        self._switch_ids: Dict[int, SwitchBox] = {}
        # This is a 2D grid designed to support fast queries with irregular tile
        # height.
        self._tile_grid: List[List[Optional[Tile]]] = []

    def add_tile(self, tile: Tile):
        tile.switchbox.id = self._assign_id(tile.switchbox)
        x = tile.x
        y = tile.y
        self._tiles[(x, y)] = tile
        height = tile.height
        # Automatically scale the grid.
        while len(self._tile_grid) < y + height:
            self._tile_grid.append([])
        for row in range(len(self._tile_grid)):
            while len(self._tile_grid[row]) <= x:
                self._tile_grid[row].append(None)
        # Store indices and check for correctness.
        self._assign_tile_grid(x, y, tile)
        for i in range(y + 1, y + height):
            # Add reference to that tile.
            self._assign_tile_grid(x, i, tile)

    def _assign_tile_grid(self, x: int, y: int, tile: Tile) -> None:
        self._check_grid(x, y)
        self._tile_grid[y][x] = tile

    def _check_grid(self, x: int, y: int) -> None:
        tile = self._tile_grid[y][x]
        if tile is None:
            return
        raise RuntimeError(f"{str(tile)} already exists")

    def _assign_id(self, switch: SwitchBox) -> int:
        for switch_id, s in self._switch_ids.items():
            if switch == s:
                return switch_id
        switch_id = len(self._switch_ids)
        self._switch_ids[switch_id] = switch
        return switch_id

    def get_tile(self, x: int, y: int) -> Optional[Tile]:
        width, height = self.get_size()
        if x >= width or y >= height:
            return None
        return self._tile_grid[y][x]

    def has_empty_tile(self) -> bool:
        for y in range(len(self._tile_grid)):
            for x in range(len(self._tile_grid[y])):
                if self._tile_grid[y][x] is None:
                    return True
        return False

    def is_original_tile(self, x: int, y: int):
        tile = self.get_tile(x, y)
        return tile is not None and tile.x == x and tile.y == y

    def get_size(self) -> Tuple[int, int]:
        height = len(self._tile_grid)
        width = len(self._tile_grid[0])
        return width, height

    def set_core_connection(self, x: int, y: int, port_name: str,
                            connection_type: List[SwitchBoxConnectionType]):
        tile = self.get_tile(x, y)
        if tile is None:  # do nothing if emtpy
            return
        tile.set_core_connection(port_name, connection_type)

    def set_core_connection_all(self, port_name: str,
                                connection_type: List[Tuple[SwitchBoxSide,
                                                            SwitchBoxIO]]):
        """Helper function to set connections for all the tiles with the
        same port_name
        """
        for (x, y), tile in self._tiles.items():
            # Construct the connection types.
            switch = tile.switchbox
            num_track = switch.num_track
            connections: List[SwitchBoxConnectionType] = []
            for track in range(num_track):
                for side, io in connection_type:
                    connections.append(SwitchBoxConnectionType(side, track, io))
            self.set_core_connection(x, y, port_name, connections)

    def set_inter_core_connection(self, from_name: str, to_name: str):
        for tile in self._tiles.values():
            from_node: PortNode = tile.get_port(from_name)
            to_node: PortNode = tile.get_port(to_name)
            if from_node is not None and to_node is not None:
                from_node.add_edge(to_node)

    def set_core(self, x: int, y: int, core: InterconnectCore):
        tile = self.get_tile(x, y)
        tile.set_core(core)

    def remove_tile(self, coord: Tuple[int, int]):
        if coord in self._tiles:
            self._tiles.pop(coord)

    def get_sb(self, x: int, y: int, side: SwitchBoxSide, track: int,
               io: SwitchBoxIO):
        tile = self.get_tile(x, y)
        if tile is None:
            return None
        return tile.get_sb(side, track, io)

    def get_port(self, x: int, y: int, port_name: str) -> Optional[PortNode]:
        tile = self.get_tile(x, y)
        if tile is not None:
            return tile.get_port(port_name)
        return None

    def __getitem__(self, item: Tuple[int, int]):
        return self._tiles[item]

    def __contains__(self, item: Union[Tile, SwitchBox, Node]) -> bool:
        if isinstance(item, Tile):
            return self.get_tile(item.x, item.y) == item
        tile = self.get_tile(item.x, item.y)
        if tile is None:
            return False
        if isinstance(item, SwitchBox):
            return tile.switchbox == item
        if isinstance(item, PortNode):
            return tile.ports[item.name] == item
        if isinstance(item, RegisterNode):
            return tile.switchbox.registers[item.name] == item
        if isinstance(item, SwitchBoxNode):
            return tile.get_sb(item.side, item.track, item.io) == item
        return False

    def connect_switchbox(self, x0: int, y0: int, x1: int, y1: int,
                          expected_length: int, track: int,
                          policy: InterconnectPolicy):
        """Connect switches with expected length in the region
        (x0, y0) <-> (x1, y1), inclusively. It will try to connect everything
        with expected length. Connect in left -> right & top -> bottom ordering.

        policy:
            Used when there is a tile with height larger than 1.

            PASS_THROUGH: Allow to connect even if the wire length is different
                          from the expected_length. This will introduce
                          uncertainties of total wires and may introduce bugs.
                          One remedy for that is to break the tiles into smaller
                          tiles and assign switch box for each smaller tiles.

            IGNORE: Ignore the connection if the wire length is different from
                    the expected_length. It is safe but may leave some tiles
                    unconnected.
        """
        if (x1 - x0 - 1) % expected_length != 0:
            raise ValueError("the region x has to be divisible by "
                             "expected_length")
        if (y1 - y0 - 1) % expected_length != 0:
            raise ValueError("the region y has to be divisible by "
                             "expected_length")

        # NOTE(keyi): this code is very complex and hence has many
        # comments. please do not simplify this code unless you fully understand
        # the logic flow.

        # Left to right first.
        for x in range(x0, x1 - expected_length + 1, expected_length):
            for y in range(y0, y1 + 1, expected_length):
                if not self.is_original_tile(x, y):
                    continue
                tile_from = self.get_tile(x, y)
                tile_to = self.get_tile(x + expected_length, y)
                # Several outcomes to consider:
                #   1. tile_to is empty -> apply policy
                #   2. tile_to is a reference -> apply policy
                if not self.is_original_tile(x + expected_length, y):
                    if policy == InterconnectPolicy.IGNORE:
                        continue
                    # Find another tile longer than expected length that's
                    # within the range. Because, at this point we already know
                    # that the policy is passing through, just search the
                    # nearest tile (not tile reference) to meet the pass through
                    # requirement.
                    x_ = x + expected_length
                    while x_ < x1:
                        if self.is_original_tile(x_, y):
                            tile_to = self._tile_grid[y][x_]
                            break
                        x_ += 1
                    # Check again if we have resolved this issue. Since it's
                    # best effort, we will ignore if there's no tile left to
                    # connect.
                    if not self.is_original_tile(x_, y):
                        continue

                assert tile_to.y == tile_from.y
                # Add to connection list:
                # Forward:
                self._add_sb_connection(
                    tile_from, tile_to, track, SwitchBoxSide.EAST)
                # Backward:
                self._add_sb_connection(
                    tile_to, tile_from, track, SwitchBoxSide.WEST)

        # Top to bottom (very similar to the previous one (left to right)).
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 - expected_length + 1, expected_length):
                if not self.is_original_tile(x, y):
                    continue
                tile_from = self.get_tile(x, y)
                tile_to = self.get_tile(x, y + expected_length)
                # Several outcomes to consider:
                #   1. tile_to is empty -> apply policy
                #   2. tile_to is a reference -> apply policy
                if not self.is_original_tile(x, y + expected_length):
                    if policy == InterconnectPolicy.IGNORE:
                        continue
                    y_ = y + expected_length
                    while y_ < y1:
                        if self.is_original_tile(x, y_):
                            tile_to = self._tile_grid[y_][x]
                            break
                        y_ += 1
                    # Check again if we have resolved this issue. Since it's
                    # best effort, we will ignore if there's no tile left to
                    # connect.
                    if not self.is_original_tile(x, y_):
                        continue

                assert tile_to.x == tile_from.x
                # Add to connection list:
                # Forward:
                self._add_sb_connection(
                    tile_from, tile_to, track, SwitchBoxSide.SOUTH)
                # Backward:
                self._add_sb_connection(
                    tile_to, tile_from, track, SwitchBoxSide.NORTH)

    def _add_sb_connection(self, tile_from: Tile, tile_to: Tile, track: int,
                           side: SwitchBoxSide):
        assert tile_from in self
        assert tile_to in self
        # Connect the underlying routing graph.
        sb_from = tile_from.get_sb(side, track, SwitchBoxIO.SB_OUT)
        sb_to = tile_to.get_sb(
            SwitchBoxSide.get_opposite_side(side), track, SwitchBoxIO.SB_IN)
        assert sb_from is not None and sb_to is not None
        sb_from.add_edge(sb_to)

    def clone(self):
        clone = InterconnectGraph(self.bit_width)
        for (x, y), tile in self._tiles.items():
            clone._tiles[(x, y)] = tile.clone()
        # Clone the switch id list. Note that we are very slopy with the switch
        # id since the equality check will ensure it works.
        clone._switch_ids = self._switch_ids.copy()
        # Clone the tile grid.
        for row in self._tile_grid:
            new_row = []
            for entry in row:
                if entry is None:
                    new_row.append(None)
                else:
                    new_row.append(clone._tiles[(entry.x, entry.y)])
            clone._tile_grid.append(new_row)

        # Clone the connections.
        for _, tile in self._tiles.items():
            for sb_node in tile.switchbox.get_all_sbs():
                new_sb_node = self.locate_node(clone, sb_node)
                for node in sb_node:
                    new_node = self.locate_node(clone, node)
                    new_sb_node.add_edge(new_node, sb_node.get_edge_cost(node))
            for _, reg_node in tile.switchbox.registers.items():
                new_reg_node = self.locate_node(clone, reg_node)
                for node in reg_node:
                    new_node = self.locate_node(clone, node)
                    new_reg_node.add_edge(new_node,
                                          reg_node.get_edge_cost(node))
            for _, port_node in tile.ports.items():
                new_port_node = self.locate_node(clone, port_node)
                for node in port_node:
                    new_node = self.locate_node(clone, node)
                    new_port_node.add_edge(new_node,
                                           new_port_node.get_edge_cost(node))
            for _, reg_mux in tile.switchbox.reg_muxs.items():
                new_reg_mux = self.locate_node(clone, reg_mux)
                for node in reg_mux:
                    new_node = self.locate_node(clone, node)
                    new_reg_mux.add_edge(new_node,
                                         new_reg_mux.get_edge_cost(node))

        return clone

    @staticmethod
    def locate_node(graph: "InterconnectGraph", node: Node):
        tile = graph._tiles[(node.x, node.y)]
        if isinstance(node, SwitchBoxNode):
            return tile.get_sb(node.side, node.track, node.io)
        if isinstance(node, PortNode):
            return tile.ports[node.name]
        if isinstance(node, RegisterNode):
            return tile.switchbox.registers[node.name]
        assert isinstance(node, RegisterMuxNode)
        return tile.switchbox.reg_muxs[node.name]

    def __iter__(self):
        return iter(self._tiles)

    @property
    def switch_ids(self):
        return self._switch_ids.copy()

    @property
    def tiles(self):
        return self._tiles.copy()


def dump_graph(graph: InterconnectGraph, filename: str, max_num_col: int):
    with open(filename, "w+") as f:
        padding = "  "
        begin = "BEGIN"
        end = "END"

        def write_line(value):
            f.write(value + "\n")

        def write_conn(node_):
            if len(node_) == 0:
                return  # no output if no connections
            if node_.x >= max_num_col:
                return
            # TODO(keyi): Test for determinism.
            write_line(padding + node_.node_str())
            write_line(padding + begin)
            for n in node_:
                if (isinstance(node_, SwitchBoxNode) and
                    isinstance(n, SwitchBoxNode)):
                    if node_.x == n.x and node_.y == n.y:
                        continue  # skip as is an internal connection
                if n.x >= max_num_col:
                    continue
                write_line(padding * 3 + n.node_str())
            write_line(padding + end)

        for _, switch in self._switch_ids.items():
            write_line(str(switch))
            write_line(begin)
            for conn in switch.internal_wires:
                track_from, side_from, track_to, side_to = conn
                write_line(padding + " ".join([str(track_from),
                                              str(side_from.value),
                                              str(track_to),
                                              str(side_to.value)]))
            write_line(end)

        for (x, _), tile in self._tiles.items():
            if x >= max_num_col:
                # Since x starts from 0, if x == max_num_col, we are actually
                # out of bounds.
                continue
            write_line(str(tile))
            sbs = tile.switchbox.get_all_sbs()
            for sb in sbs:
                write_conn(sb)
            for _, node in tile.ports.items():
                write_conn(node)
            for _, reg in tile.switchbox.registers.items():
                write_conn(reg)
            for _, reg_mux in tile.switchbox.reg_muxs.items():
                write_conn(reg_mux)
