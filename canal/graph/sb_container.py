import dataclasses
from typing import Dict, List, Optional, Tuple

from canal.graph.sb improt SwitchBoxSide, SwitchBoxIO
from canal.graph.register_mux import RegisterMux


InternalWiresType = List[Tuple[int, SwitchBoxSide, int, SwitchBoxSide]]


@dataclasses.dataclass
class SwitchBox:
    x: int
    y: int
    num_track: int
    width: int
    internal_wires: InternalWiresType

    def __post_init__(self):
        self.id = 0
        self._sbs: List[List[List[SwitchBoxNode]]] = ([
            [[SwitchBoxNode(self.x, self.y, track, width, side, io)
              for track in range(self.num_track)] for io in SwitchBoxIO]
            for side in SwitchBoxSide
        ])

        # Assign internal wiring, with order in -> out.
        for track_src, side_src, track_dst, side_dst in self.internal_wires:
            src = self._sbs[side_src.value][SwitchBoxIO.SB_IN.value][track_src]
            dst = self._sbs[side_dst.value][SwitchBoxIO.SB_OUT.value][track_dst]
            sb_src.add_edge(sb_dst, 0)  # internal connection has no delay

        # Store the nodes for pipeline registers.
        self.registers: Dict[str, RegisterNode] = {}
        self.reg_muxs: Dict[str, RegisterMuxNode] = {}

    def __eq__(self, other):
        if not isinstance(other, SwitchBox):
            return False
        if len(self.internal_wires) != len(other.internal_wires):
            return False
        # Check bijection.
        # NOTE(rsetaluri): This only checks a subset relationship.
        for conn in self.internal_wires:
            if conn not in other.internal_wires:
                return False
        return True

    def __repr__(self):
        return f"SWITCH {self.width} {self.id} {self.num_track}"

    def __getitem__(self, key: Tuple[SwitchBoxSide, int, SwitchBoxIO]):
        if not isinstance(key, Tuple[SwitchBoxSide, int, SwitchBoxIO]):
            raise TypeError(key)
        side, track, io = key
        return self._sbs[side.value][io.value][track]

    def get_all_sbs(self) -> List[SwitchBoxNode]:
        result = []
        for track in range(self.num_track):
            for side in SwitchBoxSide:
                for io in SwitchBoxIO:
                    sb = self.get_sb(side, track, io)
                    if sb is not None:
                        result.append(sb)
        return result

    def get_sb(self, side: SwitchBoxSide, track: int,
               io: SwitchBoxIO) -> Optional[SwitchBoxNode]:
        # We may have removed some nodes.
        if track < len(self._sbs[side.value][io.value]):
            return self._sbs[side.value][io.value][track]
        return None

    def remove_side_sbs(self, side: SwitchBoxSide, io: SwitchBoxIO):
        # First remove the connections and nodes.
        for sb in self._sbs[side.value][io.value]:
            nodes_to_remove = list(sb)
            for node in nodes_to_remove:
                sb.remove_edge(node)
            for node in sb.get_conn_in():
                node.remove_edge(sb)

        self._sbs[side.value][io.value].clear()

        # Then remove the internal wires.
        wires_to_remove = set()
        for conn in self.internal_wires:
            _, side_src, _, side_dst = conn
            if io == SwitchBoxIO.SB_IN and side_src == side:
                wires_to_remove.add(conn)
            elif io == SwitchBoxIO.SB_OUT and side_dst == side:
                wires_to_remove.add(conn)
        for conn in wires_to_remove:
            self.internal_wires.remove(conn)

    def add_pipeline_register(self, side: SwitchBoxSide, track: int):
        # Find specific sb node.
        node = self.get_sb(side, track, SwitchBoxIO.SB_OUT)
        if node is None:
            return
        neighbors = {}
        for n in node:
            if isinstance(n, RegisterNode) or isinstance(n, RegisterMuxNode):
                raise Exception("Pipeline register already inserted")
            cost = node.get_edge_cost(n)
            neighbors[n] = cost
        # Disconnect first.
        for n in neighbors:
            node.remove_edge(n)
        # Create a register mux node and a register node.
        reg = RegisterNode(
            f"T{track}_{side.name}", node.x, node.y, track, node.width)
        reg_mux = RegisterMuxNode(node.x, node.y, track, node.width, side)
        # Connect node to them.
        node.add_edge(reg)
        node.add_edge(reg_mux)
        # Connect reg to the reg_mux.
        reg.add_edge(reg_mux)
        # Add the connection back from the neighbors.
        for n, cost in neighbors.items():
            reg_mux.add_edge(n, cost)
        # Finally, add to the tile level.
        assert reg.name not in self.registers
        assert reg_mux.name not in self.reg_muxs
        self.registers[reg.name] = reg
        self.reg_muxs[reg_mux.name] = reg_mux

    def clone(self):
        clone = SwitchBox(
            self.x, self.y, self.num_track, self.width, self.internal_wires)
        # Clone other regs and reg muxs.
        for reg_name, reg_node in self.registers.items():
            clone.registers[reg_name] = RegisterNode(reg_node.name,
                                                     reg_node.x,
                                                     reg_node.y,
                                                     reg_node.track,
                                                     reg_node.width)
        for mux_name, mux_node in self.reg_muxs.items():
            clone.reg_muxs[mux_name] = RegisterMuxNode(mux_node.x,
                                                       mux_node.y,
                                                       mux_node.track,
                                                       mux_node.width,
                                                       mux_node.side)

        return clone
