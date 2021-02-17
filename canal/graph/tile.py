import dataclasses
from typing import Dict, Optional, Union

from canal.graph.core import InterconnectCore, CoreConnectionType
from canal.graph.port import PortNode
from canal.graph.sb import (SwitchBoxSide, SwitchBoxIO,
                            SwitchBoxConnectionType, SwitchBoxNode)
from canal.graph.sb_container import SwitchBox


@dataclasses.dataclass
class Tile:
    x: int
    y: int
    track_width: int
    switchbox: SwitchBox
    height: int = 1

    def __post_init__(self):
        # Create a copy of switch box because the nodes have to be created.
        self.switchbox: SwitchBox = SwitchBox(x, y, switchbox.num_track,
                                              switchbox.width,
                                              switchbox.internal_wires)

        self.ports: Dict[str, PortNode] = {}
        self.inputs = OrderedSet()
        self.outputs = OrderedSet()

        # Placeholders for the core.
        self.core: InterconnectCore = None
        self.additional_cores = []
        self.__port_core: Dict[str, InterconnectCore] = {}

    def __eq__(self, other):
        return (isinstance(other, Tile) and
                self.x == other.x and
                self.y == other.y and
                self.height == other.height)

    def __repr__(self):
        return (f"TILE ({self.x}, {self.y}, {self.height}, "
                f"{self.switchbox.id})")

    def set_core(self, core: Optional[InterconnectCore]):
        self.inputs.clear()
        self.outputs.clear()
        self.ports.clear()
        self.core = core

        if core is None:  # clears the core
            return

        self._add_core(core)

    def _add_core(self, core: InterconnectCore,
                  connection_type:
                  CoreConnectionType=CoreConnectionType.Default):

        if connection_type & CoreConnectionType.CB == CoreConnectionType.CB:
            inputs = core.inputs()[:]
            inputs.sort(key=lambda x: x[1])
            for width, port_name in inputs:
                if width == self.track_width:
                    self.inputs.add(port_name)
                    self.ports[port_name] = PortNode(
                        port_name, self.x, self.y, width)
                    self._port_core[port_name] = core

        if connection_type & CoreConnectionType.SB == CoreConnectionType.SB:
            outputs = core.outputs()[:]
            outputs.sort(key=lambda x: x[1])
            for width, port_name in outputs:
                if width == self.track_width:
                    self.outputs.add(port_name)
                            self.ports[port_name] = PortNode(
                                port_name, self.x, self.y, width)
                    self._port_core[port_name] = core

    def add_additional_core(self, core: InterconnectCore,
                            connection_type: CoreConnectionType):
        if self.core is None:
            raise ValueError("Main core cannot be null", core)
        self.additional_cores.append((core, connection_type))
        self._add_core(core, connection_type)
        # Handle the extra cases.
        if connection_type & CoreConnectionType.Core == CoreConnectionType.Core:
            # Connect the output ports to the CB input. We directly add the
            # graph connection here.
            core_cbs: List[PortNode] = []
            for width, port_name in self.core.inputs():
                if width == self.track_width:
                    assert port_name in self.ports
                    core_cbs.append(self.ports[port_name])
            outputs = core.outputs()[:]
            outputs.sort(key=lambda x: x[1])
            for width, port_name in outputs:
                assert port_name not in self.ports
                if width == self.track_width:
                    for cb_node in core_cbs:
                        self.ports[port_name] = PortNode(
                            port_name, self.x, self.y, width)
                        self.ports[port_name].add_edge(cb_node)
                    self._port_core[port_name] = core

    def get_port_ref(self, port_name):
        return self._port_core[port_name].get_port_ref(port_name)

    def get_port(self, port_name):
        return self.ports.get(port_name, None)

    def core_has_input(self, port: str):
        return port in self.inputs

    def core_has_output(self, port: str):
        return port in self.outputs

    def name(self):
        return str(self)

    def get_sb(self, side: SwitchBoxSide, track: int,
               io: SwitchBoxIO) -> Union[SwitchBoxNode, None]:
        return self.switchbox.get_sb(side, track, io)

    def set_core_connection(self, port_name: str,
                            connection_type: List[SBConnectionType]):
        # Make sure it's an input port.
        is_input = self.core_has_input(port_name)
        is_output = self.core_has_output(port_name)

        if not is_input and not is_output:  # core doesn't have port_name
            return
        if not (is_input ^ is_output):
            raise ValueError(f"Core error: {port_name} cannot be "
                             f"both input and output port")

        port_node = self.ports[port_name]
        # Add to graph node first. RTL (mamga) muxes will be handled separately.
        for side, track, io in connection_type:
            sb = self.get_sb(side, track, io)
            if is_input:
                sb.add_edge(port_node)
            else:
                port_node.add_edge(sb)

    @staticmethod
    def create_tile(x: int, y: int, bit_width: int, num_tracks: int,
                    internal_wires: List[Tuple[int, SwitchBoxSide,
                                               int, SwitchBoxSide]],
                    height: int = 1) -> "Tile":
        switch = SwitchBox(x, y, num_tracks, bit_width, internal_wires)
        return Tile(x, y, bit_width, switch, height)

    def clone(self):
        switchbox = self.switchbox.clone()
        clone = Tile(self.x, self.y, self.track_width, switchbox, self.height)
        clone.switchbox = switchbox
        # NOTE: We don't clone the cores.
        clone.set_core(self.core)
        for core, conn in self.additional_cores:
            clone.add_additional_core(core, conn)
        return clone
