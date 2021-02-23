from typing import Dict, Tuple

import magma as m

from canal.generators.common import (InterconnectConfigurable,
                                     make_mux_or_wire, make_name,
                                     make_mux_sel_name)
from canal.graph.node import Node
from canal.graph.register import RegisterNode
from canal.graph.register_mux import RegisterMuxNode
from canal.graph.sb import SwitchBoxNode, SwitchBoxIO
from canal.graph.sb_container import SwitchBox


def _make_registers(switch_box):
    registers = {}
    for name, node in switch_box.registers.items():
        inst_name = make_name(str(node))
        inst = m.Register(m.Bits[node.width], has_enable=True)(name=inst_name)
        registers[name] = node, inst
    return registers


def _make_switch_box_muxes(switch_box):
    return {str(sb): (sb, make_mux_or_wire(sb))
            for sb in switch_box.get_all_sbs()}


def _make_reg_muxes(switch_box):
    muxes = {}
    for _, reg_mux in switch_box.reg_muxes.items():
        # Check the # of connections to ensure it's a valid register.
        conn_ins = reg_mux.get_conn_in()
        assert len(conn_ins) == 2
        # Find the switch box it's connected in.
        node1, node2 = conn_ins
        if isinstance(node1, RegisterNode):
            assert isinstance(node2, SwitchBoxNode)
            assert node2.io == SwitchBoxIO.SB_OUT
            sb_node = node2
        elif isinstance(node2, RegisterNode):
            assert isinstance(node1, SwitchBoxNode)
            assert node1.io == SwitchBoxIO.SB_OUT
            sb_node = node1
        else:
            raise ValueError("Expected an sb connected to the reg_mux")
        # We use the sb_name instead so that when we lift the port up, we can
        # use the mux output instead.
        muxes[str(sb_node)] = (reg_mux, make_mux_or_wire(reg_mux))
    return muxes


def _connect_switch_boxes(switch_box_muxes):
    # Works by only connecting to the nodes within a nodes range. For instance,
    # in SB we only connect to sb nodes.
    for _, (sb, mux) in switch_box_muxes.items():
        if sb.io != SwitchBoxIO.SB_IN:
            continue
        for node in sb:
            if not isinstance(node, SwitchBoxNode):
                continue
            assert node.io == SwitchBoxIO.SB_OUT
            assert node.x == sb.x and node.y == sb.y
            idx = node.get_conn_in().index(sb)
            node_, node_mux = switch_box_muxes[str(node)]
            assert node_ == node
            node_mux.I[idx] @= mux.O


def _connect_switch_box_outputs(switch_box_muxes, registers, reg_muxes):
    for _, (sb, mux) in switch_box_muxes.items():
        if sb.io != SwitchBoxIO.SB_OUT:
            continue
        for node in sb:
            if isinstance(node, RegisterNode):
                reg_node, reg = registers[node.name]
                assert len(reg_node.get_conn_in()) == 1
                reg.I @= mux.O
            elif isinstance(node, RegisterMuxNode):
                assert len(node.get_conn_in()) == 2
                idx = node.get_conn_in().index(sb)
                node_, reg_mux = reg_muxes[str(sb)]
                assert node == node_
                reg_mux.I[idx] @= mux.O


def _connect_registers(registers, reg_muxes):
    for _, (node, reg) in registers.items():
        assert len(node) == 1
        reg_mux_node = list(node)[0]
        reg_mux_conn = reg_mux_node.get_conn_in().copy()  # mutable
        assert len(reg_mux_conn) == 2
        reg_mux_conn.remove(node)
        sb_node = reg_mux_conn[0]
        assert node in sb_node
        sb_name = str(sb_node)
        node_, mux = reg_muxes[sb_name]
        assert reg_mux_node == node_
        idx = reg_mux_node.get_conn_in().index(node)
        mux.I[idx] @= reg.O


class SB(InterconnectConfigurable):
    def __init__(self, switch_box: SwitchBox, config_addr_width: int,
                 config_data_width: int, core_name: str = "",
                 stall_signal_width: int = 4):
        name = (f"SB_ID{switch_box.id}_{switch_box.num_tracks}TRACKS_"
                f"B{switch_box.width}_{core_name}")
        super().__init__(name, config_addr_width, config_data_width)
        self.switch_box = switch_box

        # Instantiate registers, switch box muxes, and register muxes.
        TRegDict = Dict[str, Tuple[RegisterNode, m.Circuit]]
        TSBMuxDict = Dict[str, Tuple[SwitchBoxNode, m.Circuit]]
        TRegMuxDict = Dict[str, Tuple[RegisterMuxNode, m.Circuit]]
        with self._open():
            self.registers: TRegDict = _make_registers(switch_box)
            self.switch_box_muxes: TSBMuxDict = _make_switch_box_muxes(
                switch_box)
            self.reg_muxes: TRegMuxDict = _make_reg_muxes(switch_box)

        # Add stall signal if necessary (non-zero registers).
        if self.registers:
            self._add_port("stall", m.In(m.Bits[stall_signal_width]))

        # Lift and wire ports on switch boxes.
        for sb_name, (sb, mux) in self.switch_box_muxes.items():
            # Only lift if ports are connected to outside.
            port_name = make_name(sb_name)
            if sb.io == SwitchBoxIO.SB_IN:
                mux.I @= self._add_port(port_name, m.In(type(mux.I)))
            else:
                output = self._add_port(port_name, m.Out(type(mux.O)))
                # See if we have a register mux here. If so, we need to lift the
                # reg_mux output instead.
                if sb_name in self.reg_muxes:
                    node, mux = self.reg_muxes[sb_name]  # use the reg mux
                    assert isinstance(node, RegisterMuxNode)
                    assert node in sb
                output @= mux.O

        # Connect internal switch boxes.
        _connect_switch_boxes(self.switch_box_muxes)

        # Connect registers and reg muxes. There are three connections in total:
        #
        #        REG
        #    1 /    \ 3
        #   SB ______ MUX
        #         2
        #
        _connect_switch_box_outputs(
            self.switch_box_muxes, self.registers, self.reg_muxes)
        _connect_registers(self.registers, self.reg_muxes)

        # Add configuration for muxes. Additionally, keep a mapping from name to
        # node so that we can trace back during the configuration.
        self.mux_name_to_node: Dict[str, Node] = {}
        for _, (sb, mux) in self.switch_box_muxes.items():
            self._add_mux_config(sb, mux)
        for _, (reg_mux, mux) in self.reg_muxes.items():
            assert isinstance(type(mux), m.Mux)
            self._add_mux_config(reg_mux, mux)

        self._wire_config_ce()  # clock-gate unused pipeline registers

    @property
    def _stall(self):
        return self._port("stall")

    def _add_mux_config(self, node: Node, mux: m.Circuit):
        name = super()._add_mux_config(node, mux)
        if name is not None:
            self.mux_name_to_node[name] = node
        return name

    @m.builder_method
    def _wire_config_ce(self):
        if not self.registers:  # early exit since stall signal doesn't exist
            return
        # Fanout the stall signals to registers, and invert the stall signal to
        # clk_en.
        # TODO(keyi): Use the low bits of stall signal to stall.
        nstall = ~(self._stall[0])
        for reg_node, reg in self.registers.values():
            reg_mux = list(reg_node)[0]
            config_name = make_mux_sel_name(reg_mux)
            config_value = self._get_value(config_name)
            index = reg_mux.get_conn_in().index(reg_node)
            reg.CE @= nstall & (config_value == index)
