"""
This is a layer build on top of Cyclone
"""
from collections import OrderedDict

import kratos.util
from gemstone.common.core import Core
from gemstone.common.mux_with_default import MuxWithDefaultWrapper
from gemstone.common.configurable import Configurable, ConfigurationType
from .cyclone import Node, PortNode, Tile, SwitchBoxNode, SwitchBoxIO, \
    SwitchBox, InterconnectCore, RegisterNode, RegisterMuxNode, create_name
import mantle
import math
from gemstone.common.mux_wrapper import MuxWrapper
from gemstone.common.mux_wrapper_aoi import AOIMuxWrapper, AOIMuxType
import magma
from typing import Dict, Tuple, List, Union
from gemstone.generator.generator import Generator as GemstoneGenerator
from gemstone.generator.from_magma import FromMagma
from mantle import DefineRegister
from gemstone.generator.const import Const
from .logic import ExclusiveNodeFanout, InclusiveNodeFanout, \
    ReadyValidLoopBack, FifoRegWrapper
import os


def get_mux_sel_name(node: Node):
    return f"{create_name(str(node))}_sel"


class _PassThroughFromMux(GemstoneGenerator):
    def __init__(self, mux, ready_valid: bool = False):
        super().__init__()
        self.width = mux.width
        self.height = mux.height
        self.instance_name = mux.instance_name
        self._name = mux.name()
        self.add_ports(I=magma.In(magma.Bits[mux.width]),
                       O=magma.Out(magma.Bits[mux.width]))
        self.wire(self.ports.I, self.ports.O)

        if ready_valid:
            bit = magma.Bit
            self.add_ports(ready_in=magma.In(bit),
                           ready_out=magma.Out(bit),
                           valid_in=magma.In(bit),
                           valid_out=magma.Out(bit))
            self.wire(self.ports.ready_in, self.ports.ready_out)
            self.wire(self.ports.valid_in, self.ports.valid_out)

    def name(self):
        return self._name


def flatten_mux(mux, ready_valid: bool = False):
    if mux.height > 1:
        return mux
    return _PassThroughFromMux(mux, ready_valid)


def create_mux(node: Node, suffix: str = "", width: int = 0, ready_valid=False):
    conn_in = node.get_conn_in()
    height = len(conn_in)
    node_name = create_name(str(node)) + suffix
    if height > 1:
        if "MUX" not in node_name:
            name = f"MUX_{node_name}"
        else:
            name = node_name
    else:
        name = f"WIRE_{node_name}"
    if width == 0:
        width = node.width
    if ready_valid:
        mux = AOIMuxWrapper(height, width,
                            mux_type=AOIMuxType.RegularReadyValid, name=name)
    else:
        mux = AOIMuxWrapper(height, width, mux_type=AOIMuxType.Regular, name=name)
    return flatten_mux(mux, ready_valid=ready_valid)


def _safe_wire(self, port1, port2):
    if port1.base_type() is magma.BitIn and port2.base_type() is magma.In(magma.Bits[1]) \
            or port1.base_type() is magma.BitIn and port2.base_type() is magma.Out(magma.Bits[1]) \
            or port1.base_type() is magma.BitOut and port2.base_type() is magma.In(magma.Bits[1]) \
            or port1.base_type() is magma.BitOut and port2.base_type() is magma.Out(magma.Bits[1]):
        return self.wire(port1, port2[0])
    elif port2.base_type() is magma.BitIn and port1.base_type() is magma.In(magma.Bits[1]) \
            or port2.base_type() is magma.BitIn and port1.base_type() is magma.Out(magma.Bits[1]) \
            or port2.base_type() is magma.BitOut and port1.base_type() is magma.In(magma.Bits[1]) \
            or port2.base_type() is magma.BitOut and port1.base_type() is magma.Out(magma.Bits[1]):
        return self.wire(port1[0], port2)
    return self.wire(port1, port2)


class InterconnectConfigurable(Configurable):
    def safe_wire(self, port1, port2):
        return _safe_wire(self, port1, port2)


class CB(InterconnectConfigurable):
    def __init__(self, node: PortNode,
                 config_addr_width: int, config_data_width: int,
                 double_buffer: bool = False,
                 ready_valid: bool = False):
        if not isinstance(node, PortNode):
            raise ValueError(node, PortNode.__name__)
        self.node: PortNode = node

        super().__init__(config_addr_width, config_data_width,
                         double_buffer=double_buffer)

        self.mux = create_mux(self.node, ready_valid=ready_valid)

        # lift the port to the top level
        self.add_ports(O=self.mux.ports.O.base_type())
        self.wire(self.ports.O, self.mux.ports.O)
        self.add_ports(I=self.mux.ports.I.base_type())
        self.wire(self.ports.I, self.mux.ports.I)

        # ready valid has other stuff
        if ready_valid:
            self.add_ports(
                ready_in=magma.In(magma.Bit),
                ready_out=magma.Out(magma.Bit),
                valid_in=magma.In(magma.Bits[self.mux.height]),
                valid_out=magma.Out(magma.Bit)
            )
            for port_type in ["ready", "valid"]:
                for direction in ["in", "out"]:
                    port_name = f"{port_type}_{direction}"
                    self.safe_wire(self.ports[port_name],
                                   self.mux.ports[port_name])

        if self.mux.height > 1:
            self.add_ports(
                config=magma.In(ConfigurationType(config_addr_width,
                                                  config_data_width)),
            )
            config_name = get_mux_sel_name(self.node)
            self.add_config(config_name, self.mux.sel_bits)
            self.wire(self.registers[config_name].ports.O,
                      self.mux.ports.S)

            if ready_valid:
                self.add_port("out_sel", self.mux.ports.out_sel.base_type())
                self.wire(self.ports.out_sel, self.mux.ports.out_sel)
                self.add_config(str(node) + "_enable", 1)
                enable = self.add_port("enable", magma.BitOut)
                self.wire(enable, self.registers[str(node) + "_enable"].ports.O[0])

        else:
            # remove clk and reset ports from the base class since it's going
            # to be a pass through wire anyway
            self.ports.pop("clk")
            self.ports.pop("reset")
            self.ports.pop("read_config_data")
            if self.double_buffer:
                self.ports.pop("use_db")
                self.ports.pop("config_db")

            if ready_valid:
                self.add_port("out_sel", magma.Out(magma.Bits[1]))
                self.wire(Const(magma.Bits[1](0)), self.ports.out_sel)

        self.instance_name = self.name()

    def name(self):
        return create_name(str(self.node))


class SB(InterconnectConfigurable):
    def __init__(self, switchbox: SwitchBox, config_addr_width: int,
                 config_data_width: int, core_name: str = "",
                 stall_signal_width: int = 4,
                 double_buffer: bool = False,
                 ready_valid: bool = False,
                 combinational_ports=None):
        self.finalized = False
        self.ready_valid = ready_valid
        self.switchbox = switchbox
        self.__core_name = core_name
        self.stall_signal_width = stall_signal_width
        self.combinational_ports = combinational_ports if combinational_ports is not None else {}

        self.sb_muxs: Dict[str, Tuple[SwitchBoxNode, AOIMuxWrapper]] = {}
        self.reg_muxs: Dict[str, Tuple[RegisterMuxNode, MuxWrapper]] = {}
        self.regs: Dict[str, Tuple[RegisterNode, FromMagma]] = {}

        self.name_to_fanout = {}
        self.defer_mux_valid_in = True

        self.mux_name_to_node: Dict[str:, Node] = {}

        use_non_split_fifos = "USE_NON_SPLIT_FIFOS" in os.environ and os.environ.get("USE_NON_SPLIT_FIFOS") == "1"
        self.use_non_split_fifos = use_non_split_fifos

        super().__init__(config_addr_width, config_data_width,
                         double_buffer=double_buffer)

        # turn off hashing because we control the hash by ourselves
        self.set_skip_hash(True)

        # first pass to create the mux and register circuit
        self.__create_reg()
        self.__create_sb_mux()
        self.__create_reg_mux()

        # second pass to lift the ports and wire them
        for sb_name, (sb, mux) in self.sb_muxs.items():
            # only lift them if the ports are connect to the outside world
            port_name = create_name(sb_name)
            if sb.io == SwitchBoxIO.SB_IN:
                self.add_port(port_name, magma.In(mux.ports.I.base_type()))
                self.wire(self.ports[port_name], mux.ports.I)

            else:
                self.add_port(port_name, magma.Out(mux.ports.O.base_type()))

                # to see if we have a register mux here
                # if so , we need to lift the reg_mux output instead
                if sb_name in self.reg_muxs:
                    # override the mux value
                    node, mux = self.reg_muxs[sb_name]
                    assert isinstance(node, RegisterMuxNode)
                    assert node in sb
                self.wire(self.ports[port_name], mux.ports.O)

        # connect internal sbs
        self.__connect_sbs()

        # connect regs and reg muxs
        # we need to make three connections in total
        #      REG
        #  1 /    \ 3
        # SB ______ MUX
        #       2
        self.__connect_sb_out()
        self.__connect_regs()

        # set up the configuration registers, if needed
        if len(self.sb_muxs) > 0:
            self.add_ports(
                config=magma.In(ConfigurationType(config_addr_width,
                                                  config_data_width)),
            )
        else:
            # remove added ports since it's a empty switchbox
            self.ports.pop("clk")
            self.ports.pop("reset")
            self.ports.pop("read_config_data")

        for _, (sb, mux) in self.sb_muxs.items():
            config_name = get_mux_sel_name(sb)
            if mux.height > 1:
                assert mux.sel_bits > 0
                self.add_config_node(sb, config_name, mux.sel_bits)
                self.wire(self.registers[config_name].ports.O,
                          mux.ports.S)

        for _, (reg_mux, mux) in self.reg_muxs.items():
            config_name = get_mux_sel_name(reg_mux)
            assert mux.height == 2
            assert mux.sel_bits > 0
            self.add_config_node(reg_mux, config_name, mux.sel_bits)
            self.wire(self.registers[config_name].ports.O,
                      mux.ports.S)

        # name
        self.instance_name = self.name()

        # ready valid interface
        self._wire_flush()
        self.__wire_reg_reset()
        self.__connect_nodes_fanin()
        self.__lift_ready_valid()

        # extra hashing because we don't represent it in the module name
        _hash = hash(self)
        # ordering doesn't mater here
        for reg_name in self.regs:
            _hash ^= hash(reg_name)
        # also hash the internal wires based on switch box id
        _hash ^= hash(switchbox.id)
        self.set_hash(_hash)
        self.set_skip_hash(False)

    def finalize(self):
        if self.finalized:
            return
        self.finalized = True
        self._setup_config()

        # clock gate the pipeline registers if not used
        self.__wire_config_ce()

    def add_config_node(self, node: Node, name, width):
        super().add_config(name, width)
        # index the name to node so that we can trace back during the
        # configuration
        self.mux_name_to_node[name] = node

    def __create_sb_mux(self):
        sbs = self.switchbox.get_all_sbs()
        for sb in sbs:
            sb_name = str(sb)
            self.sb_muxs[sb_name] = (
                sb, create_mux(sb, ready_valid=self.ready_valid))

            # for ready valid, we need 1-bit config to know whether
            # the mux is being used or not
            if self.ready_valid:
                enable_name = sb_name + "_enable"
                self.add_config(enable_name, 1)
                self.add_port(enable_name, magma.BitOut)
                self.wire(self.registers[enable_name].ports.O[0], self.ports[enable_name])

    def __create_reg_mux(self):
        for _, reg_mux in self.switchbox.reg_muxs.items():
            # assert the connections to make sure it's a valid register
            # mux
            conn_ins = reg_mux.get_conn_in()
            assert len(conn_ins) == 2
            # find out the sb it's connected in. also do some checking
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
                raise ValueError("expect a sb connected to the reg_mux")
            # we use the sb_name instead so that when we lift the port up,
            # we can use the mux output instead
            sb_name = str(sb_node)
            self.reg_muxs[sb_name] = (
                reg_mux, create_mux(reg_mux, ready_valid=self.ready_valid))

    def __create_reg(self):
        for reg_name, reg_node in self.switchbox.registers.items():
            if self.ready_valid:
                reg = FifoRegWrapper(reg_node.width, self.use_non_split_fifos)
            else:
                reg_cls = DefineRegister(reg_node.width, has_ce=True)
                reg = FromMagma(reg_cls)
            reg.instance_name = create_name(str(reg_node))
            self.regs[reg_name] = reg_node, reg
        # add stall ports
        if len(self.regs) > 0:
            self.add_port("stall",
                          magma.In(magma.Bits[self.stall_signal_width]))

    def __wire_config_ce(self):
        if len(self.regs) == 0:
            return
        # fanout the stall signals to registers
        # invert the stall signal to clk_en
        invert = FromMagma(mantle.DefineInvert(1))
        # FIXME: use the low bits of stall signal to stall
        self.wire(invert.ports.I[0], self.ports.stall[0])
        for (reg_node, reg) in self.regs.values():
            rmux: RegisterMuxNode = list(reg_node)[0]
            # get rmux address
            config_name = get_mux_sel_name(rmux)
            config_reg = self.registers[config_name]
            index_val = rmux.get_conn_in().index(reg_node)
            eq_gate = FromMagma(mantle.DefineEQ(config_reg.width))
            self.wire(eq_gate.ports.I0, Const(index_val))
            self.wire(eq_gate.ports.I1, config_reg.ports.O)
            and_gate = FromMagma(mantle.DefineAnd(2, 1))
            self.wire(and_gate.ports.I0[0], eq_gate.ports.O)
            self.wire(and_gate.ports.I1, invert.ports.O)
            self.wire(reg.ports.CE, self.convert(and_gate.ports.O[0],
                                                 magma.enable))

    def _wire_flush(self):
        if len(self.regs) == 0:
            return
        # Only for ready valid mode
        if self.ready_valid:
            self.add_port("flush", magma.In(magma.BitIn))
            for (_, reg) in self.regs.values():
                self.wire(self.ports.flush, reg.ports.flush)

    def __wire_reg_reset(self):
        # only for ready valid mode
        if self.ready_valid:
            for (_, reg) in self.regs.values():
                self.wire(self.ports.reset, reg.ports.ASYNCRESET)

    def __get_connected_port_names(self) -> List[str]:
        # this is to uniquify the SB given different port connections
        result = set()
        for sb in self.switchbox.get_all_sbs():
            nodes = sb.get_conn_in()[:]
            nodes += list(sb)
            for node in nodes:
                if isinstance(node, PortNode) and node.x == self.switchbox.x \
                        and node.y == self.switchbox.y \
                        and len(node.get_conn_in()) == 0:
                    result.add(node.name)
        # make it deterministic
        result = list(result)
        result.sort()
        return result

    def name(self):
        return f"SB_ID{self.switchbox.id}_{self.switchbox.num_track}TRACKS_" \
               f"B{self.switchbox.width}_{self.__core_name}"

    def __connect_sbs(self):
        # the principle is that it only connects to the nodes within
        # its range. for instance, in SB we only connect to sb nodes
        for _, (sb, mux) in self.sb_muxs.items():
            if sb.io == SwitchBoxIO.SB_IN:
                for node in sb:
                    if isinstance(node, SwitchBoxNode):
                        assert node.io == SwitchBoxIO.SB_OUT
                        assert node.x == sb.x and node.y == sb.y
                        output_port = mux.ports.O
                        idx = node.get_conn_in().index(sb)
                        node_, node_mux = self.sb_muxs[str(node)]
                        assert node_ == node
                        input_port = node_mux.ports.I[idx]
                        self.wire(input_port, output_port)
                        if self.ready_valid and not self.defer_mux_valid_in:
                            self.wire(node_mux.ports.valid_in[idx],
                                      mux.ports.valid_out)

    def __connect_sb_out(self):
        for _, (sb, mux) in self.sb_muxs.items():
            if sb.io == SwitchBoxIO.SB_OUT:
                for node in sb:
                    if isinstance(node, RegisterNode):
                        reg_name = node.name
                        reg_node, reg = self.regs[reg_name]
                        assert len(reg_node.get_conn_in()) == 1
                        # wire 1
                        self.wire(mux.ports.O, reg.ports.I)
                        if self.ready_valid:
                            self.wire(mux.ports.valid_out, reg.ports.valid_in)
                    elif isinstance(node, RegisterMuxNode):
                        assert len(node.get_conn_in()) == 2
                        idx = node.get_conn_in().index(sb)
                        sb_name = str(sb)
                        n, reg_mux = self.reg_muxs[sb_name]
                        assert n == node
                        # wire 2
                        self.wire(mux.ports.O, reg_mux.ports.I[idx])
                        if self.ready_valid:
                            self.wire(mux.ports.valid_out,
                                      reg_mux.ports.valid_in[idx])

    def __connect_regs(self):
        for _, (node, reg) in self.regs.items():
            assert len(node) == 1, "pipeline register only has 1" \
                                   " connection"
            reg_mux_node: RegisterMuxNode = list(node)[0]
            # make a copy since we need to pop the list
            reg_mux_conn = reg_mux_node.get_conn_in()[:]
            assert len(reg_mux_conn) == 2, "register mux can only have 2 " \
                                           "incoming connections"
            reg_mux_conn.remove(node)
            sb_node: SwitchBoxNode = reg_mux_conn[0]
            assert node in sb_node, "register has to be connected together " \
                                    "with a reg mux"
            sb_name = str(sb_node)
            n, mux = self.reg_muxs[sb_name]
            assert n == reg_mux_node
            idx = reg_mux_node.get_conn_in().index(node)
            # wire 3
            self.wire(reg.ports.O, mux.ports.I[idx])

            if self.ready_valid:
                # need to connect valid signals
                self.wire(reg.ports.valid_out, mux.ports.valid_in[idx])
                self.wire(reg.ports.ready_in, mux.ports.ready_out)
                self.__handle_rmux_fanin(sb_node, n, node)

                # set fifo enable mode
                fifo_name = str(node) + "_fifo"
                self.add_config(fifo_name, 1)
                fifo_en = self.registers[fifo_name]
                self.wire(fifo_en.ports.O[0], reg.ports.fifo_en)

                # set start and end if using splitFIFOs
                if not self.use_non_split_fifos:
                    start_name = str(node) + "_start"
                    self.add_config(start_name, 1)
                    start = self.registers[start_name]
                    self.wire(start.ports.O[0], reg.ports.start_fifo)
                    end_name = str(node) + "_end"
                    self.add_config(end_name, 1)
                    end = self.registers[end_name]
                    self.wire(end.ports.O[0], reg.ports.end_fifo)

                # Set bogus mode
                bogus_init_name = str(node) + "_bogus_init"
                if self.use_non_split_fifos:
                    self.add_config(bogus_init_name, 2)
                else:
                    self.add_config(bogus_init_name, 1)
                bogus_init = self.registers[bogus_init_name]

                if self.use_non_split_fifos:
                    self.wire(bogus_init.ports.O, reg.ports.bogus_init)
                else:
                    self.wire(bogus_init.ports.O[0], reg.ports.bogus_init)

    def __handle_rmux_fanin(self, sb: SwitchBoxNode, rmux: RegisterMuxNode,
                            reg: RegisterNode):
        # exclusive because the destination mux can only select one
        sb_fanout = ExclusiveNodeFanout.get(len(sb))
        sb_fanout.instance_name = create_name(str(sb)) + "_FANOUT"
        reg_circuit = self.regs[reg.name][1]
        rmux_circuit = self.reg_muxs[str(sb)][1]
        sb_circuit = self.sb_muxs[str(sb)][1]
        reg_index = rmux.get_conn_in().index(reg)
        sb_index = rmux.get_conn_in().index(sb)
        self.wire(sb_fanout.ports.I[reg_index], reg_circuit.ports.ready_out)
        self.wire(sb_fanout.ports.I[sb_index], rmux_circuit.ports.ready_out)
        self.wire(sb_fanout.ports.O[0], sb_circuit.ports.ready_in)
        # use rmux selection
        self.wire(sb_fanout.ports.S, rmux_circuit.ports.out_sel)

    def handle_node_fanin(self, node: Node):
        # FIXME
        for n in node:
            if len(n.get_conn_in()) <= 1:
                return
        fanout = InclusiveNodeFanout.get(node)

        # Going to lift this valid up to gate it...
        if self.defer_mux_valid_in:
            out_port_name = str(node) + "_ready_reduction_for_gating"
            # print(f"Adding ready reduction for gating on : {node}")
            # print(f"{out_port_name}")
            self.add_port(out_port_name, magma.Out(magma.Bits[1]))
            self.wire(fanout.ports.O, self.ports[out_port_name])

        # fanout ports is I{i} S{i} E{i} and sel{i}
        for idx, n in enumerate(list(node)):
            # need to get the mux the node is connected to
            if not isinstance(n, SwitchBoxNode):
                assert isinstance(n, PortNode)
                assert len(n) == 0
                # this is a port input, i.e. CB
                # we assume it's properly connected from outside
                if n.name in self.combinational_ports:
                    # this is a combinational port
                    en = Const(magma.Bits[1](0))
                    self.wire(fanout.ports[f"I{idx}"][0], Const(magma.Bit(0)))
                    self.wire(fanout.ports[f"S{idx}"], Const(0))
                else:
                    ready = n.name + "_ready"
                    sel = n.name + "_out_sel"
                    if ready not in self.ports:
                        self.add_port(ready, magma.BitIn)
                        num_in = int(
                            math.pow(2, kratos.clog2(len(n.get_conn_in()))))
                        self.add_port(sel, magma.In(magma.Bits[num_in]))
                    self.wire(fanout.ports[f"I{idx}"][0], self.ports[ready])
                    self.wire(fanout.ports[f"S{idx}"], self.ports[sel])
                    enable = n.name + "_enable"
                    if enable not in self.ports:
                        self.add_port(enable, magma.In(magma.Bits[1]))
                    en = self.ports[enable]
            else:
                assert isinstance(n, SwitchBoxNode)
                sb: SwitchBoxNode = n
                assert sb.io == SwitchBoxIO.SB_OUT
                sb_circuit = self.sb_muxs[str(sb)][1]
                self.wire(fanout.ports[f"I{idx}"][0],
                          sb_circuit.ports.ready_out)
                self.wire(fanout.ports[f"S{idx}"], sb_circuit.ports.out_sel)
                en = self.registers[str(n) + "_enable"].ports.O
                # self.name_to_fanout[n] = fanout.ports.O[0]
            self.wire(fanout.ports[f"E{idx}"], en)
        # get the node circuit
        if isinstance(node, SwitchBoxNode):
            port = self.sb_muxs[str(node)][1].ports.ready_in
        else:
            # has to be a port
            assert isinstance(node, PortNode)
            port_name = f"{node.name}_ready_out"
            if port_name not in self.ports:
                self.add_port(port_name, magma.BitOut)
            port = self.ports[port_name]

        self.name_to_fanout[node] = fanout.ports.O[0]
        self.wire(fanout.ports.O[0], port)

    def add_port_valid(self, node: PortNode):
        valid_port = node.name + "_valid"
        port = self.add_port(valid_port, magma.BitIn)
        assert len(node.get_conn_in()) == 0
        for n in node:
            idx = n.get_conn_in().index(node)
            if isinstance(n, SwitchBoxNode):
                assert n.io == SwitchBoxIO.SB_OUT
                mux = self.sb_muxs[str(n)][1]
            else:
                assert isinstance(n, PortNode)
                # this is port to port connection
                # the dest has to be a CB
                assert len(n) == 0
                # FIXME
                continue
            self.wire(mux.ports.valid_in[idx], port)

        return port

    def __connect_nodes_fanin(self):
        if not self.ready_valid:
            return
        for sb, _ in self.sb_muxs.values():
            if sb.io == SwitchBoxIO.SB_IN:
                self.handle_node_fanin(sb)

    def __lift_ready_valid(self):
        # lift ready valid signals from the muxes to the top
        # in this way we can connect them to each other in the
        if not self.ready_valid:
            return

        # At this point we can hook up the gated valids to the output muxes
        if self.defer_mux_valid_in:
            for sb_name, (sb, mux) in self.sb_muxs.items():
                if sb.io == SwitchBoxIO.SB_IN:
                    for node in sb:
                        if isinstance(node, SwitchBoxNode):
                            assert node.io == SwitchBoxIO.SB_OUT
                            assert node.x == sb.x and node.y == sb.y
                            # output_port = mux.ports.O
                            idx = node.get_conn_in().index(sb)
                            node_, node_mux = self.sb_muxs[str(node)]
                            assert node_ == node
                            # Gate it here...
                            fanin_node_port = self.name_to_fanout[sb]
                            and_gate = FromMagma(mantle.DefineAnd(2, 1))
                            self.wire(and_gate.ports.I0[0], mux.ports.valid_out)
                            self.wire(and_gate.ports.I1[0], fanin_node_port)
                            self.wire(node_mux.ports.valid_in[idx], and_gate.ports.O[0])

        for sb_name, (sb, mux) in self.sb_muxs.items():
            port_name = create_name(sb_name)
            in_bit = magma.In(magma.Bit)
            out_bit = magma.Out(magma.Bit)
            if sb.io == SwitchBoxIO.SB_IN:
                self.add_port(port_name + "_ready_out", out_bit)
                self.add_port(port_name + "_valid_in", in_bit)
                self.wire(self.ports[port_name + "_ready_out"],
                          mux.ports.ready_out)
                self.wire(self.ports[port_name + "_valid_in"],
                          mux.ports.valid_in)
            else:
                self.add_port(port_name + "_ready_in", in_bit)
                self.add_port(port_name + "_valid_out", out_bit)

                # to see if we have a register mux here
                # if so , we need to lift the reg_mux output instead
                if sb_name in self.reg_muxs:
                    # override the mux value
                    node, mux = self.reg_muxs[sb_name]
                    assert isinstance(node, RegisterMuxNode)
                    assert node in sb
                self.wire(self.ports[port_name + "_ready_in"],
                          mux.ports.ready_in)
                self.wire(self.ports[port_name + "_valid_out"],
                          mux.ports.valid_out)


class TileCircuit(GemstoneGenerator):
    """We merge tiles at the same coordinates with different bit widths
    The only requirements is that the tiles have to be on the same
    coordinates. Their heights do not have to match.

    We don't deal with stall signal here since it's not interconnect's job
    to handle that signal
    """

    def __init__(self, tiles: Dict[int, Tile],
                 config_addr_width: int, config_data_width: int,
                 tile_id_width: int = 16,
                 full_config_addr_width: int = 32,
                 stall_signal_width: int = 4,
                 double_buffer: bool = False,
                 ready_valid: bool = False,
                 give_north_io_sbs: bool = False):
        super().__init__()

        # turn off hashing because we controls that hashing here
        self.set_skip_hash(True)

        self.tiles = tiles
        self.config_addr_width = config_addr_width
        self.config_data_width = config_data_width
        self.tile_id_width = tile_id_width

        self.double_buffer = double_buffer
        self.ready_valid = ready_valid

        # compute config addr sizes
        # (16, 24)
        full_width = full_config_addr_width
        self.full_config_addr_width = full_config_addr_width
        self.feature_addr_slice = slice(full_width - self.tile_id_width,
                                        full_width - self.config_addr_width)
        # (0, 16)
        self.tile_id_slice = slice(0, self.tile_id_width)
        # (24, 32)
        self.feature_config_slice = slice(full_width - self.config_addr_width,
                                          full_width)

        # sanity check
        x = -1
        y = -1
        core = None
        self.additional_cores = []
        additional_core_names = set()
        for bit_width, tile in self.tiles.items():
            assert bit_width == tile.track_width
            if x == -1:
                x = tile.x
                y = tile.y
                core = tile.core
            else:
                assert x == tile.x
                assert y == tile.y
                # the restriction is that all the tiles in the same coordinate
                # have to have the same core, otherwise it's physically
                # impossible
                assert core == tile.core
            for a_core, _ in tile.additional_cores:
                a_core = a_core.core
                assert isinstance(a_core, Core)
                core_name = a_core.name()
                if core_name not in additional_core_names:
                    self.additional_cores.append(a_core)
                    additional_core_names.add(core_name)

        assert x != -1 and y != -1
        self.x = x
        self.y = y
        self.core = core.core
        self.core_interface = core

        # compute combinational core ports
        if self.core_interface is None:
            self.combinational_ports = set()
        else:
            self.combinational_ports = self.core_interface.combinational_ports()
            for a_core in self.additional_cores:
                self.combinational_ports = self.combinational_ports.union(a_core.combinational_ports())

        # create cb and switchbox
        self.cbs: Dict[str, CB] = {}
        self.sbs: Dict[int, SB] = {}
        # we only create cb if it's an input port, which doesn't have
        # graph neighbors
        for bit_width, tile in self.tiles.items():
            # connection box time
            for port_name, port_node in tile.ports.items():
                if give_north_io_sbs:
                    # Lift up if io2glb or glb2io port (and skip the rest i.e., adding SB and CB connections)
                    if "glb2io" in port_name:
                        ready_port_name = port_name + "_ready"
                        valid_port_name = port_name + "_valid"

                        core_port = self.__get_port(port_name)
                        core_ready_port = self.__get_port(ready_port_name)
                        core_valid_port = self.__get_port(valid_port_name)

                        self.add_port(port_name, magma.In(core_port.base_type()))
                        self.add_port(valid_port_name, magma.In(magma.Bit))
                        self.add_port(ready_port_name, magma.Out(magma.Bit))

                        self.wire(self.ports[port_name], core_port)
                        self.wire(self.convert(self.ports[valid_port_name], magma.Bits[1]), core_valid_port)
                        self.wire(self.convert(core_ready_port, magma.bit), self.ports[ready_port_name])

                        continue

                    if "io2glb" in port_name:
                        ready_port_name = port_name + "_ready"
                        valid_port_name = port_name + "_valid"

                        core_port = self.__get_port(port_name)
                        core_ready_port = self.__get_port(ready_port_name)
                        core_valid_port = self.__get_port(valid_port_name)

                        self.add_port(port_name, magma.Out(core_port.base_type()))
                        self.add_port(valid_port_name, magma.Out(magma.Bit))
                        self.add_port(ready_port_name, magma.In(magma.Bit))

                        self.wire(core_port, self.ports[port_name])
                        self.wire(self.convert(core_valid_port, magma.bit), self.ports[valid_port_name])
                        self.wire(self.convert(self.ports[ready_port_name], magma.Bits[1]), core_ready_port)
                        continue

                # input ports
                if len(port_node) == 0:
                    assert bit_width == port_node.width
                    # make sure that it has at least one connection
                    if len(port_node.get_conn_in()) == 0:
                        continue
                    # create a CB
                    port_refs = tile.get_port_ref(port_node.name)
                    cb = CB(port_node, config_addr_width, config_data_width,
                            double_buffer=self.double_buffer,
                            ready_valid=self.ready_valid)
                    if not isinstance(port_refs, list):
                        port_refs = [port_refs]
                    for p in port_refs:
                        self.wire(cb.ports.O, p)
                    self.cbs[port_name] = cb

                    # if the input is combinational, wire constant
                    no_rv = self.ready_valid and port_name in self.combinational_ports
                    if no_rv:
                        # we wire constant 1 to ready valid port
                        self.wire(cb.ports.ready_in, Const(1))
                else:
                    # output ports
                    assert len(port_node.get_conn_in()) == 0
                    assert bit_width == port_node.width

            # switch box time
            core_name = self.core.name() if self.core is not None else ""
            sb = SB(tile.switchbox, config_addr_width, config_data_width,
                    core_name, stall_signal_width=stall_signal_width,
                    double_buffer=self.double_buffer,
                    ready_valid=self.ready_valid,
                    combinational_ports=self.combinational_ports)
            self.sbs[sb.switchbox.width] = sb

        # lift all the sb ports up
        for _, switchbox in self.sbs.items():
            sbs = switchbox.switchbox.get_all_sbs()
            assert switchbox.switchbox.x == self.x
            assert switchbox.switchbox.y == self.y
            bit_in = magma.In(magma.Bit)
            bit_out = magma.Out(magma.Bit)
            for sb in sbs:
                sb_name = create_name(str(sb))
                node, mux = switchbox.sb_muxs[str(sb)]
                assert node == sb
                assert sb.x == self.x
                assert sb.y == self.y
                port = switchbox.ports[sb_name]
                if node.io == SwitchBoxIO.SB_IN:
                    self.add_port(sb_name, magma.In(port.base_type()))

                    if self.ready_valid:
                        vi = self.add_port(sb_name + "_valid", bit_in)
                        ro = self.add_port(sb_name + "_ready", bit_out)
                        self.wire(vi, switchbox.ports[sb_name + "_valid_in"])
                        self.wire(ro, switchbox.ports[sb_name + "_ready_out"])
                else:
                    self.add_port(sb_name, magma.Out(port.base_type()))
                    if self.ready_valid:
                        vo = self.add_port(sb_name + "_valid", bit_out)
                        ri = self.add_port(sb_name + "_ready", bit_in)

                        self.wire(vo, switchbox.ports[sb_name + "_valid_out"])
                        self.wire(ri, switchbox.ports[sb_name + "_ready_in"])

                assert port.owner() == switchbox
                self.wire(self.ports[sb_name], port)

        # connect ports from cb to switch box and back
        for _, cb in self.cbs.items():
            conn_ins = cb.node.get_conn_in()
            for idx, node in enumerate(conn_ins):
                assert isinstance(node,
                                  (SwitchBoxNode, RegisterMuxNode, PortNode))
                # for IO tiles they have connections to other tiles
                if node.x != self.x or node.y != self.y:
                    continue
                bit_width = node.width
                sb_circuit = self.sbs[bit_width]
                if not isinstance(node, PortNode):
                    # get the internal wire
                    n, sb_mux = sb_circuit.sb_muxs[str(node)]
                    assert n == node
                    sb_name = create_name(str(node))
                    if node.io == SwitchBoxIO.SB_IN:
                        self.wire(self.ports[sb_name], cb.ports.I[idx])
                        if self.ready_valid:
                            # print("GATING INPUT VALID")
                            port_name = str(node) + "_valid"
                            # print(f"{port_name}")
                            sb_ready_reduction_port = str(node) + "_ready_reduction_for_gating"
                            # print(f"{sb_ready_reduction_port}")
                            and_gate = FromMagma(mantle.DefineAnd(2, 1))
                            self.wire(and_gate.ports.I0[0], self.ports[port_name])
                            self.wire(and_gate.ports.I1, sb_circuit.ports[sb_ready_reduction_port])
                            self.wire(cb.ports.valid_in[idx], and_gate.ports.O[0])
                            # Actually gate it here too from the SB port...
                            # self.wire(self.ports[port_name],
                            #           cb.ports.valid_in[idx])
                    else:
                        self.wire(sb_circuit.ports[sb_name], cb.ports.I[idx])
                else:
                    # this is an additional core port
                    # just connect directly
                    width = node.width
                    tile = self.tiles[width]
                    self.wire(tile.get_port_ref(node.name), cb.ports.I[idx])
                    if self.ready_valid:
                        node_valid = node.name + "_valid"
                        p = tile.get_port_ref(node_valid, node.name)
                        self.wire(p[0], cb.ports.valid_in[idx])

        # connect ports from core to switch box
        for bit_width, tile in self.tiles.items():
            sb_circuit = self.sbs[bit_width]
            for _, port_node in tile.ports.items():
                if len(port_node) == 0:
                    continue
                assert len(port_node.get_conn_in()) == 0
                port_name = port_node.name
                for sb_node in port_node:
                    assert isinstance(sb_node, (SwitchBoxNode, PortNode))
                    if isinstance(sb_node, PortNode):
                        continue
                    # for IO tiles they have connections to other tiles
                    if sb_node.x != self.x or sb_node.y != self.y:
                        continue
                    idx = sb_node.get_conn_in().index(port_node)
                    # we need to find the actual mux
                    n, mux = sb_circuit.sb_muxs[str(sb_node)]
                    assert n == sb_node
                    # the generator doesn't allow circular reference
                    # we have to be very creative here
                    if port_name not in sb_circuit.ports:
                        sb_circuit.add_port(port_name,
                                            magma.In(magma.Bits[bit_width]))
                        self.wire(self.__get_port(port_name),
                                  sb_circuit.ports[port_name])
                    sb_circuit.wire(sb_circuit.ports[port_name],
                                    mux.ports.I[idx])

                if self.ready_valid:
                    # need to create fan-in logic
                    if len(sb_circuit.ports) == 0:
                        continue
                    else:
                        sb_circuit.handle_node_fanin(port_node)
                        ready_in = port_node.name + "_ready"
                        ready_out = port_node.name + "_ready_out"
                        core_ready_in = self.__get_port(ready_in)
                        t = core_ready_in.base_type()
                        if t is magma.In(magma.Bits[1]):
                            core_ready_in = core_ready_in[0]
                        self.wire(sb_circuit.ports[ready_out], core_ready_in)
                        valid_port = sb_circuit.add_port_valid(port_node)
                        loop_back = ReadyValidLoopBack.get()
                        loop_back.instance_name = port_node.name + "_loopback"
                        self.wire(loop_back.ports.ready_in[0], sb_circuit.ports[ready_out])
                        valid_out = port_node.name + "_valid"
                        core_valid_out = self.__get_port(valid_out)
                        t = core_valid_out.base_type()
                        if t is magma.Out(magma.Bits[1]):
                            core_valid_out = core_valid_out[0]
                        self.wire(loop_back.ports.valid_in[0], core_valid_out)
                        self.wire(loop_back.ports.valid_out[0], valid_port)

        # CB ready-valid
        if self.ready_valid:
            for bit_width, tile in self.tiles.items():
                sb_circuit = self.sbs[bit_width]
                for _, port_node in tile.ports.items():
                    if len(port_node) != 0 or port_node.name not in self.cbs:
                        continue
                    if port_node.name in self.combinational_ports:
                        continue
                    cb = self.cbs[port_node.name]
                    out_sel = port_node.name + "_out_sel"
                    enable = port_node.name + "_enable"
                    ready = port_node.name + "_ready"
                    valid = port_node.name + "_valid"
                    if out_sel in sb_circuit.ports:
                        self.wire(cb.ports.out_sel, sb_circuit.ports[out_sel])
                        self.wire(cb.ports.enable, sb_circuit.ports[enable][0])
                        self.wire(cb.ports.ready_out, sb_circuit.ports[ready])

                    no_rv = port_node.name in self.combinational_ports
                    if not no_rv:
                        core_ready = self.__get_port(ready)
                        core_valid = self.__get_port(valid)
                        t = core_ready.base_type()
                        if t is magma.Out(magma.Bits[1]):
                            core_ready = core_ready[0]
                        t = core_valid.base_type()
                        if t is magma.In(magma.Bits[1]):
                            core_valid = core_valid[0]
                        self.wire(cb.ports.valid_out, core_valid)
                        self.wire(cb.ports.ready_in, core_ready)

        self.__add_tile_id()
        # add ports
        self.add_ports(stall=magma.In(magma.Bits[stall_signal_width]),
                       reset=magma.In(magma.AsyncReset),
                       clk=magma.In(magma.Clock))

        # lift ports if there is empty sb
        self.__lift_ports()

        # tile ID
        self.instance_name = f"Tile_X{self.x:02X}_Y{self.y:02X}"

        # add features
        self.__features: List[GemstoneGenerator] = []
        # users are free to add more features to the core

        # add feature
        if self.core is not None:
            # add core features first
            assert isinstance(self.core, Core)
            for feature in self.core.features():
                self.add_feature(feature)

        for c in self.additional_cores:
            assert isinstance(c, Core)
            for feature in c.features():
                self.add_feature(feature)

        # then CBs
        cb_names = list(self.cbs.keys())
        cb_names.sort()
        for name in cb_names:
            self.add_feature(self.cbs[name])
        # then SBs
        sb_widths = list(self.sbs.keys())
        sb_widths.sort()
        for width in sb_widths:
            self.add_feature(self.sbs[width])

        # placeholder for global signal wiring
        self.read_data_mux: MuxWithDefaultWrapper = None

        self.finalized = False

        hash_value = hash(self.name())
        for sb in self.sbs.values():
            hash_value ^= hash(sb)
        for cb in self.cbs.values():
            hash_value ^= hash(cb)
        self.set_hash(hash_value)
        self.set_skip_hash(False)

    def __get_port(self, port_name):
        if port_name in self.core.ports:
            return self.core.ports[port_name]
        else:
            for core in self.additional_cores:
                if port_name in core.ports:
                    return core.ports[port_name]
        return None

    def __add_tile_id(self):
        self.add_port("tile_id",
                      magma.In(magma.Bits[self.tile_id_width]))

    def __add_stall(self):
        # automatically add stall signal and connect it to the features if the
        # feature supports it
        stall_ports = set()
        for feature in self.features():
            if "stall" in feature.ports.keys():
                stall_ports.add(feature.ports.stall)
        # some core may not expose the port as features, such as mem cores
        if self.core is not None and "stall" in self.core.ports and \
                self.core.ports.stall not in stall_ports:
            stall_ports.add(self.core.ports.stall)
        for stall_port in stall_ports:
            self.wire(self.ports.stall, stall_port)

    def __add_reset(self):
        # automatically add reset signal and connect it to the features if the
        # feature supports it
        reset_ports = set()
        for feature in self.features():
            if "reset" in feature.ports.keys():
                reset_ports.add(feature.ports.reset)
        # some core may not expose the port as features, such as mem cores
        if self.core is not None and "reset" in self.core.ports and \
                self.core.ports.reset not in reset_ports:
            reset_ports.add(self.core.ports.reset)

        for reset_port in reset_ports:
            self.wire(self.ports.reset, reset_port)

    def __add_clk(self):
        # automatically add clk signal and connect it to the features if the
        # feature supports it
        clk_ports = set()
        for feature in self.features():
            if "clk" in feature.ports.keys():
                clk_ports.add(feature.ports.clk)
        # some core may not expose the port as features, such as mem cores
        if self.core is not None and "clk" in self.core.ports and \
                self.core.ports.clk not in clk_ports:
            clk_ports.add(self.core.ports.clk)

        for clk_port in clk_ports:
            self.wire(self.ports.clk, clk_port)

    def __wire_flush_to_sbs(self):
        for _, switchbox in self.sbs.items():
            if len(switchbox.regs) == 0:
                return
            self.wire(self.ports.flush[0], switchbox.ports.flush)

    def __should_add_config(self):
        # a introspection on itself to determine whether to add config
        # or not
        for feature in self.features():
            if "config" in feature.ports:
                return True
        return False

    def wire_internal(self, port1: str, port2: str):
        # wire internal ports directly, for the cores
        # just silently quit if the port doesn't exist
        assert port1 != port2
        port1_ref = None
        port2_ref = None
        cores = [self.core] + self.additional_cores
        for core in cores:
            for port_name, port in core.ports().items():
                if port1_ref is not None and port_name == port1:
                    port1_ref = port
                if port2_ref is not None and port_name == port2:
                    port2_ref = port
        assert port1_ref.owner() != port2_ref.owner()
        self.wire(port1_ref, port2_ref)

    def needs_signal(self, signal_name):
        features = self.features()
        for feat in features:
            if signal_name in feat.ports:
                return True
        return False

    def finalize(self):
        if self.finalized:
            raise Exception("Circuit already finalized")
        self.finalized = True
        # add stall and reset signal
        self.__add_stall()
        self.__add_reset()
        self.__add_clk()

        # MO: Flush signal HACK
        self.__wire_flush_to_sbs()

        # see if we really need to add config or not
        if not self.__should_add_config():
            return

        self.add_ports(
            config=magma.In(ConfigurationType(self.full_config_addr_width,
                                              self.config_data_width)),
            read_config_data=magma.Out(magma.Bits[self.config_data_width])
        )
        # double buffer ports
        if self.double_buffer:
            self.add_ports(
                config_db=magma.In(magma.Bit),
                use_db=magma.In(magma.Bit)
            )

        features = self.features()
        num_features = len(features)
        self.read_data_mux = MuxWithDefaultWrapper(num_features,
                                                   self.config_data_width,
                                                   self.config_addr_width,
                                                   0)
        self.read_data_mux.instance_name = "read_data_mux"
        # most of the logic copied from tile_magma.py
        # remove all hardcoded values
        for feature in self.features():
            if hasattr(feature, "finalize"):
                feature.finalize()
            if "config" not in feature.ports:
                continue
            self.wire(self.ports.config.config_addr[self.feature_config_slice],
                      feature.ports.config.config_addr)
            self.wire(self.ports.config.config_data,
                      feature.ports.config.config_data)
            self.wire(self.ports.config.read, feature.ports.config.read)

            if self.double_buffer and "config_db" in feature.ports:
                self.wire(self.ports.config_db, feature.ports.config_db)
                self.wire(self.ports.use_db, feature.ports.use_db)

        # Connect S input to config_addr.feature.
        self.wire(self.ports.config.config_addr[self.feature_addr_slice],
                  self.read_data_mux.ports.S)
        self.wire(self.read_data_mux.ports.O, self.ports.read_config_data)

        # Logic to generate EN input for read_data_mux
        read_and_tile = FromMagma(mantle.DefineAnd(2))
        eq_tile = FromMagma(mantle.DefineEQ(self.tile_id_width))
        # config_addr.tile_id == self.tile_id?
        self.wire(self.ports.tile_id, eq_tile.ports.I0)
        self.wire(self.ports.config.config_addr[self.tile_id_slice],
                  eq_tile.ports.I1)
        # (config_addr.tile_id == self.tile_id) & READ
        self.wire(read_and_tile.ports.I0, eq_tile.ports.O)
        self.wire(read_and_tile.ports.I1, self.ports.config.read[0])
        # read_data_mux.EN = (config_addr.tile_id == self.tile_id) & READ
        self.wire(read_and_tile.ports.O, self.read_data_mux.ports.EN[0])

        # Logic for writing to config registers
        # Config_en_tile = (config_addr.tile_id == self.tile_id & WRITE)
        write_and_tile = FromMagma(mantle.DefineAnd(2))
        self.wire(write_and_tile.ports.I0, eq_tile.ports.O)
        self.wire(write_and_tile.ports.I1, self.ports.config.write[0])
        decode_feat = []
        feat_and_config_en_tile = []
        for i, feat in enumerate(self.features()):
            # wire each feature's read_data output to
            # read_data_mux inputs
            if "read_config_data" in feat.ports:
                self.wire(feat.ports.read_config_data,
                          self.read_data_mux.ports.I[i])
            else:
                # wire constant
                self.wire(Const(0), self.read_data_mux.ports.I[i])
            # for each feature,
            # config_en = (config_addr.feature == feature_num) & config_en_tile
            decode_feat.append(
                FromMagma(mantle.DefineDecode(i, self.config_addr_width)))
            decode_feat[-1].instance_name = f"DECODE_FEATURE_{i}"
            feat_and_config_en_tile.append(FromMagma(mantle.DefineAnd(2)))
            feat_and_config_en_tile[-1].instance_name = f"FEATURE_AND_{i}"
            self.wire(decode_feat[i].ports.I,
                      self.ports.config.config_addr[self.feature_addr_slice])
            self.wire(decode_feat[i].ports.O,
                      feat_and_config_en_tile[i].ports.I0)
            self.wire(write_and_tile.ports.O,
                      feat_and_config_en_tile[i].ports.I1)
            if "config" in feat.ports:
                self.wire(feat_and_config_en_tile[i].ports.O,
                          feat.ports.config.write[0])
            if "config_en" in feat.ports:
                self.wire(decode_feat[i].ports.O, feat.ports["config_en"])

    def add_feature(self, feature: GemstoneGenerator):
        assert isinstance(feature, GemstoneGenerator)
        self.__features.append(feature)

    def features(self) -> List[GemstoneGenerator]:
        return self.__features

    __CONFIG_TYPE = Tuple[int, int, int]
    __BITSTREAM_TYPE = Union[__CONFIG_TYPE, List[__CONFIG_TYPE]]

    def __add_additional_config(self, name, value, _circuit, configs):
        _reg_idx, _config_data = _circuit.get_config_data(name, value)
        _feature_addr = self.features().index(_circuit)
        _additional_config = _reg_idx, _feature_addr, _config_data
        configs.append(_additional_config)

    def get_route_bitstream_config(self, src_node: Node, dst_node: Node) -> __BITSTREAM_TYPE:
        assert src_node.width == dst_node.width
        tile = self.tiles[src_node.width]
        assert dst_node.x == tile.x and dst_node.y == tile.y, \
            f"{dst_node} is not in {tile}"
        assert dst_node in src_node, \
            f"{dst_node} is not connected to {src_node}"

        config_data = dst_node.get_conn_in().index(src_node)
        # find the circuit
        if isinstance(dst_node, SwitchBoxNode):
            circuit = self.sbs[src_node.width]
        elif isinstance(dst_node, PortNode):
            circuit = self.cbs[dst_node.name]
        elif isinstance(dst_node, RegisterMuxNode):
            circuit = self.sbs[src_node.width]
        else:
            raise NotImplementedError(type(dst_node))
        reg_name = get_mux_sel_name(dst_node)
        reg_idx, config_data = circuit.get_config_data(reg_name, config_data)
        feature_addr = self.features().index(circuit)
        base_config = reg_idx, feature_addr, config_data
        if self.ready_valid:
            configs = [base_config]
            # need to get mux enable if necessary
            circuit = None
            if isinstance(dst_node, SwitchBoxNode):
                circuit = self.sbs[src_node.width]
            elif isinstance(dst_node, PortNode):
                circuit = self.cbs[dst_node.name]

            if circuit is not None:
                self.__add_additional_config(str(dst_node) + "_enable", 1, circuit, configs)

            return configs
        return base_config

    def configure_fifo(self, node: RegisterNode, start: bool, end: bool, use_non_split_fifos: bool = False,
                       bogus_init: bool = False, bogus_init_num: int = 0):
        configs = []
        # we only turn this on if it's a path from register to mux with ready-valid
        circuit = self.sbs[node.width]
        reg_name = str(node) + "_fifo"
        self.__add_additional_config(reg_name, 1, circuit, configs)
        bogus_init_name = str(node) + "_bogus_init"
        self.__add_additional_config(bogus_init_name, bogus_init_num, circuit, configs)

        # Only do start and end config for split FIFOs
        if not use_non_split_fifos:
            start_name = str(node) + "_start"
            end_name = str(node) + "_end"
            start = int(start)
            end = int(end)
            bogus_init_name = str(node) + "_bogus_init"
            start = int(start)
            end = int(end)
            bogus_init = int(bogus_init)
            self.__add_additional_config(start_name, start, circuit, configs)
            self.__add_additional_config(end_name, end, circuit, configs)
            self.__add_additional_config(bogus_init_name, bogus_init, circuit, configs)

        return configs

    def __lift_ports(self):
        # lift the internal ports only if we have empty switch boxes
        for bit_width, sb in self.sbs.items():
            if sb.switchbox.num_track > 0:
                continue
            # lift the input ports up
            for bt, port_name in self.core_interface.inputs():
                if bt != bit_width:
                    continue
                # depends on if the port has any connection or not
                # we lift the port up first
                # if it has no connection, then we lift it up
                port_node = self.tiles[bit_width].ports[port_name]
                if port_node.get_conn_in():
                    cb_input_port = self.cbs[port_name].ports.I
                    # use the CB input type instead
                    self.add_port(port_name, cb_input_port.base_type())
                    self.wire(self.ports[port_name], cb_input_port)
                    if self.ready_valid and port_name not in self.combinational_ports:
                        p = self.add_port(port_name + "_ready", magma.BitOut)
                        # could be a passthrough mux
                        self.safe_wire(p, self.cbs[port_name].ports.ready_out)
                        p = self.add_port(port_name + "_valid", magma.BitIn)
                        self.safe_wire(p, self.cbs[port_name].ports.valid_in)
                else:
                    self.add_port(port_name, magma.In(magma.Bits[bit_width]))
                    self.wire(self.ports[port_name], self.core.ports[port_name])
                    if self.ready_valid and port_name not in self.combinational_ports:
                        core_ready = self.core.ports[port_name + "_ready"]
                        core_valid = self.core.ports[port_name + "_valid"]
                        if core_valid.base_type() is magma.In(magma.Bits[1]):
                            core_valid = core_valid[0]
                        if core_ready.base_type() is magma.Out(magma.Bits[1]):
                            core_ready = core_ready[0]
                        p = self.add_port(port_name + "_ready", magma.BitOut)
                        self.safe_wire(p, core_ready)
                        p = self.add_port(port_name + "_valid", magma.BitIn)
                        self.safe_wire(p, core_valid)

            # lift the output ports up
            for bt, port_name in self.core_interface.outputs():
                if bt != bit_width:
                    continue
                port_node = self.tiles[bit_width].ports[port_name]
                # depends on if the port has any connection or not
                # we lift the port up first
                # if it has connection, then we connect it to the core
                self.add_port(port_name, magma.Out(magma.Bits[bit_width]))
                self.wire(self.ports[port_name], self.core.ports[port_name])
                if self.ready_valid and port_name not in self.combinational_ports:
                    core_ready = self.core.ports[port_name + "_ready"]
                    core_valid = self.core.ports[port_name + "_valid"]
                    if core_valid.base_type() is magma.Out(magma.Bits[1]):
                        core_valid = core_valid[0]
                    if core_ready.base_type() is magma.In(magma.Bits[1]):
                        core_ready = core_ready[0]
                    if len(port_node) > 1:
                        fanout = FromMagma(mantle.DefineAnd(len(port_node)))
                        fanout.instance_name = port_name + "_ready_merge"
                        p = self.add_port(port_name + "_ready", magma.In(magma.Bits[len(port_node)]))
                        self.wire(fanout.ports.O[0], core_ready)
                        # it has to be done with a loop
                        for i in range(len(port_node)):
                            self.wire(p[i], fanout.ports[f"I{i}"])
                    else:
                        p = self.add_port(port_name + "_ready", magma.BitIn)
                        self.wire(p, core_ready)
                    p = self.add_port(port_name + "_valid", magma.BitOut)
                    self.wire(p, core_valid)

    def safe_wire(self, port1, port2):
        return _safe_wire(self, port1, port2)

    def name(self):
        if self.core is not None:
            return f"Tile_{self.core.name()}"
        else:
            return "Tile_Empty"


class CoreInterface(InterconnectCore):
    def __init__(self, core: Core):
        super().__init__()

        self.input_ports = OrderedDict()
        self.output_ports = OrderedDict()

        self.core: Core = core

        # empty tile
        if core is None:
            return

        for port in core.inputs():
            port_name = port.qualified_name()
            width = self.__get_bit_width(port)
            self.input_ports[port_name] = (width, port)

        for port in core.outputs():
            port_name = port.qualified_name()
            width = self.__get_bit_width(port)
            self.output_ports[port_name] = (width, port)

    def inputs(self):
        return [(width, name) for name, (width, _) in self.input_ports.items()]

    def outputs(self):
        return [(width, name) for name, (width, _) in self.output_ports.items()]

    def get_port_ref(self, port_name: str):
        if port_name in self.input_ports:
            return self.input_ports[port_name][1]
        elif port_name in self.output_ports:
            return self.output_ports[port_name][1]
        return self.core.ports[port_name]

    def combinational_ports(self):
        res = set()
        if self.core is not None:
            for p in self.core.combinational_ports():
                if not isinstance(p, str):
                    res.add(p.qualified_name())
                else:
                    res.add(p)
        return res

    @staticmethod
    def __get_bit_width(port):
        # nasty function to get the actual bit width from the port reference
        t = port.type()
        if issubclass(t, magma.Digital):
            return 1
        if issubclass(port.type(), magma.Bits):
            return len(t)
        raise NotImplementedError(t, type(t))

    def __eq__(self, other: "CoreInterface"):
        return other.core == self.core
