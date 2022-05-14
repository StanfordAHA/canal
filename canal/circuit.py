"""
This is a layer build on top of Cyclone
"""
from collections import OrderedDict

import kratos.util
import _kratos
from gemstone.common.core import Core
from gemstone.common.mux_with_default import MuxWithDefaultWrapper
from gemstone.common.configurable import Configurable, ConfigurationType
from .cyclone import Node, PortNode, Tile, SwitchBoxNode, SwitchBoxIO, \
    SwitchBox, InterconnectCore, RegisterNode, RegisterMuxNode
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
from kratos import Generator, posedge, negedge, always_comb, always_ff, clog2, \
    const


def create_name(name: str):
    tokens = " (),"
    for t in tokens:
        name = name.replace(t, "_")
    name = name.replace("__", "_")
    if name[-1] == "_":
        name = name[:-1]
    return name


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


def flatten_mux(mux, ready_valid):
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
        mux = MuxWrapper(height, width, name=name)
    return flatten_mux(mux, ready_valid=ready_valid)


class RegFIFO(Generator):
    '''
    This module generates register-based FIFOs. These are useful
    when we only need a few entries with no prefetching needed
    '''

    def __init__(self,
                 data_width,
                 width_mult,
                 depth,
                 parallel=False,
                 break_out_rd_ptr=False,
                 almost_full_diff=2):

        super().__init__(f"reg_fifo_d_{depth}_w_{width_mult}")

        self.data_width = self.parameter("data_width", 16)
        self.data_width.value = data_width
        self.depth = depth
        self.width_mult = width_mult
        self.parallel = parallel
        self.break_out_rd_ptr = break_out_rd_ptr
        self.almost_full_diff = almost_full_diff

        assert not (depth & (depth - 1)), "FIFO depth needs to be a power of 2"

        # CLK and RST
        self._clk = self.clock("clk")
        self._rst_n = self.reset("rst_n")
        self._clk_en = self.clock_en("clk_en", 1)

        # INPUTS
        self._data_in = self.input("data_in",
                                   self.data_width,
                                   size=self.width_mult,
                                   explicit_array=True,
                                   packed=True)
        self._data_out = self.output("data_out",
                                     self.data_width,
                                     size=self.width_mult,
                                     explicit_array=True,
                                     packed=True)

        if self.parallel:
            self._parallel_load = self.input("parallel_load", 1)
            self._parallel_read = self.input("parallel_read", 1)
            self._num_load = self.input("num_load", clog2(self.depth) + 1)
            self._parallel_in = self.input("parallel_in",
                                           self.data_width,
                                           size=(self.depth,
                                                 self.width_mult),
                                           explicit_array=True,
                                           packed=True)

            self._parallel_out = self.output("parallel_out",
                                             self.data_width,
                                             size=(self.depth,
                                                   self.width_mult),
                                             explicit_array=True,
                                             packed=True)

        self._push = self.input("push", 1)
        self._pop = self.input("pop", 1)

        self._valid = self.output("valid", 1)

        ptr_width = max(1, clog2(self.depth))

        self._rd_ptr = self.var("rd_ptr", ptr_width)
        if self.break_out_rd_ptr:
            self._rd_ptr_out = self.output("rd_ptr_out", ptr_width)
            self.wire(self._rd_ptr_out, self._rd_ptr)
        self._wr_ptr = self.var("wr_ptr", ptr_width)
        self._read = self.var("read", 1)
        self._write = self.var("write", 1)
        self._reg_array = self.var("reg_array",
                                   self.data_width,
                                   size=(self.depth,
                                         self.width_mult),
                                   packed=True,
                                   explicit_array=True)

        self._passthru = self.var("passthru", 1)
        self._empty = self.output("empty", 1)
        self._full = self.output("full", 1)
        self._almost_full = self.output("almost_full", 1)

        self._num_items = self.var("num_items", clog2(self.depth) + 1)
        # self.wire(self._full, (self._wr_ptr + 1) == self._rd_ptr)
        self.wire(self._full, self._num_items == self.depth)
        # Experiment to cover latency
        self.wire(self._almost_full,
                  self._num_items >= (self.depth - self.almost_full_diff))
        # self.wire(self._empty, self._wr_ptr == self._rd_ptr)
        self.wire(self._empty, self._num_items == 0)

        self.wire(self._read, self._pop & ~self._passthru & ~self._empty)

        # Disallow passthru for now to prevent combinational loops
        self.wire(self._passthru, const(0, 1))
        # self.wire(self._passthru, self._pop & self._push & self._empty)

        # Should only write

        # Boilerplate Add always @(posedge clk, ...) blocks
        if self.parallel:
            self.add_code(self.set_num_items_parallel)
            self.add_code(self.reg_array_ff_parallel)
            self.add_code(self.wr_ptr_ff_parallel)
            self.add_code(self.rd_ptr_ff_parallel)
            self.wire(self._parallel_out, self._reg_array)
            self.wire(self._write,
                      self._push & ~self._passthru & (
                              ~self._full | (self._parallel_read)))
        else:
            # self.wire(self._write, self._push & ~self._passthru & (~self._full | self._pop))
            # Don't want to write when full at all for decoupling
            self.wire(self._write, self._push & ~self._passthru & (~self._full))
            self.add_code(self.set_num_items)
            self.add_code(self.reg_array_ff)
            self.add_code(self.wr_ptr_ff)
            self.add_code(self.rd_ptr_ff)
        self.add_code(self.data_out_ff)
        self.add_code(self.valid_comb)

    @always_ff((posedge, "clk"), (negedge, "rst_n"))
    def rd_ptr_ff(self):
        if ~self._rst_n:
            self._rd_ptr = 0
        elif self._read:
            self._rd_ptr = self._rd_ptr + 1

    @always_ff((posedge, "clk"), (negedge, "rst_n"))
    def rd_ptr_ff_parallel(self):
        if ~self._rst_n:
            self._rd_ptr = 0
        elif self._parallel_load | self._parallel_read:
            self._rd_ptr = 0
        elif self._read:
            self._rd_ptr = self._rd_ptr + 1

    @always_ff((posedge, "clk"), (negedge, "rst_n"))
    def wr_ptr_ff(self):
        if ~self._rst_n:
            self._wr_ptr = 0
        elif self._write:
            if self._wr_ptr == (self.depth - 1):
                self._wr_ptr = 0
            else:
                self._wr_ptr = self._wr_ptr + 1

    @always_ff((posedge, "clk"), (negedge, "rst_n"))
    def wr_ptr_ff_parallel(self):
        if ~self._rst_n:
            self._wr_ptr = 0
        elif self._parallel_load:
            self._wr_ptr = self._num_load[max(1, clog2(self.depth)) - 1, 0]
        elif self._parallel_read:
            if self._push:
                self._wr_ptr = 1
            else:
                self._wr_ptr = 0
        elif self._write:
            self._wr_ptr = self._wr_ptr + 1
            # if self._wr_ptr == (self.depth - 1):
            #     self._wr_ptr = 0
            # else:
            #     self._wr_ptr = self._wr_ptr + 1

    @always_ff((posedge, "clk"), (negedge, "rst_n"))
    def reg_array_ff(self):
        if ~self._rst_n:
            self._reg_array = 0
        elif self._write:
            self._reg_array[self._wr_ptr] = self._data_in

    @always_ff((posedge, "clk"), (negedge, "rst_n"))
    def reg_array_ff_parallel(self):
        if ~self._rst_n:
            self._reg_array = 0
        elif self._parallel_load:
            self._reg_array = self._parallel_in
        elif self._write:
            if self._parallel_read:
                self._reg_array[0] = self._data_in
            else:
                self._reg_array[self._wr_ptr] = self._data_in

    @always_comb
    def data_out_ff(self):
        if (self._passthru):
            self._data_out = self._data_in
        else:
            self._data_out = self._reg_array[self._rd_ptr]

    @always_comb
    def valid_comb(self):
        self._valid = ((~self._empty) | self._passthru)
        # self._valid = self._pop & ((~self._empty) | self._passthru)

    @always_ff((posedge, "clk"), (negedge, "rst_n"))
    def set_num_items(self):
        if ~self._rst_n:
            self._num_items = 0
        elif self._write & ~self._read:
            self._num_items = self._num_items + 1
        elif ~self._write & self._read:
            self._num_items = self._num_items - 1

    @always_ff((posedge, "clk"), (negedge, "rst_n"))
    def set_num_items_parallel(self):
        if ~self._rst_n:
            self._num_items = 0
        elif self._parallel_load:
            # When fetch width > 1, by definition we cannot load
            # 0 (only fw or fw - 1), but we need to handle this immediate
            # pass through in these cases
            if self._num_load == 0:
                self._num_items = self._push.extend(self._num_items.width)
            else:
                self._num_items = self._num_load
        # One can technically push while a parallel
        # read is happening...
        elif self._parallel_read:
            if self._push:
                self._num_items = 1
            else:
                self._num_items = 0
        elif self._write & ~self._read:
            self._num_items = self._num_items + 1
        elif ~self._write & self._read:
            self._num_items = self._num_items - 1


class FifoRegWrapper(GemstoneGenerator):
    __cache = {}

    def __init__(self, width):
        super(FifoRegWrapper, self).__init__()
        self.width = width

        if width not in FifoRegWrapper.__cache:
            gen = RegFIFO(width, 1, depth=2)
            _kratos.passes.auto_insert_clock_enable(gen.internal_generator)
            circuit = kratos.util.to_magma(gen)
            FifoRegWrapper.__cache[width] = circuit
        self.__circuit = FromMagma(FifoRegWrapper.__cache[width])

        # create wrapper ports so that it has the same interface as normal
        # magma registers modulo ready valid
        self.add_ports(
            I=magma.In(magma.Bits[width]),
            O=magma.Out(magma.Bits[width]),
            clk=magma.In(magma.Clock),
            CE=magma.In(magma.Enable),
            ASYNCRESET=magma.In(magma.AsyncReset),
            valid_in=magma.In(magma.Bit),
            valid_out=magma.Out(magma.Bit),
            ready_in=magma.In(magma.Bit),
            ready_out=magma.Out(magma.Bit)
        )

        self.wire(self.ports.I, self.__circuit.ports.data_in)
        self.wire(self.ports.O, self.__circuit.ports.data_out)
        self.wire(self.ports.clk, self.__circuit.ports.clk)
        self.wire(self.ports.CE, self.__circuit.ports.clk_en[0])
        # need an inverter for async reset
        async_inverter = FromMagma(mantle.Not)
        async_inverter.instance_name = "async_inverter"
        self.wire(self.ports.ASYNCRESET,
                  self.convert(async_inverter.ports.I, magma.asyncreset))
        self.wire(self.convert(async_inverter.ports.O, magma.asyncreset),
                  self.__circuit.ports.rst_n)
        self.wire(self.ports.valid_in, self.__circuit.ports.push[0])
        self.wire(self.ports.valid_out, self.__circuit.ports.valid[0])
        self.wire(self.ports.ready_in, self.__circuit.ports["pop"][0])
        # invert full
        full_inverter = FromMagma(mantle.Not)
        full_inverter.instance_name = "full_inverter"
        self.wire(full_inverter.ports.I, self.__circuit.ports.full[0])
        self.wire(self.ports.ready_out, full_inverter.ports.O)

    def name(self):
        return f"FifoRegWrapper_{self.width}"


class ExclusiveNodeFanout(Generator):
    __cache = {}

    def __init__(self, height: int):
        super(ExclusiveNodeFanout, self).__init__(
            f"ExclusiveNodeFanout_H{height}")
        self.input("I", height)
        self.output("O", 1)
        sel_size = int(math.pow(2, kratos.clog2(height)))
        self.input("S", sel_size)
        assert height >= 1

        expr = self.ports.I[0] & self.ports.S[0]
        for i in range(1, height):
            expr = expr | (self.ports.I[i] & self.ports.S[i])

        self.wire(self.ports.O, expr)

    @staticmethod
    def get(height: int):
        if height not in ExclusiveNodeFanout.__cache:
            inst = ExclusiveNodeFanout(height)
            circuit = kratos.util.to_magma(inst)
            ExclusiveNodeFanout.__cache[height] = circuit
        circuit = ExclusiveNodeFanout.__cache[height]
        return FromMagma(circuit)


class InclusiveNodeFanout(Generator):
    __cache = {}

    def __init__(self, node: Node):
        _hash = InclusiveNodeFanout.__compute_hash(node)

        # make sure we have proper mux
        for n in node:
            assert len(n.get_conn_in()) > 1

        super(InclusiveNodeFanout, self).__init__(f"FanoutHash_{_hash}")

        self.output("O", 1)

        temp_vars = []
        for idx, n in enumerate(list(node)):
            s = self.input(f"S{idx}", InclusiveNodeFanout.get_sel_size(
                len(n.get_conn_in())))
            i = self.input(f"I{idx}", 1)
            e = self.input(f"E{idx}", 1)
            v = self.var(f"sel{idx}", 1)
            # each term is (~E[i] OR I[i] OR ~S[i])
            mux_i = n.get_conn_in().index(node)
            self.add_stmt(v.assign(((~e) | (~s[mux_i])) | i))
            temp_vars.append(v)

        self.wire(self.ports.O, kratos.util.reduce_and(*temp_vars))

    @staticmethod
    def get_sel_size(height):
        return int(math.pow(2, kratos.clog2(height)))

    @staticmethod
    def __compute_hash(node: Node):
        selections = []
        for n in list(node):
            s = InclusiveNodeFanout.get_sel_size(len(n.get_conn_in()))
            selections.append((s, n.get_conn_in().index(node)))
        selections = tuple(selections)
        _hash = hash(selections)
        _hash = "{0:X}".format(abs(_hash))
        return _hash

    @staticmethod
    def get(node: Node):
        _hash = InclusiveNodeFanout.__compute_hash(node)
        if _hash not in InclusiveNodeFanout.__cache:
            inst = InclusiveNodeFanout(node)
            circuit = kratos.util.to_magma(inst)
            InclusiveNodeFanout.__cache[_hash] = circuit
        circuit = InclusiveNodeFanout.__cache[_hash]
        m = FromMagma(circuit)
        m.instance_name = create_name(str(node)) + "_fan_in"
        return m


class ReadyValidLoopBack(Generator):
    __cache = None

    def __init__(self):
        super(ReadyValidLoopBack, self).__init__("ReadyValidLoopBack")
        ri = self.input("ready_in", 1)
        vi = self.input("valid_in", 1)
        vo = self.output("valid_out", 1)
        ro = self.output("ready_out", 1)

        self.wire(ri, ro)
        self.wire(vo, ri & vi)

    @staticmethod
    def get():
        if ReadyValidLoopBack.__cache is None:
            inst = ReadyValidLoopBack()
            circuit = kratos.util.to_magma(inst)
            ReadyValidLoopBack.__cache = circuit
        circuit = ReadyValidLoopBack.__cache
        return FromMagma(circuit)


class InterconnectConfigurable(Configurable):
    pass


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

        if self.mux.height > 1:
            self.add_ports(
                config=magma.In(ConfigurationType(config_addr_width,
                                                  config_data_width)),
            )
            config_name = get_mux_sel_name(self.node)
            self.add_config(config_name, self.mux.sel_bits)
            self.wire(self.registers[config_name].ports.O,
                      self.mux.ports.S)

            # ready valid has other stuff
            if ready_valid:
                self.add_ports(
                    ready_in=magma.In(magma.Bit),
                    ready_out=magma.Out(magma.Bit),
                    valid_in=magma.In(magma.Bits[self.mux.height]),
                    valid_out=magma.Out(magma.Bit),
                    out_sel=self.mux.ports.out_sel.base_type()
                )
                for port_type in ["ready", "valid"]:
                    for direction in ["in", "out"]:
                        port_name = f"{port_type}_{direction}"
                        self.wire(self.ports[port_name],
                                  self.mux.ports[port_name])
                self.wire(self.ports.out_sel, self.mux.ports.out_sel)
                self.add_config(self.node.name, 1)
                enable = self.add_port("enable", magma.BitOut)
                self.wire(enable, self.registers[self.node.name].ports.O[0])

        else:
            # remove clk and reset ports from the base class since it's going
            # to be a pass through wire anyway
            self.ports.pop("clk")
            self.ports.pop("reset")
            self.ports.pop("read_config_data")
            if self.double_buffer:
                self.ports.pop("use_db")
                self.ports.pop("config_db")

        self.instance_name = self.name()

    def name(self):
        return create_name(str(self.node))


class SB(InterconnectConfigurable):
    def __init__(self, switchbox: SwitchBox, config_addr_width: int,
                 config_data_width: int, core_name: str = "",
                 stall_signal_width: int = 4,
                 double_buffer: bool = False,
                 ready_valid: bool = False):
        self.finalized = False
        self.ready_valid = ready_valid
        self.switchbox = switchbox
        self.__core_name = core_name
        self.stall_signal_width = stall_signal_width

        self.sb_muxs: Dict[str, Tuple[SwitchBoxNode, AOIMuxWrapper]] = {}
        self.reg_muxs: Dict[str, Tuple[RegisterMuxNode, MuxWrapper]] = {}
        self.regs: Dict[str, Tuple[RegisterNode, FromMagma]] = {}

        self.mux_name_to_node: Dict[str:, Node] = {}

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
            if self.ready_valid and len(sb.get_conn_in()) > 0:
                self.add_config(sb_name, 1)

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
                reg = FifoRegWrapper(reg_node.width)
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
                        if self.ready_valid:
                            self.wire(node_mux.ports.valid_in[idx],
                                      mux.ports.valid_out)
                            # TODO: FIX THIS
                            self.wire(node_mux.ports.valid_in[idx],
                                      Const(1))

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
        fanout = InclusiveNodeFanout.get(node)
        # fanout ports is I{i} S{i} E{i} and sel{i}
        for idx, n in enumerate(list(node)):
            # need to get the mux the node is connected to
            if not isinstance(n, SwitchBoxNode):
                assert isinstance(n, PortNode)
                assert len(n) == 0
                # this is a port input, i.e. CB
                # we assume it's properly connected from outside
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
                en = self.registers[str(n)].ports.O
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

        self.wire(fanout.ports.O[0], port)

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
                 ready_valid: bool = False):
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

        # create cb and switchbox
        self.cbs: Dict[str, CB] = {}
        self.sbs: Dict[int, SB] = {}
        # we only create cb if it's an input port, which doesn't have
        # graph neighbors
        for bit_width, tile in self.tiles.items():
            # connection box time
            for port_name, port_node in tile.ports.items():
                # input ports
                if len(port_node) == 0:
                    assert bit_width == port_node.width
                    # make sure that it has at least one connection
                    if len(port_node.get_conn_in()) == 0:
                        continue
                    # create a CB
                    port_ref = tile.get_port_ref(port_node.name)
                    cb = CB(port_node, config_addr_width, config_data_width,
                            double_buffer=self.double_buffer,
                            ready_valid=self.ready_valid)
                    self.wire(cb.ports.O, port_ref)
                    self.cbs[port_name] = cb
                else:
                    # output ports
                    assert len(port_node.get_conn_in()) == 0
                    assert bit_width == port_node.width

            # switch box time
            core_name = self.core.name() if self.core is not None else ""
            sb = SB(tile.switchbox, config_addr_width, config_data_width,
                    core_name, stall_signal_width=stall_signal_width,
                    double_buffer=self.double_buffer,
                    ready_valid=self.ready_valid)
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

                        # TODO: FIX BELOW
                        print(id(switchbox))
                        switchbox.wire(Const(0), mux.ports.valid_in)

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
                            port_name = create_name(str(node)) + "_valid"
                            self.wire(self.ports[port_name],
                                      cb.ports.valid_in[idx])
                    else:
                        self.wire(sb_circuit.ports[sb_name], cb.ports.I[idx])
                else:
                    # this is an additional core port
                    # just connect directly
                    width = node.width
                    tile = self.tiles[width]
                    self.wire(tile.get_port_ref(node.name), cb.ports.I[idx])
                    if self.ready_valid:
                        raise RuntimeError(
                            "port to port connection not "
                            "supported for ready-valid")

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
                    # TODO: add loopback module
                    sb_circuit.handle_node_fanin(port_node)
                    ready_in = port_node.name + "_ready"
                    ready_out = port_node.name + "_ready_out"
                    self.wire(sb_circuit.ports[ready_out],
                              self.core.ports[ready_in])

        # CB ready-valid
        if self.ready_valid:
            for bit_width, tile in self.tiles.items():
                sb_circuit = self.sbs[bit_width]
                for _, port_node in tile.ports.items():
                    if len(port_node) != 0:
                        continue
                    cb = self.cbs[port_node.name]
                    out_sel = port_node.name + "_out_sel"
                    enable = port_node.name + "_enable"
                    ready = port_node.name + "_ready"
                    self.wire(cb.ports.out_sel, sb_circuit.ports[out_sel])
                    self.wire(cb.ports.valid_out,
                              self.core.ports[port_node.name + "_valid"])
                    self.wire(cb.ports.ready_in,
                              self.core.ports[ready])
                    self.wire(cb.ports.enable, sb_circuit.ports[enable][0])
                    self.wire(cb.ports.ready_out, sb_circuit.ports[ready])

        self.__add_tile_id()
        # add ports
        self.add_ports(stall=magma.In(magma.Bits[stall_signal_width]),
                       reset=magma.In(magma.AsyncReset))

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
        stall_ports = []
        for feature in self.features():
            if "stall" in feature.ports.keys():
                stall_ports.append(feature.ports.stall)
        # some core may not expose the port as features, such as mem cores
        if self.core is not None and "stall" in self.core.ports and \
                self.core.ports.stall not in stall_ports:
            stall_ports.append(self.core.ports.stall)
        for stall_port in stall_ports:
            self.wire(self.ports.stall, stall_port)

    def __add_reset(self):
        # automatically add reset signal and connect it to the features if the
        # feature supports it
        reset_ports = []
        for feature in self.features():
            if "reset" in feature.ports.keys():
                reset_ports.append(feature.ports.reset)
        # some core may not expose the port as features, such as mem cores
        if self.core is not None and "reset" in self.core.ports and \
                self.core.ports.reset not in reset_ports:
            reset_ports.append(self.core.ports.reset)

        for reset_port in reset_ports:
            self.wire(self.ports.reset, reset_port)

    def __should_add_config(self):
        # a introspection on itself to determine whether to add config
        # or not
        for feature in self.features():
            if "config" in feature.ports:
                return True
            else:
                # if the feature doesn't have config port, it shouldn't have
                # reset either, although the other way around may be true
                # that is, a feature may have some internal states that need
                # to reset, but not necessarily has config port
                assert "reset" not in feature.ports
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

        # see if we really need to add config or not
        if not self.__should_add_config():
            return

        self.add_ports(
            config=magma.In(ConfigurationType(self.full_config_addr_width,
                                              self.config_data_width)),
            clk=magma.In(magma.Clock),
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

    def get_route_bitstream_config(self, src_node: Node, dst_node: Node):
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
        return reg_idx, feature_addr, config_data

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
                else:
                    self.add_port(port_name, magma.In(magma.Bits[bit_width]))
                    self.wire(self.ports[port_name], self.core.ports[port_name])
            # lift the output ports up
            for bt, port_name in self.core_interface.outputs():
                if bt != bit_width:
                    continue
                # depends on if the port has any connection or not
                # we lift the port up first
                # if it has connection, then we connect it to the core
                self.add_port(port_name, magma.Out(magma.Bits[bit_width]))
                self.wire(self.ports[port_name], self.core.ports[port_name])

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
        else:
            return self.output_ports[port_name][1]

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
