import magma
import _kratos
import kratos
import math
import mantle
from kratos import Generator, posedge, negedge, always_comb, always_ff, clog2, \
    const
from gemstone.generator.generator import Generator as GemstoneGenerator
from gemstone.generator.from_magma import FromMagma
from .cyclone import Node, create_name


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

        super().__init__(f"reg_fifo_d_{depth}_w_{width_mult}_D_{data_width}")

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

        # control whether to function as a pipeline register or not
        self._fifo_en = self.input("fifo_en", 1)

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

        # based on enable semantics. if fifo mode is disabled, we always enable push
        # and pop to match with pipeline register behavior
        self._push_en = self.var("push_en", 1)
        self._pop_en = self.var("pop_en", 1)
        self.wire(self._push_en, kratos.ternary(self._fifo_en, self._push, const(1, 1)))
        self.wire(self._pop_en, kratos.ternary(self._fifo_en, self._pop, const(1, 1)))

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

        self.wire(self._read, self._pop_en & ~self._passthru & ~self._empty)

        # Disallow passthru for now to prevent combinational loops
        self.wire(self._passthru, const(0, 1))

        # Should only write

        # Boilerplate Add always @(posedge clk, ...) blocks
        if self.parallel:
            self.add_code(self.set_num_items_parallel)
            self.add_code(self.reg_array_ff_parallel)
            self.add_code(self.wr_ptr_ff_parallel)
            self.add_code(self.rd_ptr_ff_parallel)
            self.wire(self._parallel_out, self._reg_array)
            self.wire(self._write,
                      self._push_en & ~self._passthru & (
                              ~self._full | (self._parallel_read)))
        else:
            # Don't want to write when full at all for decoupling
            self.wire(self._write, self._push_en & ~self._passthru & (~self._full))
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
            ready_out=magma.Out(magma.Bit),
            fifo_en=magma.In(magma.Bit)
        )

        self.wire(self.ports.I, self.__circuit.ports.data_in)
        self.wire(self.ports.O, self.__circuit.ports.data_out)
        self.wire(self.ports.clk, self.__circuit.ports.clk)
        self.wire(self.ports.CE, self.__circuit.ports.clk_en[0])
        self.wire(self.ports.fifo_en, self.__circuit.ports.fifo_en[0])
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

        self.wire(vo, ri & vi)

    @staticmethod
    def get():
        if ReadyValidLoopBack.__cache is None:
            inst = ReadyValidLoopBack()
            circuit = kratos.util.to_magma(inst)
            ReadyValidLoopBack.__cache = circuit
        circuit = ReadyValidLoopBack.__cache
        return FromMagma(circuit)

