from hwtypes import BitVector
from gemstone.common.dummy_core_magma import DummyCore, ReadyValidCore
from gemstone.common.testers import BasicTester
from gemstone.common.util import compress_config_data
from canal.checker import check_graph_isomorphic
from canal.pnr_io import route_one_tile
from canal.interconnect import *
import tempfile
import fault.random
from canal.util import create_uniform_interconnect, SwitchBoxType, IOSide
from canal.global_signal import apply_global_fanout_wiring, \
    apply_global_meso_wiring, apply_global_parallel_meso_wiring, \
    GlobalSignalWiring
from util import copy_sv_files
import pytest
import filecmp
import magma
import queue
import random


def assert_tile_coordinate(tile: Tile, x: int, y: int):
    assert tile.x == x and tile.y == y
    for sb in tile.switchbox.get_all_sbs():
        assert_coordinate(sb, x, y)
    for _, node in tile.ports.items():
        assert_coordinate(node, x, y)
    for _, node in tile.switchbox.registers.items():
        assert_coordinate(node, x, y)


def assert_coordinate(node: Node, x: int, y: int):
    assert node.x == x and node.y == y


def create_dummy_cgra(chip_size, num_tracks, reg_mode, wiring, num_cfg=1,
                      ready_valid=False):
    addr_width = 8
    data_width = 32
    bit_widths = [1, 16]
    tile_id_width = 16
    track_length = 1
    # creates all the cores here
    # we don't want duplicated cores when snapping into different interconnect
    # graphs
    cores = {}
    if ready_valid:
        core_type = ReadyValidCore
    else:
        core_type = DummyCore
    for x in range(chip_size):
        for y in range(chip_size):
            cores[(x, y)] = core_type()

    def create_core(xx: int, yy: int):
        return cores[(xx, yy)]

    in_conn = []
    out_conn = []
    for side in SwitchBoxSide:
        in_conn.append((side, SwitchBoxIO.SB_IN))
        out_conn.append((side, SwitchBoxIO.SB_OUT))
    pipeline_regs = []
    for track in range(num_tracks):
        for side in SwitchBoxSide:
            pipeline_regs.append((track, side))
    # if reg mode is off, reset to empty
    if not reg_mode:
        pipeline_regs = []
    ics = {}
    for bit_width in bit_widths:
        ic = create_uniform_interconnect(chip_size, chip_size, bit_width,
                                         create_core,
                                         {f"data_in_{bit_width}b": in_conn,
                                          f"data_out_{bit_width}b": out_conn,
                                          f"data_in_{bit_width}b_comb":
                                              in_conn},
                                         {track_length: num_tracks},
                                         SwitchBoxType.Disjoint,
                                         pipeline_regs)
        ics[bit_width] = ic
    interconnect = Interconnect(ics, addr_width, data_width, tile_id_width,
                                lift_ports=True, ready_valid=ready_valid)
    # finalize the design
    interconnect.finalize()
    # wiring
    if wiring == GlobalSignalWiring.Fanout:
        apply_global_fanout_wiring(interconnect)
    elif wiring == GlobalSignalWiring.Meso:
        apply_global_meso_wiring(interconnect)
    else:
        assert wiring == GlobalSignalWiring.ParallelMeso
        apply_global_parallel_meso_wiring(interconnect, num_cfg=num_cfg)

    return bit_widths, data_width, ics, interconnect


@pytest.mark.parametrize("num_tracks", [2, 4])
@pytest.mark.parametrize("chip_size", [2, 4])
@pytest.mark.parametrize("reg_mode", [True, False])
@pytest.mark.parametrize("wiring", [GlobalSignalWiring.Fanout,
                                    GlobalSignalWiring.Meso])
def test_interconnect(num_tracks: int, chip_size: int,
                      reg_mode: bool,
                      wiring: GlobalSignalWiring):
    bit_widths, data_width, ics, interconnect = create_dummy_cgra(chip_size,
                                                                  num_tracks,
                                                                  reg_mode,
                                                                  wiring)

    # assert tile coordinates
    for (x, y), tile_circuit in interconnect.tile_circuits.items():
        for _, tile in tile_circuit.tiles.items():
            assert_tile_coordinate(tile, x, y)

    circuit = interconnect.circuit()
    tester = BasicTester(circuit, circuit.clk, circuit.reset)
    config_data = []
    test_data = []

    # we have a 2x2 (for instance) chip as follows
    # |-------|
    # | A | B |
    # |---|---|
    # | C | D |
    # ---------
    # we need to test all the line-tile routes. that is, per row and per column
    # A <-> B, A <-> C, B <-> D, C <-> D
    # TODO: add core input/output as well

    vertical_conns = [[SwitchBoxSide.NORTH, SwitchBoxIO.SB_IN,
                       SwitchBoxSide.SOUTH, SwitchBoxIO.SB_OUT,
                       0, chip_size - 1],
                      [SwitchBoxSide.SOUTH, SwitchBoxIO.SB_IN,
                       SwitchBoxSide.NORTH, SwitchBoxIO.SB_OUT,
                       chip_size - 1, 0]]

    horizontal_conns = [[SwitchBoxSide.WEST, SwitchBoxIO.SB_IN,
                         SwitchBoxSide.EAST, SwitchBoxIO.SB_OUT,
                         0, chip_size - 1],
                        [SwitchBoxSide.EAST, SwitchBoxIO.SB_IN,
                         SwitchBoxSide.WEST, SwitchBoxIO.SB_OUT,
                         chip_size - 1, 0]]

    for bit_width in bit_widths:
        for track in range(num_tracks):
            # vertical
            for x in range(chip_size):
                for side_io in vertical_conns:
                    from_side, from_io, to_side, to_io, start, end = side_io
                    src_node = None
                    dst_node = None
                    config_entry = []
                    for y in range(chip_size):
                        tile_circuit = interconnect.tile_circuits[(x, y)]
                        tile = tile_circuit.tiles[bit_width]
                        pre_node = tile.get_sb(from_side, track, from_io)
                        tile_circuit = interconnect.tile_circuits[(x, y)]
                        tile = tile_circuit.tiles[bit_width]
                        next_node = tile.get_sb(to_side, track, to_io)
                        if y == start:
                            src_node = pre_node
                        if y == end:
                            dst_node = next_node

                        entry = \
                            interconnect.get_node_bitstream_config(pre_node,
                                                                   next_node)

                        config_entry.append(entry)
                    config_entry = compress_config_data(config_entry)
                    assert src_node is not None and dst_node is not None
                    config_data.append(config_entry)
                    value = fault.random.random_bv(bit_width)
                    src_name = interconnect.get_top_port_name(src_node)
                    dst_name = interconnect.get_top_port_name(dst_node)
                    test_data.append((circuit.interface[src_name],
                                      circuit.interface[dst_name],
                                      value))

            # horizontal connections
            for y in range(chip_size):
                for side_io in horizontal_conns:
                    from_side, from_io, to_side, to_io, start, end = side_io
                    src_node = None
                    dst_node = None
                    config_entry = []
                    for x in range(chip_size):
                        tile_circuit = interconnect.tile_circuits[(x, y)]
                        tile = tile_circuit.tiles[bit_width]
                        pre_node = tile.get_sb(from_side, track, from_io)
                        tile_circuit = interconnect.tile_circuits[(x, y)]
                        tile = tile_circuit.tiles[bit_width]
                        next_node = tile.get_sb(to_side, track, to_io)
                        if x == start:
                            src_node = pre_node
                        if x == end:
                            dst_node = next_node

                        entry = \
                            interconnect.get_node_bitstream_config(pre_node,
                                                                   next_node)

                        config_entry.append(entry)
                    config_entry = compress_config_data(config_entry)
                    assert src_node is not None and dst_node is not None
                    config_data.append(config_entry)
                    value = fault.random.random_bv(bit_width)
                    src_name = interconnect.get_top_port_name(src_node)
                    dst_name = interconnect.get_top_port_name(dst_node)
                    test_data.append((circuit.interface[src_name],
                                      circuit.interface[dst_name],
                                      value))

    # the actual test
    assert len(config_data) == len(test_data)
    # NOTE:
    # we don't test the configuration read here
    for i in range(len(config_data)):
        tester.reset()
        input_port, output_port, value = test_data[i]
        for addr, index in config_data[i]:
            tester.configure(BitVector[data_width](addr), index)
            tester.configure(BitVector[data_width](addr), index + 1, False)
            tester.config_read(addr)
            tester.eval()
            tester.expect(circuit.read_config_data, index)

        tester.poke(input_port, value)
        tester.eval()
        tester.expect(output_port, value)

    with tempfile.TemporaryDirectory() as tempdir:
        copy_sv_files(tempdir)
        tester.compile_and_run(target="verilator",
                               magma_output="coreir-verilog",
                               directory=tempdir,
                               flags=["-Wno-fatal"])

    # also test the interconnect clone
    new_interconnect = interconnect.clone()
    with tempfile.TemporaryDirectory() as tempdir_old:
        interconnect.dump_pnr(tempdir_old, "old")
        with tempfile.TemporaryDirectory() as tempdir_new:
            new_interconnect.dump_pnr(tempdir_new, "new")

            # they should be exactly the same
            graph1_old = os.path.join(tempdir_old, "1.graph")
            graph1_new = os.path.join(tempdir_new, "1.graph")
            assert filecmp.cmp(graph1_old, graph1_new)

            graph16_old = os.path.join(tempdir_old, "16.graph")
            graph16_new = os.path.join(tempdir_new, "16.graph")
            assert filecmp.cmp(graph16_old, graph16_new)

            layout_old = os.path.join(tempdir_old, "old.layout")
            layout_new = os.path.join(tempdir_new, "new.layout")
            assert filecmp.cmp(layout_old, layout_new)

    # check the construction of graph
    # with tempfile.TemporaryDirectory() as tempdir:
    #     rtl_path = os.path.join(tempdir, "rtl")
    #     magma.compile(rtl_path, circuit, output="coreir")
    #     rtl_path += ".json"
    #     check_graph_isomorphic(ics, rtl_path)


@pytest.mark.parametrize("double_buffer", [True, False])
def test_1x1(double_buffer):
    ics = {}
    in_conn = []
    out_conn = []
    addr_width = 8
    data_width = 32
    for side in SwitchBoxSide:
        in_conn.append((side, SwitchBoxIO.SB_IN))
        out_conn.append((side, SwitchBoxIO.SB_OUT))

    core = DummyCore()

    for bit_width in {1, 16}:
        ic = create_uniform_interconnect(1, 1, bit_width,
                                         lambda _, __: core,
                                         {f"data_in_{bit_width}b": in_conn,
                                          f"data_out_{bit_width}b": out_conn},
                                         {1: 2},
                                         SwitchBoxType.Disjoint)
        ics[bit_width] = ic
    interconnect = Interconnect(ics, addr_width, data_width, 16,
                                lift_ports=True, double_buffer=double_buffer)
    interconnect.finalize()
    apply_global_fanout_wiring(interconnect)
    circuit = interconnect.circuit()
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "test1x1")
        magma.compile(filename, circuit, output="coreir-verilog")
    # test routing for 1x1
    compares = {}
    for seed in {0, 1}:
        routing_result, _ = route_one_tile(interconnect, 0, 0,
                                           ports=["data_in_16b",
                                                  "data_out_16b"],
                                           seed=seed)
        # routing result ordering is the same as ports
        assert len(routing_result) == 2
        bs = interconnect.get_route_bitstream(routing_result)
        assert len(bs) > 0
        compares[seed] = bs
    for i in range(2):
        assert compares[0][i] != compares[1][i]


def test_dump_pnr():
    num_tracks = 2
    addr_width = 8
    data_width = 32
    bit_widths = [1, 16]

    tile_id_width = 16

    chip_size = 2
    track_length = 1

    # creates all the cores here
    # we don't want duplicated cores when snapping into different interconnect
    # graphs
    cores = {}
    for x in range(chip_size):
        for y in range(chip_size):
            cores[(x, y)] = DummyCore()

    def create_core(xx: int, yy: int):
        return cores[(xx, yy)]

    in_conn = []
    out_conn = []
    for side in SwitchBoxSide:
        in_conn.append((side, SwitchBoxIO.SB_IN))
        out_conn.append((side, SwitchBoxIO.SB_OUT))

    pipeline_regs = []
    for track in range(num_tracks):
        for side in SwitchBoxSide:
            pipeline_regs.append((track, side))

    ics = {}
    for bit_width in bit_widths:
        ic = create_uniform_interconnect(chip_size, chip_size, bit_width,
                                         create_core,
                                         {f"data_in_{bit_width}b": in_conn,
                                          f"data_out_{bit_width}b": out_conn},
                                         {track_length: num_tracks},
                                         SwitchBoxType.Disjoint,
                                         pipeline_reg=pipeline_regs)
        ics[bit_width] = ic

    interconnect = Interconnect(ics, addr_width, data_width, tile_id_width,
                                lift_ports=True)

    design_name = "test"
    with tempfile.TemporaryDirectory() as tempdir:
        interconnect.dump_pnr(tempdir, design_name)

        assert os.path.isfile(os.path.join(tempdir, f"{design_name}.info"))
        assert os.path.isfile(os.path.join(tempdir, "1.graph"))
        assert os.path.isfile(os.path.join(tempdir, "16.graph"))
        assert os.path.isfile(os.path.join(tempdir, f"{design_name}.layout"))


@pytest.mark.parametrize("num_cfg", [1, 2, 4])
def test_parallel_meso_wiring(num_cfg: int):
    _, _, _, interconnect = create_dummy_cgra(2,
                                              2,
                                              True,
                                              GlobalSignalWiring.ParallelMeso,
                                              num_cfg)
    # assert tile coordinates
    circuit = interconnect.circuit()

    # just check it compiles to rtl
    with tempfile.TemporaryDirectory() as tempdir:
        rtl_path = os.path.join(tempdir, "rtl")
        magma.compile(rtl_path, circuit, output="coreir-verilog")


def test_ready_valid_verilog():
    _, _, _, interconnect = create_dummy_cgra(2,
                                              2,
                                              True,
                                              GlobalSignalWiring.ParallelMeso,
                                              ready_valid=True)
    # assert tile coordinates
    circuit = interconnect.circuit()

    # just check it compiles to rtl
    with tempfile.TemporaryDirectory() as tempdir:
        rtl_path = os.path.join(tempdir, "rtl")
        magma.compile(rtl_path, circuit, output="coreir-verilog")


def test_skip_addr():
    _, _, _, interconnect = create_dummy_cgra(2, 2, False,
                                              GlobalSignalWiring.ParallelMeso)
    # set 1, 1 to be ignored addr
    interconnect.tile_circuits[(1, 1)].core.skip_compression = True
    skip_addrs = interconnect.get_skip_addr()
    assert len(skip_addrs) == 256


def test_ready_valid_fifo():
    chip_size = 2
    _, _, _, interconnect = create_dummy_cgra(chip_size,
                                              1,
                                              True,
                                              GlobalSignalWiring.Fanout,
                                              ready_valid=True)

    # we have a 2x1 (for instance) chip as follows
    # |-------|
    # | A | B |
    # |---|---|
    # we need to test from A -> B with FIFO enabled
    track = 0
    route = []
    for x in range(chip_size):
        tile_circuit = interconnect.tile_circuits[(x, 0)]
        # only interested in 16 bit
        tile = tile_circuit.tiles[16]
        node = tile.get_sb(SwitchBoxSide.WEST, track, SwitchBoxIO.SB_IN)
        route.append(node)
        node = tile.get_sb(SwitchBoxSide.EAST, track, SwitchBoxIO.SB_OUT)
        route.append(node)
        # the register mux
        node = tile.switchbox.get_register(SwitchBoxSide.EAST, track)
        route.append(node)
        node = tile.switchbox.get_reg_mux(SwitchBoxSide.EAST, track)
        route.append(node)

    config_data = interconnect.get_route_bitstream({"e1": [route]},
                                                   use_fifo=True)
    config_data = compress_config_data(config_data)
    circuit = interconnect.circuit()
    tester = BasicTester(circuit, circuit.clk, circuit.reset)

    tester.reset()

    for addr, data in config_data:
        tester.configure(addr, data)

    src_node = interconnect.tile_circuits[(0, 0)].tiles[16].get_sb(
        SwitchBoxSide.WEST, track, SwitchBoxIO.SB_IN)
    dst_node = interconnect.tile_circuits[(1, 0)].tiles[16].get_sb(
        SwitchBoxSide.EAST, track, SwitchBoxIO.SB_OUT)

    src_name = str(src_node) + "_X{0:X}_Y{1:X}".format(0, 0)
    dst_name = str(dst_node) + "_X{0:X}_Y{1:X}".format(1, 0)
    ready_in = dst_name + "_ready"
    ready_out = src_name + "_ready"
    valid_in = src_name + "_valid"
    valid_out = dst_name + "_valid"

    # test out holding values
    tester.poke(circuit.interface[valid_in], 1)
    tester.eval()
    test_inputs = list(range(42, 48))

    # the output should be ready
    tester.expect(circuit.interface[ready_out], 1)

    for v in test_inputs:
        tester.poke(circuit.interface[src_name], v)
        tester.step(2)

    tester.eval()
    # output should not be ready anymore
    tester.expect(circuit.interface[ready_out], 0)

    # push in more data into the fifo
    new_value = 10
    tester.poke(circuit.interface[ready_in], 1)
    tester.poke(circuit.interface[src_name], new_value)
    tester.eval()
    # should be valid
    tester.expect(circuit.interface[valid_out], 1)
    tester.expect(circuit.interface[dst_name], test_inputs[0])

    tester.step(2)
    tester.expect(circuit.interface[valid_out], 1)
    tester.expect(circuit.interface[dst_name], test_inputs[1])

    tester.step(2)
    tester.expect(circuit.interface[valid_out], 1)
    tester.expect(circuit.interface[dst_name], new_value)

    with tempfile.TemporaryDirectory() as tempdir:
        copy_sv_files(tempdir)
        tester.compile_and_run(target="verilator",
                               magma_output="coreir-verilog",
                               directory=tempdir,
                               flags=["-Wno-fatal"])


def __route(src_node: Node, dst_node: None):
    path_mapping = {}
    visited = set()
    q = queue.Queue()
    q.put(src_node)

    while not q.empty():
        node = q.get()
        if node in visited:
            continue
        visited.add(node)

        if node == dst_node:
            # found it
            break

        for n in node:
            if n in visited or n in path_mapping:
                continue
            path_mapping[n] = node

            q.put(n)

    assert dst_node in path_mapping
    path = [dst_node]
    while path[-1] in path_mapping:
        node = path[-1]
        next_node = path_mapping[node]
        path.append(next_node)

    path = path[::-1]
    assert path[0] == src_node
    return path


def insert_fifo(path, interconnect: Interconnect):
    result = []
    start: RegisterNode = RegisterNode("", 0, 0, 0, 0)
    end: RegisterNode = RegisterNode("", 0, 0, 0, 0)
    idx = 0
    count = 0

    for idx, node in enumerate(path):
        if isinstance(node, RegisterMuxNode):
            reg = interconnect.tile_circuits[(node.x, node.y)].sbs[node.width].switchbox.get_register(node.side,
                                                                                                      node.track)
            result.append(reg)
            if count == 0:
                start = reg
            elif count == 1:
                end = reg
            count += 1
        result.append(node)
        if count == 2:
            idx += 1
            break

    return (start, end), result, path[idx:]


class RefValue:
    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return int(self.value)


class SingleFifo:
    def __init__(self, id_, capacity, ready_in: RefValue, valid_in: RefValue):
        self.id_ = id_
        self.capacity = capacity
        self.fifo = []
        self.ready_out = RefValue(True)
        self.valid_out = RefValue(False)
        self.ready_in = ready_in
        self.valid_in = valid_in
        self.in_ = RefValue(0)
        self.out = RefValue(0)

    def __len__(self):
        return len(self.fifo)

    def eval(self):
        if self.valid_in:
            if len(self) < self.capacity:
                self.fifo.append(self.in_.value)
            self.out.value = self.fifo[0]
            if self.ready_in:
                self.fifo = self.fifo[1:]
            self.valid_out.value = True
        else:
            if len(self.fifo) > 0:
                self.out.value = self.fifo[0]
                self.valid_out.value = True
                if self.ready_in.value:
                    self.fifo = self.fifo[1:]
            else:
                self.valid_out.value = False

        self.ready_out.value = len(self.fifo) < self.capacity


class FifoModel:
    def __init__(self, num_fifo, capacity=2):
        self.valid_in = RefValue(True)
        self.ready_in = RefValue(True)
        self.in_ = RefValue(0)
        self.fifos = []
        for i in range(num_fifo):
            ref = RefValue(0)
            self.fifos.append(SingleFifo(i, capacity, ref, ref))
        # fix connections
        self.fifos[-1].ready_in = self.ready_in
        self.fifos[0].valid_in = self.valid_in
        self.fifos[0].in_ = self.in_

        for i in range(num_fifo - 1):
            pre_fifo = self.fifos[i]
            next_fifo = self.fifos[i + 1]
            pre_fifo.out = next_fifo.in_
            pre_fifo.valid_out = next_fifo.valid_in
            pre_fifo.ready_in = next_fifo.ready_out

        # registered out
        self.out = self.fifos[-1].out
        self.ready_out = self.fifos[0].ready_out
        self.valid_out = self.fifos[-1].valid_out

    def eval(self):
        print("RI:", bool(self.ready_in.value), "VI", bool(self.valid_in.value),
              "Value", self.in_.value)
        for fifo in self.fifos[::-1]:
            fifo.eval()


def test_fifo_model():
    fifo = FifoModel(2)
    ri = fifo.ready_in
    vi = fifo.valid_in
    in_ = fifo.in_

    ri.value = True
    vi.value = False
    in_.value = 72
    fifo.eval()
    in_.value = 48
    fifo.eval()
    ri.value = False
    vi.value = True
    in_.value = 37
    fifo.eval()
    assert not fifo.valid_out.value
    in_.value = 22
    fifo.eval()
    assert fifo.valid_out.value
    assert fifo.out.value == 37
    in_.value = 78
    fifo.eval()

    in_.value = 28
    vi.value = False
    fifo.eval()

    ri.value = True
    in_.value = 97
    fifo.eval()
    assert fifo.valid_out.value
    assert fifo.out.value == 37

    in_.value = 55
    fifo.eval()
    assert fifo.valid_out.value
    assert fifo.out.value == 22

    in_.value = 36
    fifo.eval()
    assert fifo.valid_out.value
    assert fifo.out.value == 78

    in_.value = 76
    vi.value = True
    ri.value = False
    fifo.eval()
    assert not fifo.valid_out.value


def test_ready_valid_randomized():
    chip_size = 4
    _, _, _, interconnect = create_dummy_cgra(chip_size,
                                              2,
                                              True,
                                              GlobalSignalWiring.Fanout,
                                              ready_valid=True)

    rnd = random.Random(0)
    start_x = rnd.randint(0, chip_size)
    start_y = 0
    end_x = rnd.randint(0, chip_size)
    end_y = chip_size - 1
    src_node = interconnect.tile_circuits[(start_x,
                                           start_y)].sbs[16].switchbox.get_sb(SwitchBoxSide.NORTH, 0,
                                                                              SwitchBoxIO.SB_IN)
    dst_node = interconnect.tile_circuits[(end_x,
                                           end_y)].sbs[16].switchbox.get_reg_mux(SwitchBoxSide.SOUTH, 0)

    path = __route(src_node, dst_node)
    final_path = []
    fifo = []
    for i in range(chip_size // 2):
        f, r, path = insert_fifo(path, interconnect)
        fifo.append(f)
        final_path += r
    config_data = interconnect.get_route_bitstream({"e1": [final_path]}, use_fifo=True)
    config_data += interconnect.set_fifo_mode(fifo[0][-1], False, True)
    config_data += interconnect.set_fifo_mode(fifo[1][0], True, False)

    dst_node = interconnect.tile_circuits[(end_x,
                                           end_y)].sbs[16].switchbox.get_sb(SwitchBoxSide.SOUTH, 0, SwitchBoxIO.SB_OUT)

    src_name = str(src_node) + "_X{0:X}_Y{1:X}".format(src_node.x, src_node.y)
    dst_name = str(dst_node) + "_X{0:X}_Y{1:X}".format(dst_node.x, dst_node.y)

    print("Input", src_name)
    print("Output", dst_name)

    config_data = compress_config_data(config_data)
    circuit = interconnect.circuit()
    tester = BasicTester(circuit, circuit.clk, circuit.reset)

    tester.reset()

    for addr, data in config_data:
        tester.configure(addr, data)

    ready_in = dst_name + "_ready"
    ready_out = src_name + "_ready"
    valid_in = src_name + "_valid"
    valid_out = dst_name + "_valid"

    model = FifoModel(chip_size // 2)

    values = []
    for i in range(10):
        ri = int(rnd.random() < 0.5)
        vi = int(rnd.random() < 0.5)
        tester.poke(circuit.interface[ready_in], ri)
        tester.poke(circuit.interface[valid_in], vi)
        model.ready_in.value = ri
        model.valid_in.value = vi

        tester.eval()
        value = rnd.randrange(10, 100)
        tester.poke(circuit.interface[src_name], value)
        model.in_.value = value
        tester.eval()
        model.eval()

        tester.step(2)

        valid = int(model.valid_out.value)
        ready = int(model.ready_in.value)
        tester.expect(circuit.interface[valid_out], valid)
        if valid:
            value = model.out.value
            values.append(value)
            tester.expect(circuit.interface[dst_name], value)


    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = "tempdir"
        copy_sv_files(tempdir)
        tester.compile_and_run(target="verilator",
                               magma_output="coreir-verilog",
                               directory=tempdir,
                               flags=["-Wno-fatal", "--trace"])


if __name__ == "__main__":
    test_ready_valid_randomized()
