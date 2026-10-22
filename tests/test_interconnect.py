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
import re
import magma
from gemstone.generator.generator import clear_generator_cache


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
                      ready_valid=False, fifo_only_regs=False,
                      input_fifo_ports=None, inter_core_connection=None):
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
                                         pipeline_regs,
                                         inter_core_connection=(inter_core_connection or {}).get(bit_width))
        ics[bit_width] = ic
    interconnect = Interconnect(ics, addr_width, data_width, tile_id_width,
                                lift_ports=True, ready_valid=ready_valid,
                                fifo_only_regs=fifo_only_regs,
                                input_fifo_ports=input_fifo_ports)
    if ready_valid:
        # ready-valid switch-box registers take the tile's flush (garnet gives each
        # tile one); without it finalize() fails with KeyError 'flush'
        interconnect.add_port("flush", magma.In(magma.Bits[1]))
        for tile_circuit in interconnect.tile_circuits.values():
            tile_circuit.add_port("flush", magma.In(magma.Bits[1]))
            interconnect.wire(interconnect.ports.flush, tile_circuit.ports.flush)
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


@pytest.mark.parametrize("fifo_only", [False, True])
def test_fifo_only_regs(monkeypatch, fifo_only):
    """fifo_only_regs: switch-box registers are always FIFOs. No `<reg>_fifo` config
    bit and no fifo_en port (the pipeline-register mode, which drops tokens on a
    ready-valid route, is gone); the route bitstream sets only bogus_init. Off by
    default, so other fabrics keep the mode."""
    monkeypatch.setenv("USE_NON_SPLIT_FIFOS", "1")
    clear_generator_cache()   # tiles are cached by name, not by content
    _, _, _, interconnect = create_dummy_cgra(2, 1, True, GlobalSignalWiring.Fanout,
                                              ready_valid=True,
                                              fifo_only_regs=fifo_only)
    tile_circuit = interconnect.tile_circuits[(0, 0)]
    sb = tile_circuit.sbs[16]
    assert sb.regs
    names = list(sb.registers.keys())
    assert any(n.endswith("_bogus_init") for n in names)
    assert any(n.endswith("_fifo") for n in names) != fifo_only
    for _, reg in sb.regs.values():
        assert ("fifo_en" in reg.ports) != fifo_only

    reg_node = tile_circuit.tiles[16].switchbox.get_register(SwitchBoxSide.EAST, 0)
    configs = tile_circuit.configure_fifo(reg_node, False, False, use_non_split_fifos=True)
    assert len(configs) == (1 if fifo_only else 2)   # (fifo mode +) bogus_init

    with tempfile.TemporaryDirectory() as tempdir:
        rtl_path = os.path.join(tempdir, "rtl")
        magma.compile(rtl_path, interconnect.circuit(), output="coreir-verilog")
        with open(rtl_path + ".v") as f:
            rtl = f.read()
    assert ("module reg_fifo_d_2_w_1_D_16_fifo_only" in rtl) == fifo_only
    assert ("fifo_en" in rtl) != fifo_only


def test_input_fifo_ports(monkeypatch):
    """input_fifo_ports: the named core input's CB gets a 2-deep FIFO between its
    switch-box sources and the core; an inter-core source bypasses it, chosen by the
    CB's own select (no new config bits, so route bitstreams are unchanged). Off by
    default. The dummy core's output feeds its own input as the inter-core source."""
    monkeypatch.setenv("USE_NON_SPLIT_FIFOS", "1")
    port = "data_in_16b"
    inter_core = {16: {"data_out_16b": [port]}}
    built = {}
    for on in (False, True):
        # tiles are cached by name, not by content, here and in coreir's context
        clear_generator_cache()
        magma.frontend.coreir_.ResetCoreIR()
        _, _, _, ic = create_dummy_cgra(2, 1, True, GlobalSignalWiring.Fanout,
                                        ready_valid=True, fifo_only_regs=True,
                                        input_fifo_ports=[port] if on else None,
                                        inter_core_connection=inter_core)
        tile_circuit = ic.tile_circuits[(0, 0)]
        cb = tile_circuit.cbs[port]
        assert cb.input_fifo == on
        assert ("flush" in cb.ports and "stall" in cb.ports) == on
        assert cb.name().endswith("_input_fifo") == on
        assert cb.instance_name == "CB_" + port
        # no other CB has one
        assert not any(c.input_fifo for name, c in tile_circuit.cbs.items() if name != port)

        tile = tile_circuit.tiles[16]
        node = tile.ports[port]
        conn_in = node.get_conn_in()
        loop_src = tile.ports["data_out_16b"]
        assert loop_src in conn_in   # the inter-core source
        net_src = next(n for n in conn_in if isinstance(n, SwitchBoxNode))
        routes = {"net": [[net_src, node]], "loop": [[loop_src, node]]}
        bitstream = ic.get_route_bitstream(routes, use_fifo=True)

        with tempfile.TemporaryDirectory() as tempdir:
            rtl_path = os.path.join(tempdir, "rtl")
            magma.compile(rtl_path, ic.circuit(), output="coreir-verilog")
            with open(rtl_path + ".v") as f:
                rtl = f.read()
        built[on] = (dict((n, r.width) for n, r in cb.registers.items()), bitstream, rtl)

    regs_off, bs_off, rtl_off = built[False]
    regs_on, bs_on, rtl_on = built[True]
    assert regs_on == regs_off   # same config registers: select + enable only
    assert bs_on == bs_off
    assert len(bs_on) == 4       # select + enable for each route
    assert "InputFifoSteer" not in rtl_off
    assert "module CB_data_in_16b_input_fifo" in rtl_on
    assert "module InputFifoSteer_W16" in rtl_on
    # the inter-core source's select steers around the FIFO
    loop_idx = conn_in.index(loop_src)
    assert re.search(rf"\.loop\(\w*out_sel\[{loop_idx}\]\)", rtl_on)
    assert f".loop_data(I[{loop_idx}])" in rtl_on


@pytest.mark.parametrize("bad", ["comb", "static"])
def test_input_fifo_ports_needs_ready_valid(monkeypatch, bad):
    monkeypatch.setenv("USE_NON_SPLIT_FIFOS", "1")
    clear_generator_cache()
    with pytest.raises(AssertionError):
        if bad == "comb":   # a combinational input has no ready-valid
            create_dummy_cgra(2, 1, True, GlobalSignalWiring.Fanout, ready_valid=True,
                              input_fifo_ports=["data_in_16b_comb"])
        else:
            create_dummy_cgra(2, 1, True, GlobalSignalWiring.Fanout,
                              input_fifo_ports=["data_in_16b"])


if __name__ == "__main__":
    test_ready_valid_fifo()
