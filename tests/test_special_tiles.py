from gemstone.common.dummy_core_magma import DummyCore
from canal.circuit import *
from canal.util import *
from canal.global_signal import *
import tempfile
import magma
import os


def test_empty_tile():
    bit_width = 1
    core = CoreInterface(None)
    tile = Tile(0, 0, bit_width, SwitchBox(0, 0, 0, bit_width, []))
    tile.set_core(core)
    tile_circuit = TileCircuit({bit_width: tile}, 8, 32)
    tile_circuit.finalize()
    circuit = tile_circuit.circuit()
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "tile")
        magma.compile(filename, circuit, output="coreir-verilog")


def test_empty_switch_box():
    bit_width = 1
    core = CoreInterface(DummyCore())
    tile = Tile(0, 0, bit_width, SwitchBox(0, 0, 0, bit_width, []))
    tile.set_core(core)
    # because we need something to be connected to the core node
    # otherwise it's an illogical tile
    sb_node = SwitchBoxNode(0, 1, 0, bit_width, SwitchBoxSide.NORTH,
                            SwitchBoxIO.SB_IN)
    sb_node.add_edge(tile.ports["data_in_1b"])

    tile_circuit = TileCircuit({bit_width: tile}, 8, 32)
    tile_circuit.finalize()
    # also need to ground the 16 bit
    tile_circuit.wire(Const(0), tile_circuit.core.ports["data_in_16b"])
    circuit = tile_circuit.circuit()
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "tile")
        magma.compile(filename, circuit, output="coreir-verilog")


def test_empty_tile_util():
    chip_size = 4
    margin = 1
    sides = IOSide.North | IOSide.East | IOSide.South | IOSide.West
    bit_widths = [1, 16]
    cores = {}
    track_length = 1
    num_tracks = 2
    addr_width = 8
    data_width = 32
    tile_id_width = 16
    for x in range(chip_size + 2 * margin):
        for y in range(chip_size + 2 * margin):
            if x in range(margin) or \
                    x in range(chip_size - margin, chip_size) or \
                    y in range(margin) or \
                    y in range(chip_size - margin, chip_size):
                cores[(x, y)] = None
            else:
                cores[(x, y)] = DummyCore()

    def core_fn(x, y):
        return cores[(x, y)]

    in_conn = []
    out_conn = []
    for side in SwitchBoxSide:
        in_conn.append((side, SwitchBoxIO.SB_IN))
        out_conn.append((side, SwitchBoxIO.SB_OUT))

    ics = {}
    for bit_width in bit_widths:
        ic = create_uniform_interconnect(chip_size, chip_size, bit_width,
                                         core_fn,
                                         {f"data_in_{bit_width}b": in_conn,
                                          f"data_out_{bit_width}b": out_conn},
                                         {track_length: num_tracks},
                                         SwitchBoxType.Disjoint,
                                         io_sides=sides)
        ics[bit_width] = ic
    interconnect = Interconnect(ics, addr_width, data_width, tile_id_width)
    interconnect.finalize()
    # wiring
    apply_global_meso_wiring(interconnect)

    circuit = interconnect.circuit()
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "interconnect")
        magma.compile(filename, circuit, output="coreir-verilog")
