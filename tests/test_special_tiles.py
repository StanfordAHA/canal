from bit_vector import BitVector
from gemstone.common.dummy_core_magma import DummyCore
from gemstone.common.testers import BasicTester
from canal.cyclone import *
from canal.circuit import *
import tempfile
import fault
import fault.random
import magma
import os


def test_empty_tile():
    bit_width = 1
    core = CoreInterface(None)
    tile = Tile(0, 0, bit_width, SwitchBox(0, 0, 0, bit_width, []))
    tile.set_core(core)
    tile_circuit = TileCircuit({bit_width: tile}, 8, 32)
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
    # also need to ground the 16 bit
    tile_circuit.wire(Const(0), tile_circuit.core.ports["data_in_16b"])
    circuit = tile_circuit.circuit()
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "tile")
        magma.compile(filename, circuit, output="coreir-verilog")


if __name__ == "__main__":
    test_empty_switch_box()