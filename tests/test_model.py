from canal.model import *
from gemstone.common.dummy_core_magma import DummyCore
from canal.util import *
import pytest
import random


@pytest.mark.parametrize("num_tracks", [2, 4])
@pytest.mark.parametrize("chip_size", [2, 4])
@pytest.mark.parametrize("reg_mode", [True, False])
def test_clone(num_tracks: int, chip_size: int,
               reg_mode: bool):
    addr_width = 8
    data_width = 32
    bit_widths = [1, 16]

    tile_id_width = 16

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
    # if reg mode is off, reset to empty
    if not reg_mode:
        pipeline_regs = []
    ics = {}
    for bit_width in bit_widths:
        ic = create_uniform_interconnect(chip_size, chip_size, bit_width,
                                         create_core,
                                         {f"data_in_{bit_width}b": in_conn,
                                          f"data_out_{bit_width}b": out_conn},
                                         {track_length: num_tracks},
                                         SwitchBoxType.Disjoint,
                                         pipeline_regs)
        ics[bit_width] = ic

    interconnect = Interconnect(ics, addr_width, data_width, tile_id_width,
                                lift_ports=True)

    # random path
    # no loop
    rnd = random.Random(0)
    interface = interconnect.interface()

    def construct_path():
        start_nodes = [interface[name] for name in interface if
                       len(interface[name]) > 0]
        end_nodes = [interface[name] for name in interface if
                     len(interface[name]) == 0]
        start_node = rnd.choice(start_nodes)
        end_node = rnd.choice(end_nodes)
        link = {}

        def construct_path(node):
            if node == end_node:
                return
            else:
                for node_ in node:
                    link[node_] = node

    compiler = InterconnectModelCompiler(interconnect)
    model = compiler.compile()
