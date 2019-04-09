from canal.model import *
from gemstone.common.dummy_core_magma import DummyCore
from canal.util import *
import pytest
import tempfile
import filecmp
import os


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
    model = InterconnectModelCompiler(interconnect)
    new_interconnect = model.clone_graph()

    # they should match with everything

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
