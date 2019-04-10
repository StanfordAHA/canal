from canal.model import *
from gemstone.common.dummy_core_magma import DummyCore
from canal.util import *


def test_simulation():
    addr_width = 8
    data_width = 32
    bit_widths = [1, 16]
    # test pipeline registers
    reg_mode = True
    chip_size = 2
    num_tracks = 2

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

    # manual route
    # I wish I can use pycyclone here to do the automatic routing
    graph = interconnect.get_graph(16)
    first_path = []
    start_node = graph.get_sb(0, 0, SwitchBoxSide.WEST,
                              0, SwitchBoxIO.SB_IN)
    next_node = graph.get_port(0, 0, "data_in_16b")
    first_path.append(start_node)
    first_path.append(next_node)

    second_path = []
    port_out = graph.get_port(0, 0, "data_out_16b")
    second_path.append(port_out)
    # route to a sb node
    next_node = graph.get_sb(0, 0, SwitchBoxSide.EAST, 0, SwitchBoxIO.SB_OUT)
    second_path.append(next_node)
    # append a register as well
    nodes = list(next_node)
    nodes = [node for node in nodes if isinstance(node, RegisterNode)]
    assert len(nodes) == 1
    reg_node = nodes[0]
    second_path.append(reg_node)
    rmux_node = list(reg_node)[0]
    second_path.append(rmux_node)
    next_node = list(rmux_node)[0]
    second_path.append(next_node)
    next_node = graph.get_sb(1, 0, SwitchBoxSide.EAST, 0, SwitchBoxIO.SB_OUT)
    second_path.append(next_node)
    rmux_node = list(next_node)[0]
    second_path.append(rmux_node)

    # two paths
    route_path = [first_path, second_path]
    compiler = InterconnectModelCompiler(interconnect)
    compiler.configure_route(route_path)
    # no instruction as we are using dummy
    model = compiler.compile()

    # poke values
    start = first_path[0]
    end = second_path[-1]

    num_data_points = 10
    values = []
    for i in range(num_data_points):
        values.append(i + 1)
    for idx, value in enumerate(values):
        model.set_value(start, value)
        model.eval()
        if idx > 0:
            # one pipeline register
            assert model.get_value(end) == values[idx - 1]
