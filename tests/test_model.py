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
        visited = set()

        def construct_path_(_node):
            if _node in visited:
                return
            if _node == end_node:
                return
            else:
                visited.add(_node)
                nodes_ = list(_node)
                rnd.shuffle(nodes_)
                for node__ in nodes_:
                    link[node__] = _node
                    construct_path_(node__)
        construct_path_(start_node)
        assert end_node in link
        path = []
        node_ = end_node
        while node_ != start_node:
            path.append(node_)
            node_ = link[node_]
        path.append(node_)
        return path

    compiler = InterconnectModelCompiler(interconnect)
    model = compiler.compile()

    route_path: List[Node] = construct_path()
    # add a PE to it if not
    has_pe = False
    for node in route_path:
        if isinstance(node, PortNode):
            has_pe = True
            break
    if not has_pe:
        # very likely it's the case
        has_port_node = False
        port_node = None
        pre_node = None
        for node in route_path:
            next_nodes = list(node)
            for next_node in next_nodes:
                if isinstance(next_node, PortNode):
                    has_port_node = True
                    port_node = next_node
                    pre_node = node
                    break
        if not has_port_node or port_node is None:
            raise Exception("unable to construct a path to test the simulator")

        pre_index = route_path.index(pre_node)
        next_path_node = route_path[pre_index + 1]
        output_port_nodes = list(port_node)
        assert len(output_port_nodes) == 1
        output_port = output_port_nodes[0]
        assert next_path_node in output_port, "Unable to find next port node"
        route_path.insert(pre_index + 1, port_node)
        route_path.insert(pre_index + 2, output_port)

    config = []
    for i in range(len(route_path) - 1):
        pass
