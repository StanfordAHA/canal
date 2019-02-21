from gemstone.common.dummy_core_magma import DummyCore
from canal.util import create_uniform_interconnect, SwitchBoxType
from canal.cyclone import *
from canal.interconnect import Interconnect


def _test_smoke():
    width = 4
    height = 4
    config_data_width = 32
    config_addr_width = 8
    tile_id_width = 16

    cores = {}
    for x in range(width):
        for y in range(height):
            cores[(x, y)] = DummyCore()

    def create_core(x_, y_):
        return cores[(x_, y_)]

    # specify input and output port connections
    inputs = ["data_in_1b", "data_in_16b"]
    outputs = ["data_out_1b", "data_out_16b"]
    # this is slightly different from the chip we tape out
    # here we connect input to every SB_IN and output to every SB_OUT
    port_conns = {}
    in_conn = []
    out_conn = []
    for side in SwitchBoxSide:
        in_conn.append((side, SwitchBoxIO.SB_IN))
        out_conn.append((side, SwitchBoxIO.SB_OUT))
    for input_port in inputs:
        port_conns[input_port] = in_conn
    for output_port in outputs:
        port_conns[output_port] = out_conn

    ic_graphs = {}
    for bit_width in [1, 16]:
        ic_graph = create_uniform_interconnect(width, height, bit_width,
                                               create_core, port_conns,
                                               {1: 5},
                                               SwitchBoxType.Disjoint)
        ic_graphs[bit_width] = ic_graph

    interconnect = Interconnect(ic_graphs, config_addr_width,
                                config_data_width, tile_id_width,
                                lift_ports=True)
