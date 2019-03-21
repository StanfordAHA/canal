from canal.interconnect import Interconnect
from canal.cyclone import SwitchBoxIO, SwitchBoxSide


def __parse_raw_routing_result(filename):
    # copied from pnr python implementation
    with open(filename) as f:
        lines = f.readlines()

    routes = {}
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].strip()
        line_index += 1
        if line[:3] == "Net":
            tokens = line.split(" ")
            net_id = tokens[2]
            routes[net_id] = []
            num_seg = int(tokens[-1])
            for seg_index in range(num_seg):
                segment = []
                line = lines[line_index].strip()
                line_index += 1
                assert line[:len("Segment")] == "Segment"
                tokens = line.split()
                seg_size = int(tokens[-1])
                for i in range(seg_size):
                    line = lines[line_index].strip()
                    line_index += 1
                    line = "".join([x for x in line if x not in ",()"])
                    tokens = line.split()
                    tokens = [int(x) if x.isdigit() else x for x in tokens]
                    segment.append(tokens)
                routes[net_id].append(segment)
    return routes


def load_routing_result(filename, interconnect: Interconnect):
    # in the original cyclone implementation we don't need this
    # since it just translate this IR into bsb format without verifying the
    # connectivity. here, however, we need to since we're producing bitstream
    raw_routing_result = __parse_raw_routing_result(filename)
    result = {}

    def __parse_node(node_str_):
        if node_str_[0] == "SB":
            track, x, y, side, io_, bit_width = node_str_[1:]
            graph = interconnect.get_graph(bit_width)
            return graph.get_sb(x, y, SwitchBoxSide(side), track,
                                SwitchBoxIO(io_))
        elif node_str_[0] == "PORT":
            port_name, x, y, bit_width = node_str_[1:]
            graph = interconnect.get_graph(bit_width)
            return graph.get_port(x, y, port_name)
        elif node_str_[0] == "REG":
            reg_name, track, x, y, bit_width = node_str[1:]
            graph = interconnect.get_graph(bit_width)
            return graph.get_tile(x, y).switchbox.registers[reg_name]
        elif node_str_[0] == "RMUX":
            rmux_name, x, y, bit_width = node_str[1:]
            graph = interconnect.get_graph(bit_width)
            return graph.get_tile(x, y).switchbox.reg_muxs[rmux_name]
        else:
            raise Exception("Unknown node " + " ".join(node_str_))

    for net_id, raw_routes in raw_routing_result.items():
        result[net_id] = []
        for raw_segment in raw_routes:
            segment = []
            for node_str in raw_segment:
                node = __parse_node(node_str)
                segment.append(node)
            result[net_id].append(segment)
    return result


def load_placement(filename):
    # copied from cyclone implementation
    with open(filename) as f:
        lines = f.readlines()
    lines = lines[2:]
    placement = {}
    id_to_name = {}
    for line in lines:
        raw_line = line.split()
        assert (len(raw_line) == 4)
        blk_name = raw_line[0]
        x = int(raw_line[1])
        y = int(raw_line[2])
        blk_id = raw_line[-1][1:]
        placement[blk_id] = (x, y)
        id_to_name[blk_id] = blk_name
    return placement, id_to_name
