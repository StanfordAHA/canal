from canal.interconnect import Interconnect
from typing import List
from canal.cyclone import Node


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


def parse_routing_result(raw_routing_result, interconnect: Interconnect):
    # in the original cyclone implementation we don't need this
    # since it just translate this IR into bsb format without verifying the
    # connectivity. here, however, we need to since we're producing bitstream
    result = {}
    for net_id, raw_routes in raw_routing_result.items():
        result[net_id] = []
        for raw_segment in raw_routes:
            segment = []
            for node_str in raw_segment:
                node = interconnect.parse_node(node_str)
                segment.append(node)
            result[net_id].append(segment)
    return result


def load_routing_result(filename, interconnect: Interconnect):
    # in the original cyclone implementation we don't need this
    # since it just translate this IR into bsb format without verifying the
    # connectivity. here, however, we need to since we're producing bitstream
    raw_routing_result = __parse_raw_routing_result(filename)
    return parse_routing_result(raw_routing_result, interconnect)


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


def __route(src_node: Node, dst_node: Node, used_nodes):
    from queue import Queue
    working_set = Queue()
    path = {}
    visited = set()
    working_set.put(src_node)

    while not working_set.empty():
        node = working_set.get()
        if node in visited or node in used_nodes:
            continue
        visited.add(node)
        for n in node:
            path[n] = node
            if n == dst_node:
                # need to return the result
                result = []
                t = n
                while t in path:
                    result.append(t)
                    t = path[t]
                assert result[-1] in src_node
                result.append(src_node)
                # reverse the result
                result = result[::-1]
                return result
            else:
                working_set.put(n)
    raise RuntimeError(f"Unable to route from {src_node} {dst_node}")


def __get_ports(interconnect: Interconnect, bit_width, predicate):
    interface = interconnect.interface()
    result = []
    for port_name in interface:
        port = interconnect.ports[port_name]
        port_type = port.base_type()
        node = interface[port_name]
        if node.width != bit_width:
            continue
        if predicate(port_type):
            result.append(node)
    return result


def __get_input_ports(interconnect: Interconnect, bit_width: int):
    return __get_ports(interconnect, bit_width, lambda x: x.is_input())


def __get_output_ports(interconnect: Interconnect, bit_width: int):
    return __get_ports(interconnect, bit_width, lambda x: x.is_output())


def __get_interface_name(interconnect: Interconnect):
    result = {}
    interface = interconnect.interface()
    for port_name, port_node in interface.items():
        result[port_node] = port_name
    return result


def route_one_tile(interconnect: Interconnect,
                   x: int,
                   y: int,
                   ports: List[str],
                   seed: int = 0):
    import random
    r = random.Random(seed)

    result = {}
    mapping = {}
    interface_mapping = __get_interface_name(interconnect)
    tile = interconnect.tile_circuits[(x, y)]
    port_nodes = {}
    for tile_g in tile.tiles.values():
        for p, node in tile_g.ports.items():
            if p in ports:
                port_nodes[p] = node

    # check if we've found all port nodes
    for p in ports:
        if p not in port_nodes:
            raise ValueError(f"Unable to find port {p}")
    used_nodes = set()
    # sort the keys to be deterministic
    port_names = list(port_nodes.keys())
    port_names.sort()
    input_ports = {}
    output_ports = {}
    for bit_width in interconnect.get_bit_widths():
        input_ports[bit_width] = __get_input_ports(interconnect, bit_width)
        output_ports[bit_width] = __get_output_ports(interconnect, bit_width)

    # based on if it's input or output
    used_nodes = set()
    for port_name in port_names:
        node = port_nodes[port_name]
        bit_width = node.width
        if len(node) == 0:
            # it's an input
            # randomly choose an input
            src_index = r.randrange(len(input_ports[bit_width]))
            src_node = input_ports[bit_width][src_index]
            input_ports[bit_width].pop(src_index)
            path = __route(src_node, node, used_nodes)
            mapping[port_name] = interface_mapping[src_node]
        else:
            # it's on output
            # randomly choose an output
            dst_index = r.randrange(len(output_ports[bit_width]))
            dst_node = output_ports[bit_width][dst_index]
            output_ports[bit_width].pop(dst_index)
            path = __route(node, dst_node, used_nodes)
            mapping[port_name] = interface_mapping[dst_node]
        for n in path:
            used_nodes.add(n)
        result[f"e{len(result)}"] = [path]

    return result, mapping
