import magma
from ordered_set import OrderedSet
import os
from .cyclone import InterconnectGraph, SwitchBoxSide, Node
from .cyclone import Tile, SwitchBoxNode, SwitchBoxIO, RegisterMuxNode
from typing import Dict, Tuple, List
import gemstone.generator.generator as generator
from .circuit import TileCircuit, create_name
from gemstone.common.configurable import ConfigurationType
from gemstone.common.core import PnRTag
from gemstone.generator.const import Const
import enum


@enum.unique
class GlobalSignalWiring(enum.Enum):
    FanOut = enum.auto()
    Meso = enum.auto()


class Interconnect(generator.Generator):
    def __init__(self, interconnects: Dict[int, InterconnectGraph],
                 config_addr_width: int, config_data_width: int,
                 tile_id_width: int,
                 stall_signal_width: int = 4,
                 lift_ports=False):
        super().__init__()

        self.__interface = {}

        self.config_data_width = config_data_width
        self.config_addr_width = config_addr_width
        self.tile_id_width = tile_id_width
        self.stall_signal_width = stall_signal_width
        self.__graphs: Dict[int, InterconnectGraph] = interconnects
        self.__lifted_ports = lift_ports

        self.__tiles: Dict[Tuple[int, int], Dict[int, Tile]] = {}
        self.tile_circuits: Dict[Tuple[int, int], TileCircuit] = {}

        # loop through the grid and create tile circuits
        # first find all the coordinates
        coordinates = OrderedSet()
        for _, graph in self.__graphs.items():
            for coord in graph:
                coordinates.add(coord)
        # add tiles
        x_min = 0xFFFF
        x_max = -1
        y_min = 0xFFFF
        y_max = -1

        for x, y in coordinates:
            for bit_width, graph in self.__graphs.items():
                if graph.is_original_tile(x, y):
                    tile = graph[(x, y)]
                    if (x, y) not in self.__tiles:
                        self.__tiles[(x, y)] = {}
                    self.__tiles[(x, y)][bit_width] = tile

            # set the dimensions
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
        assert x_max > x_min
        assert y_max > y_min

        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        # create individual tile circuits
        for coord, tiles in self.__tiles.items():
            self.tile_circuits[coord] =\
                TileCircuit(tiles, config_addr_width, config_data_width,
                            stall_signal_width=stall_signal_width)

        # we need to deal with inter-tile connections now
        # we only limit mesh

        for (x, y), tile in self.tile_circuits.items():
            for bit_width, switch_box in tile.sbs.items():
                all_sbs = switch_box.switchbox.get_all_sbs()
                for sb in all_sbs:
                    if sb.io != SwitchBoxIO.SB_OUT:
                        continue
                    assert x == sb.x and y == sb.y
                    # we need to be carefully about looping through the
                    # connections
                    # if the switch box has pipeline registers, we need to
                    # do a "jump" over the connected switch
                    # format: dst_node, src_port_name, src_node
                    neighbors: List[Tuple[Node, str, Node]] = []
                    for node in sb:
                        if isinstance(node, SwitchBoxNode):
                            neighbors.append((node, create_name(str(sb)), sb))
                        elif isinstance(node, RegisterMuxNode):
                            # making sure the register is inserted properly
                            assert len(sb) == 2
                            # we need to make a jump here
                            for n in node:
                                neighbors.clear()
                                if isinstance(n, SwitchBoxNode):
                                    neighbors.append((n, create_name(str(sb)),
                                                      node))
                            break
                    for sb_node, src_sb_name, src_node in neighbors:
                        assert isinstance(sb_node, SwitchBoxNode)
                        assert sb_node.io == SwitchBoxIO.SB_IN
                        # it has to be a different x or y
                        same_row = sb_node.x == sb.x
                        same_col = sb_node.y == sb.y
                        if not same_row ^ same_col:
                            raise RuntimeError("Only mesh interconnect is "
                                               "supported")
                        # notice that we already lift the ports up
                        # since we are not dealing with internal connections
                        # using the tile-level port is fine
                        dst_tile = self.tile_circuits[(sb_node.x, sb_node.y)]
                        # wire them up
                        idx = sb_node.get_conn_in().index(src_node)
                        dst_sb_name = create_name(str(sb_node))
                        self.wire(tile.ports[src_sb_name],
                                  dst_tile.ports[dst_sb_name][idx])

        # connect these margin tiles, if needed
        self.__connect_margin_tiles()

        # if we need to lift the ports. this can be used for testing or
        # creating circuit without IO
        if lift_ports:
            self.__lift_ports()
        else:
            self.__ground_ports()

        # global ports
        self.globals = self.__add_global_ports(stall_signal_width)

        # add config
        self.__add_read_config_data(config_data_width)

        # clean up empty tiles
        self.__cleanup_tiles()

        # set tile_id
        self.__set_tile_id()

        self.finalized = False

    def interface(self):
        return self.__interface

    def __get_tile_id(self, x: int, y: int):
        return x << (self.tile_id_width // 2) | y

    def __set_tile_id(self):
        for (x, y), tile in self.tile_circuits.items():
            tile_id = self.__get_tile_id(x, y)
            self.wire(tile.ports.tile_id,
                      Const(magma.bits(tile_id, self.tile_id_width)))

    def __lift_ports(self):
        # we assume it's a rectangular grid
        # we only care about the perimeter
        x_range = {self.x_min, self.x_max}
        y_range = {self.y_min, self.y_max}
        coordinates = OrderedSet()
        for (x, y) in self.tile_circuits:
            if x in x_range or y in y_range:
                coordinates.append((x, y))
        for x, y in coordinates:
            tile = self.tile_circuits[(x, y)]
            # we only lift sb ports
            sbs = tile.sbs
            for bit_width, switchbox in sbs.items():
                all_sbs = switchbox.switchbox.get_all_sbs()
                working_set = []
                if x == self.x_min:
                    # we lift west/left ports
                    for sb_node in all_sbs:
                        if sb_node.side != SwitchBoxSide.WEST:
                            continue
                        working_set.append(sb_node)
                elif x == self.x_max:
                    # we lift east/right ports
                    for sb_node in all_sbs:
                        if sb_node.side != SwitchBoxSide.EAST:
                            continue
                        working_set.append(sb_node)
                if y == self.y_min:
                    # we lift north/top ports
                    for sb_node in all_sbs:
                        if sb_node.side != SwitchBoxSide.NORTH:
                            continue
                        working_set.append(sb_node)
                elif y == self.y_max:
                    # we lift south/bottom ports
                    for sb_node in all_sbs:
                        if sb_node.side != SwitchBoxSide.SOUTH:
                            continue
                        working_set.append(sb_node)
                for sb_node in working_set:
                    sb_name = create_name(str(sb_node))
                    sb_port = tile.ports[sb_name]
                    # because the lifted port will conflict with each other
                    # we need to add x and y to the sb_name to avoid conflict
                    new_sb_name = sb_name + f"_X{sb_node.x}_Y{sb_node.y}"
                    self.add_port(new_sb_name, sb_port.base_type())
                    self.__interface[new_sb_name] = sb_node
                    self.wire(self.ports[new_sb_name], sb_port)

    def __connect_margin_tiles(self):
        # connect these margin tiles
        # margin tiles have empty switchbox
        for coord, tile_dict in self.__tiles.items():
            for bit_width, tile in tile_dict.items():
                if tile.switchbox.num_track > 0 or tile.core is None:
                    continue
                for port_name, port_node in tile.ports.items():
                    tile_port = self.tile_circuits[coord].ports[port_name]
                    if len(port_node) == 0 and \
                            len(port_node.get_conn_in()) == 0:
                        # lift this port up
                        x, y = coord
                        new_port_name = f"{port_name}_X{x:02X}_Y{y:02X}"
                        self.add_port(new_port_name, tile_port.base_type())
                        self.__interface[new_port_name] = port_node
                        self.wire(self.ports[new_port_name], tile_port)
                    else:
                        # connect them to the internal fabric
                        nodes = list(port_node) + port_node.get_conn_in()[:]
                        for sb_node in nodes:
                            next_coord = sb_node.x, sb_node.y
                            # depends on whether there is a pipeline register
                            # or not, we need to be very careful
                            if isinstance(sb_node, SwitchBoxNode):
                                sb_name = create_name(str(sb_node))
                            else:
                                assert isinstance(sb_node, RegisterMuxNode)
                                # because margin tiles won't connect to
                                # reg mux node, they can only be connected
                                # from
                                nodes = sb_node.get_conn_in()[:]
                                nodes = [x for x in nodes if
                                         isinstance(x, SwitchBoxNode)]
                                assert len(nodes) == 1
                                sb_node = nodes[0]
                                sb_name = create_name(str(sb_node))

                            next_port = \
                                self.tile_circuits[next_coord].ports[sb_name]
                            if sb_node.io == SwitchBoxIO.SB_IN:
                                next_port = next_port[0]
                            else:
                                tile_port = tile_port[0]

                            self.wire(tile_port, next_port)

    def __ground_ports(self):
        # this is a pass to ground every sb ports that's not connected
        for coord, tile_dict in self.__tiles.items():
            for bit_width, tile in tile_dict.items():
                ground = Const(magma.bits(0, bit_width))
                for sb in tile.switchbox.get_all_sbs():
                    if sb.io != SwitchBoxIO.SB_IN:
                        continue
                    if sb.get_conn_in():
                        continue
                    # no connection to that sb port, ground it
                    sb_name = create_name(str(sb))
                    sb_port = self.tile_circuits[coord].ports[sb_name]
                    self.wire(ground, sb_port[0])

    def __cleanup_tiles(self):
        tiles_to_remove = set()
        for coord, tile in self.tile_circuits.items():
            tile_circuit = self.tile_circuits[coord]
            if tile.core is None:
                tiles_to_remove.add(coord)
            else:
                core_tile = False
                for _, sb in tile_circuit.sbs.items():
                    if len(sb.wires) > 0:
                        core_tile = True
                        break
                if core_tile:
                    continue

            # remove all the global signals
            for signal in self.globals:
                if signal in tile_circuit.ports:
                    tile_circuit.ports.pop(signal)
        # remove empty tiles
        for coord in tiles_to_remove:
            # remove the tile id as well
            tile_circuit = self.tile_circuits[coord]
            tile_circuit.ports.pop("tile_id")
            self.tile_circuits.pop(coord)

    def __add_read_config_data(self, config_data_width: int):
        self.add_port("read_config_data",
                      magma.Out(magma.Bits[config_data_width]))

    def __add_global_ports(self, stall_signal_width: int):
        self.add_ports(
            config=magma.In(ConfigurationType(self.config_data_width,
                                              self.config_data_width)),
            clk=magma.In(magma.Clock),
            reset=magma.In(magma.AsyncReset),
            stall=magma.In(magma.Bits[stall_signal_width]))

        return (self.ports.config.qualified_name(),
                self.ports.clk.qualified_name(),
                self.ports.reset.qualified_name(),
                self.ports.stall.qualified_name())

    def finalize(self):
        if self.finalized:
            raise Exception("Circuit already finalized")
        self.finalized = True
        # finalize the design. after this, users are not able to add
        # features to the tiles any more
        # clean up first
        self.__cleanup_tiles()
        # finalize individual tiles first
        for _, tile in self.tile_circuits.items():
            tile.finalize()

    def get_node_bitstream_config(self, src_node: Node, dst_node: Node):
        # this is the complete one which includes the tile_id
        x, y = dst_node.x, dst_node.y
        tile = self.tile_circuits[(x, y)]
        addr, data = tile.get_route_bitstream_config(src_node, dst_node)
        tile_id = self.__get_tile_id(x, y)
        addr = addr | tile_id
        return addr, data

    def get_route_bitstream(self, routes: Dict[str, List[List[Node]]]):
        result = []
        for _, route in routes.items():
            for segment in route:
                for i in range(len(segment) - 1):
                    pre_node = segment[i]
                    next_node = segment[i + 1]
                    assert next_node in pre_node
                    if pre_node.x != next_node.x or pre_node.y != next_node.y:
                        # inter tile connection. skipping for now
                        continue
                    if len(next_node.get_conn_in()) == 1:
                        # no mux created. skip
                        continue
                    addr, data = self.get_node_bitstream_config(pre_node,
                                                                next_node)
                    result.append((addr, data))
        return result

    def configure_placement(self, x: int, y: int, instr):
        tile_id = self.__get_tile_id(x, y)
        tile = self.tile_circuits[(x, y)]
        core = tile.core
        result = core.configure(instr)
        # TODO(rsetaluri): Cache this information. Also, this should be
        # refactored and abstracted.
        feature_addr = tile.features().index(core)
        for i in range(len(result)):
            reg_index, data = result[i]
            addr = (reg_index << tile.feature_config_slice.start) | \
                   (feature_addr << tile.tile_id_width)
            addr = addr | tile_id
            result[i] = (addr, data)
        return result

    def __get_core_info(self) -> Dict[str, Tuple[PnRTag, List[PnRTag]]]:
        result = {}
        for coord in self.tile_circuits:
            tile = self.tile_circuits[coord]
            info = tile.core.pnr_info()
            core_name = tile.core.name()
            if core_name not in result:
                result[core_name] = info
            else:
                assert result[core_name] == info
        return result

    @staticmethod
    def __get_core_tag(core_info):
        name_to_tag = {}
        tag_to_name = {}
        tag_to_priority = {}
        for core_name, tags in core_info.items():
            if not isinstance(tags, list):
                tags = [tags]
            for tag in tags:    # type: PnRTag
                tag_name = tag.tag_name
                if core_name not in name_to_tag:
                    name_to_tag[core_name] = []
                name_to_tag[core_name].append(tag.tag_name)
                assert tag_name not in tag_to_name, f"{tag_name} already exists"
                tag_to_name[tag_name] = core_name
                tag_to_priority[tag_name] = (tag.priority_major,
                                             tag.priority_minor)
        return name_to_tag, tag_to_name, tag_to_priority

    def __get_registered_tile(self):
        result = set()
        for coord, tile_circuit in self.tile_circuits.items():
            for _, tile in tile_circuit.tiles.items():
                switchbox = tile.switchbox
                if len(switchbox.registers) > 0:
                    result.add(coord)
                    break
        return result

    def dump_pnr(self, dir_name, design_name):
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        dir_name = os.path.abspath(dir_name)
        graph_path_dict = {}
        for bit_width, graph in self.__graphs.items():
            graph_path = os.path.join(dir_name, f"{bit_width}.graph")
            graph_path_dict[bit_width] = graph_path
            graph.dump_graph(graph_path)

        # generate the layout file
        layout_file = os.path.join(dir_name, f"{design_name}.layout")
        self.__dump_layout_file(layout_file)
        pnr_file = os.path.join(dir_name, f"{design_name}.info")
        with open(pnr_file, "w+") as f:
            f.write(f"layout={layout_file}\n")
            graph_configs = [f"{bit_width} {graph_path_dict[bit_width]}" for
                             bit_width in self.__graphs]
            graph_config_str = " ".join(graph_configs)
            f.write(f"graph={graph_config_str}\n")

    def __dump_layout_file(self, layout_file):
        # empty tiles first first
        with open(layout_file, "w+") as f:
            f.write("LAYOUT   0 20\nBEGIN\n")
            for y in range(self.y_max + 1):
                for x in range(self.x_max + 1):
                    coord = (x, y)
                    if coord not in self.tile_circuits:
                        f.write("1")
                    else:
                        f.write("0")
                f.write("\n")
            f.write("END\n")
            # looping through the tiles to figure what core it has
            # use default priority 20
            default_priority = 20
            core_info = self.__get_core_info()
            name_to_tag, tag_to_name, tag_to_priority \
                = self.__get_core_tag(core_info)
            for core_name, tags in name_to_tag.items():
                for tag in tags:
                    priority_major, priority_minor = tag_to_priority[tag]
                    f.write(f"LAYOUT {tag} {priority_major} {priority_minor}\n")
                    f.write("BEGIN\n")
                    for y in range(self.y_max + 1):
                        for x in range(self.x_max + 1):
                            coord = (x, y)
                            if coord not in self.tile_circuits or \
                                    self.tile_circuits[
                                        coord].core.name() != core_name:
                                f.write("0")
                            else:
                                f.write("1")
                        f.write("\n")
                    f.write("END\n")
            # handle registers
            assert "r" not in tag_to_name
            r_locs = self.__get_registered_tile()
            f.write(f"LAYOUT r {default_priority} 0\nBEGIN\n")
            for y in range(self.y_max + 1):
                for x in range(self.x_max + 1):
                    if (x, y) in r_locs:
                        f.write("1")
                    else:
                        f.write("0")
                f.write("\n")
            f.write("END\n")

    def parse_node(self, node_str):
        if node_str[0] == "SB":
            track, x, y, side, io_, bit_width = node_str[1:]
            graph = self.get_graph(bit_width)
            return graph.get_sb(x, y, SwitchBoxSide(side), track,
                                SwitchBoxIO(io_))
        elif node_str[0] == "PORT":
            port_name, x, y, bit_width = node_str[1:]
            graph = self.get_graph(bit_width)
            return graph.get_port(x, y, port_name)
        elif node_str[0] == "REG":
            reg_name, track, x, y, bit_width = node_str[1:]
            graph = self.get_graph(bit_width)
            return graph.get_tile(x, y).switchbox.registers[reg_name]
        elif node_str[0] == "RMUX":
            rmux_name, x, y, bit_width = node_str[1:]
            graph = self.get_graph(bit_width)
            return graph.get_tile(x, y).switchbox.reg_muxs[rmux_name]
        else:
            raise Exception("Unknown node " + " ".join(node_str))

    def clone(self):
        bit_widths = self.get_bit_widths()
        result_graph = {}
        for bit_width in bit_widths:
            graph = self.get_graph(bit_width)
            new_graph = graph.clone()
            result_graph[bit_width] = new_graph
        ic = Interconnect(result_graph, self.config_addr_width,
                          self.config_data_width,
                          self.tile_id_width,
                          self.stall_signal_width,
                          self.__lifted_ports)
        return ic

    def get_column(self, x: int):
        # obtain a list of columns sorted by y
        result = []
        for y in range(self.y_min, self.y_max + 1):  # y_max is inclusive
            if (x, y) in self.tile_circuits:
                # if it exists
                tile = self.tile_circuits[(x, y)]
                result.append(tile)
        return result

    def get_graph(self, bit_width: int):
        return self.__graphs[bit_width]

    def get_bit_widths(self):
        return list(self.__graphs.keys())

    def name(self):
        return "Interconnect"
