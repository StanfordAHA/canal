import magma
from ordered_set import OrderedSet
import os
from .cyclone import InterconnectGraph, SwitchBoxSide, Node, PortNode
from .cyclone import Tile, SwitchBoxNode, SwitchBoxIO, RegisterMuxNode, RegisterNode
from typing import Dict, Tuple, List
import gemstone.generator.generator as generator
from .circuit import TileCircuit, create_name
from gemstone.common.configurable import ConfigurationType
from gemstone.common.core import PnRTag, ConfigurableCore
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
                 lift_ports=False,
                 double_buffer: bool = False,
                 ready_valid: bool = False,
                 give_north_io_sbs: bool = False):
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

        self.double_buffer = double_buffer
        self.ready_valid = ready_valid

        self.give_north_io_sbs = give_north_io_sbs

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
        assert x_max >= x_min
        assert y_max >= y_min

        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        # create individual tile circuits
        for coord, tiles in self.__tiles.items():
            self.tile_circuits[coord] = \
                TileCircuit(tiles, config_addr_width, config_data_width,
                            stall_signal_width=stall_signal_width,
                            double_buffer=self.double_buffer,
                            ready_valid=self.ready_valid,
                            give_north_io_sbs=self.give_north_io_sbs)

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
                        dst_sb_name = create_name(str(sb_node))
                        if len(sb_node.get_conn_in()) == 1:
                            # no array
                            self.wire(tile.ports[src_sb_name],
                                      dst_tile.ports[dst_sb_name])
                            if self.ready_valid:
                                self.wire(tile.ports[src_sb_name + "_ready"],
                                          dst_tile.ports[dst_sb_name + "_ready"])
                                self.wire(tile.ports[src_sb_name + "_valid"],
                                          dst_tile.ports[dst_sb_name + "_valid"])
                        else:
                            idx = sb_node.get_conn_in().index(src_node)
                            self.wire(tile.ports[src_sb_name][idx],
                                      dst_tile.ports[dst_sb_name])
                            if self.ready_valid:
                                self.wire(tile.ports[src_sb_name + "_ready"],
                                          dst_tile.ports[dst_sb_name + "_ready"])
                                self.wire(tile.ports[src_sb_name + "_valid"],
                                          dst_tile.ports[dst_sb_name + "_valid"][idx])

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

    def get_tile_id(self, x: int, y: int):
        return x << (self.tile_id_width // 2) | y

    def __set_tile_id(self):
        for (x, y), tile in self.tile_circuits.items():
            tile_id = self.get_tile_id(x, y)
            self.wire(tile.ports.tile_id,
                      Const(magma.Bits[self.tile_id_width](tile_id)))

    def get_config_addr(self, reg_addr: int, feat_addr: int, x: int, y: int):
        tile_id = self.get_tile_id(x, y)
        tile = self.tile_circuits[(x, y)]
        addr = (reg_addr << tile.feature_config_slice.start) | \
               (feat_addr << tile.tile_id_width)
        addr = addr | tile_id
        return addr

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
                if x == self.x_max:
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
                if y == self.y_max:
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
                    if self.ready_valid:
                        if sb_node.io == SwitchBoxIO.SB_OUT:
                            ready_name = new_sb_name + "_ready"
                            p = self.add_port(ready_name, magma.In(magma.Bit))
                            self.wire(p, tile.ports[sb_name + "_ready"])
                            self.__interface[ready_name] = sb_port

                            valid_name = new_sb_name + "_valid"
                            p = self.add_port(valid_name, magma.Out(magma.Bit))
                            self.wire(p, tile.ports[sb_name + "_valid"])
                            self.__interface[valid_name] = sb_port
                        else:
                            valid_name = new_sb_name + "_valid"
                            p = self.add_port(valid_name, magma.BitIn)
                            self.wire(p, tile.ports[sb_name + "_valid"])
                            self.__interface[valid_name] = sb_port

                            ready_name = new_sb_name + "_ready"
                            p = self.add_port(ready_name, magma.Out(magma.Bit))
                            self.wire(p, tile.ports[sb_name + "_ready"])
                            self.__interface[ready_name] = sb_port

    def skip_margin_connection(self, tile):
        if self.give_north_io_sbs:
            return tile.switchbox.num_track > 0 and tile.y != 0
        else:
            return tile.switchbox.num_track > 0

    def dont_lift_port(self, tile, port_name):
        # dont lift f2io and io2f ports to interconnect level since these connect to the SB within the I/O tile
        return self.give_north_io_sbs and (tile.y == 0 and ("f2io" in port_name or "io2f" in port_name))

    # This function makes the tile-to-tile connection for the margin tiles
    # And lifts up ports at the "edge" of the Interconnect graph as ports for the
    # Interconnect module
    def __connect_margin_tiles(self):
        # connect these margin tiles
        # margin tiles have empty switchbox
        for coord, tile_dict in self.__tiles.items():
            for bit_width, tile in tile_dict.items():
                if self.skip_margin_connection(tile) or tile.core is None:
                    continue
                for port_name, port_node in tile.ports.items():
                    if port_name == "flush" or self.dont_lift_port(tile, port_name):
                        continue
                    tile_port = self.tile_circuits[coord].ports[port_name]
                    # FIXME: this is a hack
                    valid_connected = False
                    if len(port_node) == 0 and \
                            len(port_node.get_conn_in()) == 0:
                        # lift this port up
                        x, y = coord
                        new_port_name = f"{port_name}_X{x:02X}_Y{y:02X}"
                        self.add_port(new_port_name, tile_port.base_type())
                        self.__interface[new_port_name] = port_node
                        self.wire(self.ports[new_port_name], tile_port)

                        # need to create ready-valid port for them as well
                        if self.ready_valid:
                            ready_name = port_name + "_ready"
                            ready_port = self.tile_circuits[coord].ports[ready_name]
                            if ready_name not in self.ports:
                                p = self.add_port(new_port_name + "_ready", ready_port.base_type())
                                self.wire(p, ready_port)
                                valid_port = self.tile_circuits[coord].ports[port_name + "_valid"]
                                p = self.add_port(new_port_name + "_valid", valid_port.base_type())
                                self.wire(p, valid_port)
                    else:
                        # connect them to the internal fabric
                        nodes = list(port_node) + port_node.get_conn_in()[:]
                        for sb_node in nodes:
                            next_coord = sb_node.x, sb_node.y
                            next_node = sb_node
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
                            if len(port_node.get_conn_in()) <= 1:
                                self.wire(tile_port, next_port)
                                if self.ready_valid:
                                    # FIXME: this is a hack to get stuff connected. Notice that the newest CB has
                                    # connection box, but we haven't dealt with the connection box wide connections
                                    # yet
                                    # if this is a fanout, we need to deal with the ready fanin
                                    # because this is an IO, we directly OR them together
                                    ready_port = self.tile_circuits[coord].ports[port_name + "_ready"]
                                    if len(port_node) > 1:
                                        # need to modify the ports and change it into a multiple ports
                                        idx = list(port_node).index(sb_node)
                                        ready_port = ready_port[idx]
                                    if ready_port.base_type() is magma.BitOut:
                                        next_ready_port = self.tile_circuits[next_coord].ports[sb_name + "_ready"]
                                        self.wire(next_ready_port, ready_port)
                                    elif sb_node in port_node:
                                        # coming to that node
                                        next_ready_port = self.tile_circuits[next_coord].ports[sb_name + "_ready"]
                                        self.wire(next_ready_port, ready_port)
                                    valid_port = self.tile_circuits[coord].ports[port_name + "_valid"]
                                    if valid_port.base_type() is magma.BitOut:
                                        next_valid_port = self.tile_circuits[next_coord].ports[sb_name + "_valid"]
                                        self.wire(next_valid_port, valid_port)
                                    elif not valid_connected:
                                        next_valid_port = self.tile_circuits[next_coord].ports[sb_name + "_valid"]
                                        self.wire(next_valid_port, valid_port)
                                        valid_connected = True
                            else:
                                # need to get the sliced port
                                idx = port_node.get_conn_in().index(next_node)
                                self.wire(tile_port[idx], next_port)
                                if self.ready_valid:
                                    raise RuntimeError("Not supported")

    def __ground_ports(self):
        # this is a pass to ground every sb ports that's not connected
        for coord, tile_dict in self.__tiles.items():
            for bit_width, tile in tile_dict.items():
                ground = Const(magma.Bits[bit_width](0))
                for sb in tile.switchbox.get_all_sbs():
                    if sb.io == SwitchBoxIO.SB_IN:
                        if sb.get_conn_in():
                            continue
                        # no connection to that sb port, ground it
                        sb_name = create_name(str(sb))
                        sb_port = self.tile_circuits[coord].ports[sb_name]
                        self.wire(ground, sb_port)
                        if self.ready_valid:
                            self.wire(Const(magma.Bit(0)), self.tile_circuits[coord].ports[sb_name + "_valid"])
                    else:
                        margin = False
                        if len(sb) > 0:
                            for n in sb:
                                if isinstance(n, RegisterMuxNode):
                                    margin = len(n) == 0
                        else:
                            margin = True
                        if not margin:
                            continue
                        if self.ready_valid:
                            sb_name = create_name(str(sb))
                            self.wire(Const(magma.Bit(0)), self.tile_circuits[coord].ports[sb_name + "_ready"])

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
                    if tile.core is not None and tile.needs_signal(signal):
                        continue
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
        if self.double_buffer:
            self.add_ports(
                config_db=magma.In(magma.Bit),
                use_db=magma.In(magma.Bit)
            )
            return (self.ports.config.qualified_name(),
                    self.ports.clk.qualified_name(),
                    self.ports.reset.qualified_name(),
                    self.ports.stall.qualified_name(),
                    self.ports.config_db.qualified_name(),
                    self.ports.use_db.qualified_name())

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
        res = []
        configs = tile.get_route_bitstream_config(src_node, dst_node)

        def add_config(entry):
            reg_addr, feat_addr, data = entry
            addr = self.get_config_addr(reg_addr, feat_addr, x, y)
            res.append((addr, data))

        if isinstance(configs, list):
            for entry in configs:
                add_config(entry)
        else:
            add_config(configs)

        return res

    def __set_fifo_mode(self, node: RegisterNode, start: bool, end: bool, use_non_split_fifos: bool = False,
                        bogus_init: bool = False, bogus_init_num: int = 0):
        x, y = node.x, node.y
        tile = self.tile_circuits[(x, y)]
        config_data = tile.configure_fifo(node, start, end, use_non_split_fifos=use_non_split_fifos,
                                          bogus_init=bogus_init, bogus_init_num=bogus_init_num)
        res = []
        for reg_addr, feat_addr, data in config_data:
            addr = self.get_config_addr(reg_addr, feat_addr, x, y)
            res.append((addr, data))

        return res

    def get_bogus_init_config(self, node_track: str, x: int, y: int, id_to_name: Dict[str, str],
                              reg_loc_to_id: Dict[Tuple[int, int], List[str]], id_to_metadata):
        matching_reg_list = reg_loc_to_id[(x, y)]
        for reg in matching_reg_list:
            reg_full_name = id_to_name[reg]
            if node_track in reg_full_name:
                if reg in id_to_metadata:
                    reg_metadata = id_to_metadata[reg]
                    extra_data_ = int(reg_metadata['extra_data'])
                    if extra_data_ == 1:
                        print(f"INFO: Added interconnect FIFO bogus data at {node_track} at {x}, {y}")
                        return True

        return False

    def get_bogus_num_config(self, node_track: str, x: int, y: int, id_to_name: Dict[str, str],
                             reg_loc_to_id: Dict[Tuple[int, int], List[str]], id_to_metadata):
        matching_reg_list = reg_loc_to_id[(x, y)]
        for reg in matching_reg_list:
            reg_full_name = id_to_name[reg]
            if node_track in reg_full_name:
                if reg in id_to_metadata:
                    reg_metadata = id_to_metadata[reg]
                    extra_data_ = int(reg_metadata['extra_data'])
                    if extra_data_ > 0:
                        print(f"INFO: Added interconnect FIFO bogus data at {node_track} at {x}, {y}")
                        return extra_data_
        return 0

    def merge_segments_across_routes(self, routes: Dict[str, List[List[object]]]) -> Dict[str, List[List[object]]]:
        '''
        Merge segments splitted by REG node
        '''
        # Flatten segments
        seg_list = []
        for route_key, segs in routes.items():
            for seg in segs:
                seg_list.append({"route": route_key, "nodes": seg.copy()})

        # Helper function to form a key for register nodes.
        def reg_key(node: object):
            # For checking whether reg nodes are the same
            return (str(node), node.x, node.y)

        # Standard iterative merging on endpoints
        changed = True
        while changed:
            changed = False
            new_seg_list = []
            used = [False] * len(seg_list)
            for i in range(len(seg_list)):
                if used[i]:
                    continue
                merged_route = seg_list[i]["route"]
                merged_nodes = seg_list[i]["nodes"].copy()
                merged = True
                while merged:
                    merged = False
                    for j in range(len(seg_list)):
                        if used[j] or i == j:
                            continue
                        candidate = seg_list[j]
                        if not candidate["nodes"]:
                            continue
                        # Case 1: Append candidate if current segment's last node matches candidate's first node.
                        if (isinstance(merged_nodes[-1], RegisterNode) and isinstance(candidate["nodes"][0], RegisterNode) and reg_key(merged_nodes[-1]) == reg_key(candidate["nodes"][0])):
                            merged_nodes.extend(candidate["nodes"][1:])  # skip joint REG node
                            used[j] = True
                            merged = True
                            changed = True
                            break
                        # Case 2: Prepend candidate if current segment's first node matches candidate's last node.
                        if (isinstance(merged_nodes[0], RegisterNode) and isinstance(candidate["nodes"][-1], RegisterNode) and reg_key(merged_nodes[0]) == reg_key(candidate["nodes"][-1])):
                            merged_nodes = candidate["nodes"][:-1] + merged_nodes  # skip joint REG node
                            used[j] = True
                            merged = True
                            changed = True
                            break
                new_seg_list.append({"route": merged_route, "nodes": merged_nodes})
                used[i] = True
            seg_list = new_seg_list

        # Additional pass: look for same register node in middle point to handle branching
        # Update the branch segment by prepending the candidate's prefix
        for seg in seg_list:
            if seg["nodes"] and isinstance(seg["nodes"][0], RegisterNode):
                branch_reg = seg["nodes"][0]
                best_prefix = None
                # Search all other segments for a candidate that contains branch_reg not at index 0.
                for candidate in seg_list:
                    if candidate is seg:
                        continue
                    for idx in range(1, len(candidate["nodes"])):
                        node = candidate["nodes"][idx]
                        if isinstance(node, RegisterNode) and reg_key(node) == reg_key(branch_reg):
                            candidate_prefix = candidate["nodes"][:idx + 1]
                            # Choose the candidate with the longest prefix if multiple exist.
                            if best_prefix is None or len(candidate_prefix) > len(best_prefix):
                                best_prefix = candidate_prefix
                            break  # Only need the first occurrence in this candidate.
                if best_prefix is not None:
                    # Prepend the found prefix, skipping the duplicate register at the branch start.
                    seg["nodes"] = best_prefix + seg["nodes"][1:]

        # Reconstruct the routes dictionary.
        new_routes: Dict[str, List[List[object]]] = {}
        for seg in seg_list:
            key = seg["route"]
            new_routes.setdefault(key, []).append(seg["nodes"])
        return new_routes

    def get_route_bitstream(self, routes: Dict[str, List[List[Node]]], use_fifo: bool = False,
                            id_to_name=None, reg_loc_to_id=None, id_to_metadata=None):
        result = []
        for _, route in routes.items():
            for segment in route:
                for i in range(len(segment) - 1):
                    pre_node = segment[i]
                    next_node = segment[i + 1]
                    assert next_node in pre_node
                    # notice that there is a corner case where the SB directly connect to the
                    # next tile's CB
                    if (pre_node.x != next_node.x or pre_node.y != next_node.y) and \
                            (not (isinstance(pre_node, RegisterMuxNode) and isinstance(next_node, PortNode))):
                        # inter tile connection. skipping for now
                        continue
                    if len(next_node.get_conn_in()) == 1:
                        # no mux created. skip
                        continue
                    configs = self.get_node_bitstream_config(pre_node, next_node,)
                    for addr, data in configs:
                        result.append((addr, data))

                    # FIFO config for non-split FIFOs
                    use_non_split_fifos = "USE_NON_SPLIT_FIFOS" in os.environ and os.environ.get("USE_NON_SPLIT_FIFOS") == "1"
                    if use_non_split_fifos:
                        if use_fifo:
                            if isinstance(next_node, RegisterMuxNode) and isinstance(pre_node, RegisterNode):
                                if reg_loc_to_id is None:
                                    bogus_init_num = 0
                                else:
                                    bogus_init_num = self.get_bogus_num_config(pre_node.name, pre_node.x,
                                                                               pre_node.y, id_to_name, reg_loc_to_id, id_to_metadata)
                                config = self.__set_fifo_mode(pre_node, start=False, end=False,
                                                              use_non_split_fifos=True, bogus_init_num=bogus_init_num)
                                result += config

                # FIFO config for split FIFOs
                # Only turn reg pairs into FIFOs if using split FIFOs
                if not use_non_split_fifos:
                    if use_fifo and len(segment) >= 4:
                        reg_nodes = []
                        idx = 0
                        while idx < len(segment):
                            pre_node = segment[idx]
                            if isinstance(pre_node, RegisterNode):
                                if pre_node not in reg_nodes:
                                    reg_nodes.append(pre_node)
                            idx += 1

                        # Try to configure...
                        # for idx in range(0, len(reg_nodes)):
                            # rn = reg_nodes[idx]
                            # rn_config = self.get_bogus_init_config(rn.name, rn.x, rn.y, id_to_name, reg_loc_to_id, id_to_metadata)
                        if len(reg_nodes) != 0:
                            assert len(reg_nodes) != 1, "Cannot have standalone FIFO reg in the segment"
                            assert len(reg_nodes) % 2 == 0, "Must have even number of FIFO regs"
                            for idx in range(0, len(reg_nodes), 2):
                                first_node = reg_nodes[idx]
                                last_node = reg_nodes[idx + 1]

                                if reg_loc_to_id is None:
                                    first_node_bogus_init = False
                                    last_node_bogus_init = False
                                else:
                                    first_node_bogus_init = self.get_bogus_init_config(first_node.name, first_node.x,
                                                                                       first_node.y, id_to_name, reg_loc_to_id,
                                                                                       id_to_metadata)
                                    last_node_bogus_init = self.get_bogus_init_config(last_node.name, last_node.x,
                                                                                      last_node.y, id_to_name, reg_loc_to_id,
                                                                                      id_to_metadata)
                                print(f"First node: {first_node.name} - {first_node_bogus_init}")
                                print(f"Last node: {last_node.name} - {last_node_bogus_init}")

                                config = self.__set_fifo_mode(first_node, start=True, end=False, bogus_init=first_node_bogus_init)
                                result += config
                                config = self.__set_fifo_mode(last_node, start=False, end=True, bogus_init=last_node_bogus_init)
                                result += config

        return result

    def configure_placement(self, x: int, y: int, instr, pnr_tag=None, node_num=None, active_core_ports=None, PE_fifos_bypass_config=None):
        instance_name = f"{pnr_tag}{node_num}"
        tile = self.tile_circuits[(x, y)]
        core_: ConfigurableCore = None
        result = None
        if pnr_tag is None:
            # backward-compatible with the old code usage
            result = tile.core.get_config_bitstream(instr)
            core_ = tile.core
        else:
            cores = [tile.core] + tile.additional_cores
            has_configured = False
            for core in cores:
                if has_configured:
                    break
                tags = core.pnr_info()
                if not isinstance(tags, list):
                    tags = [tags]
                for tag in tags:
                    if tag.tag_name == pnr_tag:
                        if 'P' in pnr_tag or 'p' in pnr_tag:
                            result = core.get_config_bitstream(
                                instr, active_core_ports=active_core_ports[instance_name],
                                x=x, y=y, PE_fifos_bypass_config=PE_fifos_bypass_config
                            )
                        else:
                            result = core.get_config_bitstream(instr)
                        has_configured = True
                        core_ = core
                        break
        assert result is not None, f"Unable to get config bitstream from tile ({x}, {y})"
        assert core_ is not None, f"Unable to get config bitstream from tile ({x}, {y})"
        feature_addr = tile.features().index(core_)
        for i in range(len(result)):
            entry = result[i]
            if len(entry) == 2:
                reg_index, data = result[i]
                addr = self.get_config_addr(reg_index, feature_addr, x, y)
            else:
                assert len(entry) == 3
                reg_index, idx_offset, data = result[i]
                addr = self.get_config_addr(reg_index,
                                            feature_addr + idx_offset, x, y)
            result[i] = (addr, data)
        return result

    def __get_core_info(self) -> Dict[str, Tuple[PnRTag, List[PnRTag]]]:
        result = {}
        for coord in self.tile_circuits:
            tile = self.tile_circuits[coord]
            cores = [tile.core] + tile.additional_cores
            for core in cores:
                info = core.pnr_info()
                core_name = core.name()
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
            for tag in tags:  # type: PnRTag
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

    def dump_pnr(self, dir_name, design_name, max_num_col=None):
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        dir_name = os.path.abspath(dir_name)
        if max_num_col is None:
            max_num_col = self.x_max + 1

        graph_path_dict = {}
        for bit_width, graph in self.__graphs.items():
            graph_path = os.path.join(dir_name, f"{bit_width}.graph")
            graph_path_dict[bit_width] = graph_path
            graph.dump_graph(graph_path, max_num_col)

        # generate the layout file
        layout_file = os.path.join(dir_name, f"{design_name}.layout")
        self.__dump_layout_file(layout_file, max_num_col)
        pnr_file = os.path.join(dir_name, f"{design_name}.info")
        with open(pnr_file, "w+") as f:
            f.write(f"layout={layout_file}\n")
            graph_configs = [f"{bit_width} {graph_path_dict[bit_width]}" for
                             bit_width in self.__graphs]
            graph_config_str = " ".join(graph_configs)
            f.write(f"graph={graph_config_str}\n")

    def __dump_layout_file(self, layout_file, max_num_col):
        # empty tiles first first
        with open(layout_file, "w+") as f:
            f.write("LAYOUT   0 20\nBEGIN\n")
            for y in range(self.y_max + 1):
                for x in range(max_num_col):
                    coord = (x, y)
                    if coord not in self.tile_circuits:
                        f.write("1 ")
                    else:
                        f.write("0 ")
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
                        for x in range(max_num_col):
                            coord = (x, y)
                            if coord not in self.tile_circuits:
                                f.write("0 ")
                            else:
                                tile = self.tile_circuits[coord]
                                cores = [tile.core] + tile.additional_cores
                                core_names = [core.name() for core in cores]
                                if core_name not in core_names:
                                    f.write("0 ")
                                else:
                                    f.write("1 ")
                        f.write("\n")
                    f.write("END\n")
            # handle registers
            assert "r" not in tag_to_name
            r_locs = self.__get_registered_tile()
            f.write(f"LAYOUT r {default_priority} 0\nBEGIN\n")
            for y in range(self.y_max + 1):
                for x in range(max_num_col):
                    if (x, y) in r_locs:
                        f.write("20 ")
                    else:
                        f.write("0 ")
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
                          self.__lifted_ports,
                          double_buffer=self.double_buffer,
                          give_north_io_sbs=self.give_north_io_sbs)
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

    def get_skip_addr(self):
        result = set()
        for y in range(self.y_min, self.y_max + 1):  # y_max is inclusive
            for x in range(self.x_min, self.x_max + 1):  # x_max is inclusive
                if (x, y) not in self.tile_circuits:
                    continue
                tile = self.tile_circuits[(x, y)]
                for idx, feat in enumerate(tile.features()):
                    if hasattr(feat, "skip_compression") and \
                            feat.skip_compression:
                        # need to skip all address in this feature space
                        # compute the number address here
                        num_addr = 1 << self.config_addr_width
                        for reg_addr in range(num_addr):
                            addr = self.get_config_addr(reg_addr, idx, x, y)
                            result.add(addr)
        return result

    def get_top_port_name(self, node: Node):
        interface_node = None
        if self.__lifted_ports:
            # this is straight forward
            interface_node = node
        else:
            assert (len(node.get_conn_in()) == 0) ^ (len(node) == 0), \
                "External ports cannot have incoming connections"
            x, y = node.x, node.y
            bit_width = node.width
            tile = self.__tiles[(x, y)][bit_width]
            for port_name, port_node in tile.ports.items():
                if len(port_node.get_conn_in()) == 0 and len(port_node):
                    # need to find the port that doesn't connect to anything
                    # but has the same bit width
                    interface_node = port_node
                    break
        for port_name, n in self.__interface.items():
            if n == interface_node:
                return port_name
        raise Exception(str(node) + " does not have corresponding "
                                    "top-level ports")

    def __get_top_port_by_coord(self, coord, bit_width, func):
        assert not self.__lifted_ports
        tile = self.__tiles[coord][bit_width]
        interface_node = None
        for _, port_node in tile.ports.items():
            if len(port_node.get_conn_in()) == 0 and len(port_node) == 0:
                # make sure it is input port
                interface_node = port_node
                for port_name, n in self.__interface.items():
                    if n == interface_node:
                        if func(port_name):
                            return port_name
        raise Exception(f"{coord} does not have corresponding "
                        f"top-level ports")

    def get_top_input_port_by_coord(self, coord, bit_width):
        def predicate(port_name):
            return self.ports[port_name].base_type().is_input()

        return self.__get_top_port_by_coord(coord, bit_width, predicate)

    def get_top_output_port_by_coord(self, coord, bit_width):
        def predicate(port_name):
            return self.ports[port_name].base_type().is_output()

        return self.__get_top_port_by_coord(coord, bit_width, predicate)

    def get_graph(self, bit_width: int):
        return self.__graphs[bit_width]

    def get_bit_widths(self):
        return list(self.__graphs.keys())

    def name(self):
        return "Interconnect"
