from .cyclone import *
from .interconnect import Interconnect
from ordered_set import OrderedSet
from typing import Union, Any


class PowerDomainFixer:
    """This fixer assumes every tile can be turn on or off
    as long as it obey the rules:
    1. if a tile (tile B) is below (y-axis) tile A and tile B is being used,
        tile A can't be turned off even though it's not being used
    2. "Used" means that there is a component (cb, sb or core) being used
    3. we can use an nop to the tile that creates a const as output
    """
    def __init__(self, interconnect: Interconnect,
                 placement: Dict[str, Tuple[int, int]],
                 route: Dict[str, List[List[Node]]]):
        self._interconnect = interconnect
        self._placement = placement
        self._route = route

        self._nop_instruction = {}

    def add_nop_instruction(self, bit_width: int, x: int, y: int, instr):
        if bit_width not in self._nop_instruction:
            self._nop_instruction[bit_width] = {}
        self._nop_instruction[bit_width][(x, y)] = instr

    def fix_pnr(self) -> Tuple[Dict[Tuple[int, int], Any],
                               Dict[str, List[List[Node]]]]:
        always_on, always_off = self.__get_on_off_tiles()
        boundary_tiles = self.__get_boundary_tiles(always_on, always_off)
        target_sb = self.__get_target_sb(boundary_tiles, always_on)
        nop_tiles = self.__get_nop_tiles(target_sb)
        routes = self.__get_new_routing(target_sb)
        instr = self.__get_nop_core_instr(nop_tiles)
        instr += self.__turn_off_tiles(always_on)
        return instr, routes

    def __turn_off_tiles(self, always_off_tiles):
        instr = []
        for loc in always_off_tiles:
            tile_circuit = self._interconnect.tile_circuits[loc]
            features = tile_circuit.features()
            for feat_addr, feat in enumerate(features):
                if "PowerDomainConfig" in feat.name():
                    addr, data = feat.configure(True)
                    addr = self._interconnect.get_config_addr(addr, feat_index, data)
                    instr.append(addr)
                    break
        return instr

    def __get_nop_core_instr(self,
                             nop_tiles: Dict[int, OrderedSet[Tuple[int, int]]]):
        result = {}
        for bit_width, tiles in nop_tiles.items():
            assert bit_width in self._nop_instruction
            instructions = self._nop_instruction[bit_width]
            for loc in tiles:
                instr = instructions[loc]
                if loc not in result:
                    result[loc] = []
                result[loc].append(instr)

        return result

    def __get_on_off_tiles(self):
        locations = OrderedSet()
        # placement
        for _, loc in self._placement.items():
            locations.add(loc)

        # routing
        for _, route in self._route.items():
            for segment in route:
                for node in segment:
                    loc = (node.x, node.y)
                    locations.add(loc)

        available_pos = OrderedSet()
        for loc in self._interconnect.tile_circuits.keys():
            available_pos.add(loc)

        # get the max y value to turn on
        x_columns = {}
        for x, y in locations:
            if x not in x_columns:
                x_columns[x] = OrderedSet()
            x_columns[x].add(y)

        max_y = {}
        for x in x_columns:
            y = max(x_columns[x])
            max_y[x] = y

        # turn off tiles
        always_on = OrderedSet()
        always_off = OrderedSet()
        for x, y in available_pos:
            if x in max_y and y <= max_y[x]:
                always_on.add((x, y))
            else:
                always_off.add((x, y))

        return always_on, always_off

    def __get_boundary_tiles(self,
                             always_on: OrderedSet[Tuple[int, int]],
                             always_off: OrderedSet[Tuple[int, int]]) -> \
            OrderedSet[Tuple[int, int]]:
        result = OrderedSet()
        interconnect = self._interconnect
        graph = interconnect.get_graph(interconnect.get_bit_widths()[0])
        size_x, size_y = graph.get_size()
        for x, y in always_on:
            x_min = max(0, x - 1)
            x_max = min(size_x - 1, x + 1)
            y_min = max(0, y - 1)
            y_max = min(size_y - 1, y + 1)
            points = {(x_min, y), (x_max, y), (x, y_min), (x, y_max)}
            for loc in points:
                if loc in always_off:
                    result.add(loc)
        return result

    def __get_target_sb(self,
                        boundary_tiles: OrderedSet[Tuple[int, int]],
                        always_on: OrderedSet[Tuple[int, int]]):
        result = OrderedSet()
        for loc in boundary_tiles:
            tile_circuit = self._interconnect.tile_circuits[loc]
            # need to get all sbs to see which sb needs to be taken care of
            for _, tile in tile_circuit.tiles.items():
                sbs = tile.switchbox.get_all_sbs()
                for sb in sbs:
                    # we only care about the default connection
                    for node in sb:
                        # the pipeline register complicated things a little
                        # bit
                        if isinstance(node, RegisterMuxNode):
                            assert len(node) == 1
                            node = list(node)[0]
                        pos = node.x, node.y
                        if pos in always_on:
                            result.add(sb)

        return result

    @staticmethod
    def __get_nop_tiles(target_sb: OrderedSet[Node]):
        result = {}
        for node in target_sb:
            x, y = node.x, node.y
            bit_width = node.width
            if bit_width not in result:
                result[bit_width] = OrderedSet()
            result[bit_width].add((x, y))

        return result

    def __get_new_routing(self, target_sb: OrderedSet[Node]):
        result = {}
        finished_tiles = {}
        for width in self._interconnect.get_bit_widths():
            finished_tiles[width] = set()

        for sb in target_sb:
            # get the port node in the incoming connections
            conn_in = sb.get_conn_in()
            port_node: Union[PortNode, None] = None
            for node in conn_in:
                if isinstance(node, PortNode):
                    assert node.width == sb.width
                    assert node.x == sb.x and node.y == sb.y
                    port_node = node
                    break

            assert port_node is not None, f"Unable to find port node from {sb}"
            new_segment = [port_node, sb]
            net_id = f"pd{len(result)}"
            result[net_id] = [new_segment]

            # we also need to toggle the CB connection
            loc = sb.x, sb.y
            width = sb.width
            if loc not in finished_tiles[width]:
                finished_tiles[width].add(loc)

                # find the CB with same bit width
                width = sb.width
                graph = self._interconnect.get_graph(width)
                tile = graph[loc]
                for _, cb_node in tile.ports.items():
                    if cb_node.width == width:
                        conn_in = cb_node.get_conn_in()
                        const_node = None

                        for node in conn_in:
                            if isinstance(node, ConstNode):
                                const_node = node
                                break
                        assert const_node is not None,\
                            "CB has to have a constant connected"
                        new_segment = [const_node, cb_node]
                        net_id = f"pd{len(result)}"
                        result[net_id] = [new_segment]

        return result
