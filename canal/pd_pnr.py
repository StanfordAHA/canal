from .cyclone import *
from .interconnect import Interconnect


class PowerDomainFixer:
    """This fixer assumes every tile added to _pd_tiles can be turn on or off
    as long as it obey the rules:
    1. if a tile (tile B) is below (y-axis) tile A and tile B is being used,
        tile A can't be turned off even though it's not being used
    2. "Used" means that there is a component (cb, sb or core) being used
    """
    def __init__(self, interconnect: Interconnect,
                 placement: Dict[str, Tuple[int, int]],
                 route: Dict[str, List[List[Node]]]):
        self._interconnect = interconnect
        self._placement = placement
        self._route = route

        self._default_muxes: Dict[Tuple[int, int], List[Node]] = {}

        self._pd_tiles = set()

    def add_pd_tile(self, x: int, y: int):
        # tiles can be turned on or off
        self._pd_tiles.add((x, y))

    def __get_on_off_tiles(self):
        locations = set()
        # placement
        for _, loc in self._placement.items():
            locations.add(loc)

        # routing
        for _, route in self._route.items():
            for segment in route:
                for node in segment:
                    loc = (node.x, node.y)
                    locations.add(loc)

        available_pos = set()
        for loc in self._interconnect.tile_circuits.keys():
            available_pos.add(loc)

        # get the max y value to turn on
        x_columns = {}
        for x, y in locations:
            if x not in x_columns:
                x_columns[x] = set()
            x_columns[x].add(y)

        max_y = {}
        for x in x_columns:
            y = max(x_columns[x])
            max_y[x] = y

        # turn off tiles
        always_on = set()
        always_off = set()
        for x, y in available_pos:
            if x in max_y and y <= max_y[x]:
                always_on.add((x, y))
            else:
                always_off.add((x, y))

        return always_on, always_off
