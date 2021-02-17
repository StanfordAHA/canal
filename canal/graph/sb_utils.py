from typing import List, Tuple

from canal.graph.sb import SwitchBoxSide
from canal.graph.sb_container import SwitchBox, InternalWiresType


def _mod(a: int, b: int):
    while a < 0:
        a += b
    return a % b


"""
A helper methods to create switch box internal connections. Implementation is
copied from cyclone:

https://github.com/Kuree/cgra_pnr/blob/dev/cyclone/src/util.cc
"""
def get_disjoint_sb_wires(num_tracks: int) -> InternalWiresType:
    result = []
    for track in range(num_tracks):
        for side_from in SwitchBoxSide:
            for side_to in SwitchBoxSide:
                if side_from == side_to:
                    continue
                result.append((track, side_from,
                               track, side_to))
    return result


def get_wilton_sb_wires(num_tracks: int) -> InternalWiresType:
    w = num_tracks
    result = []
    # t_i is defined as
    #     3
    #   -----
    # 2 |   | 0
    #   -----
    #     1
    for track in range(num_tracks):
        result.append((track, SwitchBoxSide.WEST,
                       track, SwitchBoxSide.EAST))
        result.append((track, SwitchBoxSide.EAST,
                       track, SwitchBoxSide.WEST))
        # t_1, t_3
        result.append((track, SwitchBoxSide.SOUTH,
                       track, SwitchBoxSide.NORTH))
        result.append((track, SwitchBoxSide.NORTH,
                       track, SwitchBoxSide.SOUTH))
        # t_0, t_1
        result.append((track, SwitchBoxSide.WEST,
                       _mod(w - track, w), SwitchBoxSide.SOUTH))
        result.append((_mod(w - track, w), SwitchBoxSide.SOUTH,
                       track, SwitchBoxSide.WEST))
        # t_1, t_2
        result.append((track, SwitchBoxSide.SOUTH,
                       _mod(track + 1, w), SwitchBoxSide.EAST))
        result.append((_mod(track + 1, w), SwitchBoxSide.EAST,
                       track, SwitchBoxSide.SOUTH))
        # t_2, t_3
        result.append((track, SwitchBoxSide.EAST,
                       _mod(2 * w - 2 - track, w), SwitchBoxSide.NORTH))
        result.append((_mod(2 * w - 2 - track, w), SwitchBoxSide.NORTH,
                       track, SwitchBoxSide.EAST))
        # t3, t_0
        result.append((track, SwitchBoxSide.NORTH,
                      _mod(track + 1, w), SwitchBoxSide.WEST))
        result.append((_mod(track + 1, w), SwitchBoxSide.WEST,
                       track, SwitchBoxSide.NORTH))
    return result


def get_imran_sb_wires(num_tracks: int) -> InternalWiresType:
    w = num_tracks
    result = []

    for track in range(num_tracks):
        # f_e1
        result.append((track, SwitchBoxSide.WEST,
                       _mod(w - track, w), SwitchBoxSide.NORTH))
        result.append((_mod(w - track, w), SwitchBoxSide.NORTH,
                       track, SwitchBoxSide.WEST))
        # f_e2
        result.append((track, SwitchBoxSide.NORTH,
                       _mod(track + 1, w), SwitchBoxSide.EAST))
        result.append((_mod(track + 1, w), SwitchBoxSide.EAST,
                       track, SwitchBoxSide.NORTH))
        # f_e3
        result.append((track, SwitchBoxSide.SOUTH,
                       _mod(w - track - 2, w), SwitchBoxSide.EAST))
        result.append((_mod(w - track - 2, w), SwitchBoxSide.EAST,
                       track, SwitchBoxSide.SOUTH))
        # f_e4
        result.append((track, SwitchBoxSide.WEST,
                       _mod(track - 1, w), SwitchBoxSide.SOUTH))
        result.append((_mod(track - 1, w), SwitchBoxSide.SOUTH,
                       track, SwitchBoxSide.WEST))
        # f_e5
        result.append((track, SwitchBoxSide.WEST,
                       track, SwitchBoxSide.EAST))
        result.append((track, SwitchBoxSide.EAST,
                       track, SwitchBoxSide.WEST))
        # f_e6
        result.append((track, SwitchBoxSide.SOUTH,
                       track, SwitchBoxSide.NORTH))
        result.append((track, SwitchBoxSide.NORTH,
                       track, SwitchBoxSide.SOUTH))
    return result


class DisjointSwitchBox(SwitchBox):
    def __init__(self, x: int, y: int, num_track: int, width: int):
        internal_wires = SwitchBoxHelper.get_disjoint_sb_wires(num_track)
        super().__init__(x, y, num_track, width, internal_wires)


class WiltonSwitchBox(SwitchBox):
    def __init__(self, x: int, y: int, num_track: int, width: int):
        internal_wires = SwitchBoxHelper.get_wilton_sb_wires(num_track)
        super().__init__(x, y, num_track, width, internal_wires)


class ImranSwitchBox(SwitchBox):
    def __init__(self, x: int, y: int, num_track: int, width: int):
        internal_wires = SwitchBoxHelper.get_imran_sb_wires(num_track)
        super().__init__(x, y, num_track, width, internal_wires)
