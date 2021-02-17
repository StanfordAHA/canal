from canal.graph import *


def test_remove_side_sb():
    """Test removing entire side of a switch. It's useful to create a tile that
    has a side missing, such as a tall memory tile.
    """
    BIT_WIDTH = 16
    NUM_TRACKS = 5
    X = 0
    Y = 0

    wires = get_disjoint_sb_wires(NUM_TRACKS)
    switch = SwitchBox(X, Y, NUM_TRACKS, BIT_WIDTH, wires)
    switch.remove_side_sbs(SwitchBoxSide.NORTH, SwitchBoxIO.SB_IN)

    # Should get an IndexError for accessing the deleted node.
    try:
        switch[SwitchBoxSide.NORTH, 0, SwitchBoxIO.SB_IN]
        assert False
    except IndexError:
        pass

    # Now see if there are any connections.
    all_sbs = switch.get_all_sbs()
    for sb in all_sbs:
        side_is_north = sb.side == SwitchBoxSide.NORTH
        io_is_in = sb.io == SwitchBoxIO.SB_IN
        assert not (side_is_north and io_is_in)
        for node in sb:
            side_is_north = node.side == SwitchBoxSide.NORTH
            io_is_in = node.io == SwitchBoxIO.SB_IN
            assert not (side_is_north and io_is_in)

        for node in sb.get_conn_in():
            side_is_north = node.side == SwitchBoxSide.NORTH
            io_is_in = node.io == SwitchBoxIO.SB_IN
            assert not (side_is_north and io_is_in)

    # Removed one side one io, so the total number of sbs left is (2 * 4 - 1) *
    # NUM_TRACKS.
    assert len(all_sbs) == (2 * 4 - 1) * NUM_TRACKS


def test_tiling():
    """Test low-level tiling. We expect the tiling be handled internally.  As a
    result, users do not need to create a graph tile by hand
    """
    width = 16
    interconnect = InterconnectGraph(width)

    tile1 = Tile.create_tile(0, 0, 1, 16, [])
    interconnect.add_tile(tile1)
    # Now we have the following layout:
    #
    #   |-0-|
    assert interconnect.get_size() == (1, 1)

    tile2 = Tile.create_tile(1, 2, 1, 16, [], height=2)
    interconnect.add_tile(tile2)
    # Now we have the following layout:
    #
    #   |-0-|---|
    #   |---|---|
    #   |---|-1-|
    #   |---|-1-|
    assert interconnect.get_size() == (2, 4)

    # Test get tile.
    tile_bottom = interconnect.get_tile(1, 3)
    assert tile_bottom == interconnect.get_tile(1, 2)
    assert tile_bottom == tile2

    # Test check empty.
    assert interconnect.has_empty_tile()

    # Add two more tiles to make it full.
    tile2 = Tile.create_tile(0, 1, 1, 16, [], height=3)
    interconnect.add_tile(tile2)
    # Now we have the following layout:
    #
    #   |-0-|---|
    #   |-2-|---|
    #   |-2-|-1-|
    #   |-2-|-1-|

    tile3 = Tile.create_tile(1, 0, 1, 16, [], height=2)
    interconnect.add_tile(tile3)
    # Now we have the following layout:
    #
    #   |-0-|-3-|
    #   |-2-|-3-|
    #   |-2-|-1-|
    #   |-2-|-1-|

    # Should be full now.
    assert not interconnect.has_empty_tile()


def _create_policy_interconnect(width, num_track):
    """Creates an interconnect with the followign layout:
    
       |-0-|---|-3-|---|---|
       |---|---|---|---|---|
       |-2-|---|---|---|---|
       |---|---|-1-|---|---|
       |-4-|---|---|---|-5-|
    """
    disjoint_wires = get_disjoint_sb_wires(1)

    interconnect = InterconnectGraph(width)
    tile0 = Tile.create_tile(0, 0, width, num_track, disjoint_wires)
    interconnect.add_tile(tile0)
    tile1 = Tile.create_tile(2, 3, width, num_track, disjoint_wires)
    interconnect.add_tile(tile1)
    tile2 = Tile.create_tile(0, 2, width, num_track, disjoint_wires)
    interconnect.add_tile(tile2)
    tile3 = Tile.create_tile(2, 0, width, num_track, disjoint_wires)
    interconnect.add_tile(tile3)
    tile4 = Tile.create_tile(0, 4, width, num_track, disjoint_wires)
    interconnect.add_tile(tile4)
    tile5 = Tile.create_tile(4, 4, width, num_track, disjoint_wires)
    interconnect.add_tile(tile5)

    return interconnect, tile0, tile1, tile2, tile3, tile4, tile5


def test_policy_ignore():
    """Test low-level interconnect policy based connection."""
    WIDTH = 16
    NUM_TRACK = 1
    WIRE_LENGTH = 2

    interconnect, *tiles = _create_policy_interconnect(WIDTH, NUM_TRACK)
    tile0, tile1, tile2, tile3, tile4, tile5 = tiles
    interconnect.connect_switchbox(0, 0, 5, 5, WIRE_LENGTH, NUM_TRACK - 1,
                                   InterconnectPolicy.IGNORE)

    assert interconnect.get_size() == (5, 5)

    # Test connections.

    # 3 <-> 1
    sb_from = tile0.get_sb(SwitchBoxSide.SOUTH, 0, SwitchBoxIO.SB_OUT)
    sb_to = tile2.get_sb(SwitchBoxSide.NORTH, 0, SwitchBoxIO.SB_IN)
    assert sb_to in sb_from

    sb_from = tile0.get_sb(SwitchBoxSide.EAST, 0, SwitchBoxIO.SB_OUT)
    sb_to = tile3.get_sb(SwitchBoxSide.WEST, 0, SwitchBoxIO.SB_IN)
    assert sb_to in sb_from

    sb_from = tile3.get_sb(SwitchBoxSide.SOUTH, 0, SwitchBoxIO.SB_OUT)
    sb_to = tile1.get_sb(SwitchBoxSide.NORTH, 0, SwitchBoxIO.SB_IN)
    assert sb_to not in sb_from

    sb_from = tile4.get_sb(SwitchBoxSide.EAST, 0, SwitchBoxIO.SB_OUT)
    sb_to = tile5.get_sb(SwitchBoxSide.WEST, 0, SwitchBoxIO.SB_IN)
    assert sb_to not in sb_from


def test_policy_pass_through():
    """Test low-level interconnect policy based connection."""
    WIDTH = 16
    NUM_TRACK = 1
    WIRE_LENGTH = 2

    interconnect, *tiles = _create_policy_interconnect(WIDTH, NUM_TRACK)
    tile0, tile1, tile2, tile3, tile4, tile5 = tiles
    interconnect.connect_switchbox(0, 0, 5, 5, WIRE_LENGTH, NUM_TRACK - 1,
                                   InterconnectPolicy.PASS_THROUGH)

    assert interconnect.get_size() == (5, 5)

    # Test connections.
    sb_from = tile3.get_sb(SwitchBoxSide.SOUTH, 0, SwitchBoxIO.SB_OUT)
    sb_to = tile1.get_sb(SwitchBoxSide.NORTH, 0, SwitchBoxIO.SB_IN)
    assert sb_to in sb_from

    sb_from = tile4.get_sb(SwitchBoxSide.EAST, 0, SwitchBoxIO.SB_OUT)
    sb_to = tile5.get_sb(SwitchBoxSide.WEST, 0, SwitchBoxIO.SB_IN)
    assert sb_to in sb_from
