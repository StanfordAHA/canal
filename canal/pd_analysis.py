from .pnr_io import __parse_raw_routing_result
import sys


def get_loc(node_str: str):
    if node_str[0] == "SB":
        _, x, y, _, _, _ = node_str[1:]
        return x, y
    elif node_str[0] == "PORT":
        _, x, y, _ = node_str[1:]
        return x, y
    elif node_str[0] == "REG":
        _, _, x, y, _ = node_str[1:]
        return x, y
    elif node_str[0] == "RMUX":
        _, x, y, _ = node_str[1:]
        return x, y
    else:
        raise Exception("Unknown node " + " ".join(node_str))


def compute_tiles(filename: str):
    raw_routing_result = __parse_raw_routing_result(filename)
    locations = set()
    for net_id, raw_routes in raw_routing_result.items():
        for raw_segment in raw_routes:
            for node_str in raw_segment:
                loc = get_loc(node_str)
                locations.add(loc)
    return locations


def main():
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "routing result", file=sys.stderr)
        exit(1)

    tiles = compute_tiles(sys.argv[1])
    pe = 0
    mem = 0
    io = 0
    for x, y in tiles:
        if y == 0:
            io += 1
        elif x % 4 == 3:
            mem += 1
        else:
            pe += 1
    print("Total tile usage:", len(tiles), "pe:", pe, "mem", mem, "io", io)


if __name__ == "__main__":
    main()
