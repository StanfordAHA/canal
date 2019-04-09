from canal.interconnect import Interconnect
from canal.circuit import CB, SB, Node
from typing import Dict, List


class InterconnectModel:
    def __init__(self, connected_pairs: Dict[Node, Node],
                 interface: List[Node]):
        self._connected_pairs = connected_pairs
        self.interface = interface

        self._values: Dict[Node, int] = {}

    def eval(self):
        pass

    def set_value(self, node, value):
        assert node in self.interface
        self._values[node] = value

    def get_value(self, node):
        assert node in self.interface
        return self._values[node]


class InterconnectModelCompiler:
    def __init__(self, interconnect: Interconnect):
        self.interconnect = interconnect
        self.config_data = []

    def configure(self, addr: int, data: int):
        # store the configuration
        self.config_data.append((addr, data))

    def compile(self) -> InterconnectModel:
        # based on the configuration, prune the graph
        connected_pair = self.config_graph()
        interface = self.interconnect.interface()

    def decode_bitstream_feature(self, addr):
        tile_id = addr & (0xFFFFFFFF >> (32 - self.interconnect.tile_id_width))
        x = tile_id >> (self.interconnect.tile_id_width / 2)
        y = tile_id & (0xFFFFFFFF >> (32 - self.interconnect.tile_id_width / 2))

        feat_addr = (addr >> self.interconnect.tile_id_width) & \
                    (0xFFFFFFFF >> (32 - self.interconnect.config_addr_width))
        tile = self.interconnect.tile_circuits[(x, y)]
        features = tile.features()
        return tile, features[feat_addr]

    def get_bitstream_reg_addr(self, addr):
        return addr & (0xFFFFFFFF >> (32 - self.interconnect.config_addr_width))

    def config_graph(self):
        connected_pair = {}
        for addr, data in self.config_data:
            tile, feat = self.decode_bitstream_feature(addr)
            reg_addr = self.get_bitstream_reg_addr(addr)
            if isinstance(feat, CB):
                assert reg_addr == 0
                conn_in = feat.node.get_conn_in()
                assert data < len(conn_in)
                to_node = feat.node
                from_node = conn_in[data]
                if from_node not in connected_pair:
                    # ordering doesn't matter
                    connected_pair[from_node] = set()
                connected_pair[from_node].add(to_node)
            elif isinstance(feat, SB):
                reg_names = list(feat.registers)
                reg_names.sort()
                reg_name = reg_names[reg_addr]
                to_node = feat.mux_name_to_node[reg_name]
                conn_in = to_node.get_conn_in()
                assert data < len(conn_in)
                from_node = conn_in[data]
                if from_node not in connected_pair:
                    # ordering doesn't matter
                    connected_pair[from_node] = set()
                connected_pair[from_node].add(to_node)
        return connected_pair
