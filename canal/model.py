from canal.interconnect import Interconnect


class InterconnectModel:
    pass


class InterconnectModelCompiler:
    def __init__(self, interconnect: Interconnect):
        self.interconnect = interconnect
        self.config_data = []

    def configure(self, addr: int, data: int):
        # store the configuration
        self.config_data.append((addr, data))

    def compile(self):
        # clone the graphs so that we can prune the graph for fast
        # simulation
        pass

    def clone_graph(self):
        bit_widths = self.interconnect.get_bit_widths()
        result_graph = {}
        for bit_width in bit_widths:
            graph = self.interconnect.get_graph(bit_width)
            new_graph = graph.clone()
            result_graph[bit_width] = new_graph
        ic = Interconnect(result_graph, self.interconnect.config_addr_width,
                          self.interconnect.config_data_width,
                          self.interconnect.tile_id_width,
                          self.interconnect.stall_signal_width)
        return ic
