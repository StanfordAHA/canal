from canal.interconnect import Interconnect, InterconnectGraph
from canal.circuit import Node, RegisterNode, PortNode
from typing import Dict, Set, Tuple, List, Union
from gemstone.common.core import Core


class InterconnectModel:
    def __init__(self, graphs: Dict[int, InterconnectGraph],
                 nodes: Set[Node],
                 interface: Dict[str, Node],
                 cores: Dict[Tuple[int, int], Core]):
        self.graphs = graphs
        self.interface = interface

        self._values: Dict[Node, int] = {}
        self._reg_values: Dict[Node, int] = {}

        self.nodes = nodes
        self.cores = cores

        # initialize everything to 0
        for node in nodes:
            self._values[node] = 0

            if isinstance(node, RegisterNode):
                self._reg_values[node] = 0

        # sort to get the eval order
        self.__eval_nodes = self.__topological_sort()

    def eval(self):
        # eval in order
        for node in self.__eval_nodes:
            from_nodes = node.get_conn_in()
            is_output_port = isinstance(node, PortNode)
            if is_output_port:
                for from_node in from_nodes:
                    if not isinstance(from_node, PortNode):
                        is_output_port = False
                        break
            values = {}
            for from_node in from_nodes:
                if isinstance(from_node, RegisterNode):
                    values[from_node] = self._reg_values[from_node]
                else:
                    values[from_node] = self._values[from_node]
            if len(values) == 0:
                # input ports, do nothing
                continue
            elif not is_output_port:
                assert len(values) == 1, "Normal nodes can only have one input"
                # save to values
                self._values[node] = list(values.values())[0]
            else:
                # prepare for the inputs
                inputs = {}
                for from_node, value in values.items():
                    inputs[from_node.name] = value
                # evaluate on the core
                x, y = node.x, node.y
                assert (x, y) in self.cores
                core = self.cores[(x, y)]
                result = core.eval_model(**inputs)
                result_name = node.name
                value = result[result_name] if result_name in result else 0
                self._values[node] = value

        # propagate the register value
        for node in self._reg_values:
            self._reg_values[node] = self._values[node]

    def __topological_sort(self):
        visited = {}
        for node in self.nodes:
            visited[node] = False
        stack = []

        def visit(node_):
            visited[node_] = True

            for node__ in node_:
                if not visited[node__]:
                    visit(node__)
            stack.insert(0, node_)

        for node in self.nodes:
            if not visited[node]:
                visit(node)

        return stack

    def set_value(self, node_name: Union[str, Node], value: int):
        if isinstance(node_name, Node):
            node = InterconnectGraph.locate_node(self.graphs[node_name.width],
                                                 node_name)
            self._values[node] = value
        else:
            assert node_name in self.interface
            self._values[self.interface[node_name]] = value

    def get_value(self, node_name: Union[str, Node]) -> int:
        if isinstance(node_name, Node):
            node = InterconnectGraph.locate_node(self.graphs[node_name.width],
                                                 node_name)
            return self._values[node]
        else:
            assert node_name in self.interface
            return self._values[self.interface[node_name]]


class InterconnectModelCompiler:
    def __init__(self, interconnect: Interconnect):
        self.interconnect = interconnect
        self._routes: List[List[Node]] = []
        self._core_instr = {}

    def configure_route(self, routes: List[List[Node]]):
        # store the configuration
        self._routes = routes

    def set_core_instr(self, x: int, y: int, instr):
        self._core_instr[(x, y)] = instr

    def compile(self) -> InterconnectModel:
        # clone the graph first
        ic = self.interconnect.clone()
        graphs, nodes = self.__prune_graph(ic)
        cores = self.__get_cores()
        interface = self.__get_interface(ic.interface(), nodes)
        return InterconnectModel(graphs, nodes, interface, cores)

    def __set_core_instr(self, ic: Interconnect):
        for (x, y), instr in self._core_instr:
            tile = ic.tile_circuits[(x, y)]
            core = tile.core
            instr = self._core_instr[(x, y)]
            core.configure_model(instr)

    def __get_cores(self):
        result = {}
        for (x, y), tile in self.interconnect.tile_circuits.items():
            result[(x, y)] = tile.core
        return result

    @staticmethod
    def __get_node_set(connected_pair: Dict[Node, Node]):
        result = set()
        for node_to, node_from in connected_pair.items():
            result.add(node_to)
            result.add(node_from)
        return result

    @staticmethod
    def __get_interface(interface: Dict[str, Node], nodes: Set[Node]):
        interface_set = set(interface.values())
        result_set = interface_set.intersection(nodes)
        result = {}
        for name, node in interface.items():
            if node in result_set:
                result[name] = node
        return result

    def __prune_graph(self, ic: Interconnect) -> Tuple[Dict[int,
                                                            InterconnectGraph],
                                                       Set[Node]]:
        graphs = {}
        nodes = set()
        for bit_width in self.interconnect.get_bit_widths():
            graphs[bit_width] = ic.get_graph(bit_width)

        # clear out every connection being made in the graph
        # this is fine since we have clone the graph
        for _, graph in graphs.items():
            for coord in graph:
                tile = graph[coord]
                for sb in tile.switchbox.get_all_sbs():
                    sb.clear()
                for _, node in tile.switchbox.registers.items():
                    node.clear()
                for _, node in tile.switchbox.reg_muxs.items():
                    node.clear()
                for _, node in tile.ports.items():
                    node.clear()
        # also need to add pseudo connection between the port input and port
        # output
        # after this point we can't use the graph to route any more
        for route in self._routes:
            for i in range(len(route) - 1):
                from_node_ref = route[i]
                to_node_ref = route[i + 1]

                graph = ic.get_graph(from_node_ref.width)
                from_node = InterconnectGraph.locate_node(graph, from_node_ref)
                to_node = InterconnectGraph.locate_node(graph, to_node_ref)
                from_node.add_edge(to_node)

                nodes.add(from_node)
                nodes.add(to_node)

        # second pass to fake port connections so that we can determine the
        # evaluate order in the graph
        for route in self._routes:
            for node in route[1:]:
                # node is a sink node
                if not isinstance(node, PortNode):
                    continue
                graph = ic.get_graph(node.width)
                sink_node = InterconnectGraph.locate_node(graph, node)
                for src_route in self._routes:
                    src_node = InterconnectGraph.locate_node(graph,
                                                             src_route[0])
                    if not isinstance(src_node, PortNode):
                        continue
                    if src_node.width == node.width and \
                            src_node.x == node.x and src_node.y == node.y:
                        sink_node.add_edge(src_node)

        return graphs, nodes
