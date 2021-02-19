import magma as m
from gemstone.configurable import Configurable

from canal.graph.node import Node


class InterconnectConfigurable(Configurable):
    def _add_mux_config(self, node: Node, mux: m.Circuit):
        MuxT = type(mux)
        if isinstance(MuxT, m.Wire):
            return None
        assert isinstance(MuxT, m.Mux):
        config_name = _make_mux_sel_name(node)
        self.add_config(config_name, len(mux.S))
        mux.S @= self._register_set.get_value(config_name)
        return config_name


def make_mux_sel_name(node: Node):
    return f"{make_name(str(node))}_sel"


def make_name(name: str):
    tokens = " (),"
    for t in tokens:
        name = name.replace(t, "_")
    name = name.replace("__", "_")
    if name[-1] == "_":
        name = name[:-1]
    return name


def make_mux_or_wire(node: Node):
    height = len(node.get_conn_in())
    node_name = create_name(str(node))
    if height <= 1:
        return m.Wire(m.Bits[node.width])(name=f"WIRE_{node_name}")
    name = node_name if "MUX" in node_name else f"MUX_{node_name}"
    return m.Mux(height, m.Bits[node.width])(name=name)
