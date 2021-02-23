import magma as m
from gemstone.configurable import Configurable

from canal.graph.node import Node


class InterconnectConfigurable(Configurable):
    def _add_mux_config(self, node: Node, mux: m.Circuit):
        MuxT = type(mux)
        if isinstance(MuxT, m.Wire):
           return None
        assert isinstance(MuxT, m.Mux)
        config_name = make_mux_sel_name(node)
        size = 1 if isinstance(mux.S, m.Bit) else len(mux.S)
        self.add_config(config_name, size)
        config_value = self._get_value(config_name)
        if size == 1:
            config_value = config_value[0]
        mux.S @= config_value
        return config_name

    # TODO(rsetaluri): Get rid of this once in magma:master.
    def _open(self):
        from magma.circuit import _DefinitionContextManager
        return _DefinitionContextManager(self._context)


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
    node_name = make_name(str(node))
    if height <= 1:
        return m.Wire(m.Bits[node.width])(name=f"WIRE_{node_name}")
    inst_name = node_name if "MUX" in node_name else f"MUX_{node_name}"
    inst = m.Mux(height, m.Bits[node.width])(name=inst_name)
    # NOTE(rsetaluri): This is a hack!
    inst.I = m.Array[height, m.Bits[node.width]](
        list(getattr(inst, f"I{i}") for i in range(height)))
    return inst
