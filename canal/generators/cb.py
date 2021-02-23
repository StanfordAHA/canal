import magma as m

from canal.generators.common import (InterconnectConfigurable,
                                     make_mux_or_wire, make_name,
                                     make_mux_sel_name)
from canal.graph import PortNode


class CB(InterconnectConfigurable):
    def __init__(self, node: PortNode, config_addr_width: int,
                 config_data_width: int):
        name = make_name(str(node))
        super().__init__(name, config_addr_width, config_data_width)

        self.node = node
        with self._open():
           self.mux = make_mux_or_wire(self.node)

        I, O = self._add_ports(I=m.In(type(self.mux.I)),
                               O=m.Out(type(self.mux.O)))

        self.mux.I @= I
        O @= self.mux.O

        self._add_mux_config(self.node, self.mux)
