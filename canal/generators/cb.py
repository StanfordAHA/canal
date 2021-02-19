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
        self.mux = make_mux_or_wire(self.node)

        I, O = self._add_ports(I=type(self.mux.I),
                               O=type(self.mux.O))
        self.mux.I @= I
        O @= self.mux.O

        MuxT = type(self.mux)
        if isinstance(MuxT, m.Mux):
            config_name = make_mux_sel_name(self.node)
            self.add_config(config_name, len(self.mux.S))
            self.mux.S @= self._register_set.get_value(config_name)
        else:
            assert isinstance(MuxT, m.Wire)
