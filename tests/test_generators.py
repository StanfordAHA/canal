import pytest
import tempfile

import fault
import fault.random
from gemstone import BasicTester
from hwtypes import BitVector
import magma as m

from canal.generators import CB
from canal.graph import PortNode, SwitchBoxSide, SwitchBoxIO, SwitchBoxNode


# TODO(rsetaluri): Get rid of this function once it is part of the
# CircuitBuilder API.
def _finalize(cls: type, *args, **kwargs):

    class _Container(m.Circuit):
        c = cls(*args, **kwargs)

    assert len(_Container.instances) == 1
    return type(_Container.instances[0])


@pytest.mark.parametrize('num_tracks', [2, 5])
@pytest.mark.parametrize('bit_width', [1, 16])
def test_cb(num_tracks: int, bit_width: int):
    ADDR_WIDTH = 8
    DATA_WIDTH = 32

    # Setup graph.
    port_node = PortNode(name="data_in", x=0, y=0, width=bit_width)
    for i in range(num_tracks):
        sb = SwitchBoxNode(x=0, y=0, track=i, width=bit_width,
                           side=SwitchBoxSide.NORTH,
                           io=SwitchBoxIO.SB_IN)
        sb.add_edge(port_node)

    # Finalize circuit.
    ckt = _finalize(CB, port_node, ADDR_WIDTH, DATA_WIDTH)

    #assert ckt.mux.height == num_tracks

    tester = BasicTester(ckt, ckt.clk, ckt.reset)

    for config_data in [BitVector[DATA_WIDTH](x) for x in range(num_tracks)]:
        tester.reset()
        tester.configure(BitVector[ADDR_WIDTH](0), config_data)
        tester.configure(BitVector[ADDR_WIDTH](0), config_data + 1, False)
        tester.config_read(BitVector[ADDR_WIDTH](0))
        tester.eval()
        tester.expect(ckt.read_config_data, config_data)
        inputs = [fault.random.random_bv(bit_width) for _ in range(num_tracks)]
        for i, input_ in enumerate(inputs):
            tester.poke(ckt.I[i], BitVector[bit_width](input_))
        tester.eval()
        tester.expect(ckt.O, inputs[config_data.as_uint()])

    with tempfile.TemporaryDirectory() as tempdir:
        tester.compile_and_run(target="verilator",
                               magma_output="coreir-verilog",
                               directory=tempdir,
                               flags=["-Wno-fatal"])
