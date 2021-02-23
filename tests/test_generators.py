import pytest
import tempfile
from typing import Tuple, Union

import fault
import fault.random
from gemstone import BasicTester
from hwtypes import BitVector
import magma as m

from canal.generators import CB, SB
from canal.generators.common import make_mux_sel_name, make_name
from canal.graph import (Node, PortNode, SwitchBoxSide, SwitchBoxIO,
                         SwitchBoxNode, RegisterNode, RegisterMuxNode,
                         DisjointSwitchBox, WiltonSwitchBox, ImranSwitchBox)


# TODO(rsetaluri): Get rid of this function once it is part of the
# CircuitBuilder API.
def _finalize(cls: type, *args, **kwargs):

    class _Container(m.Circuit):
        c = cls(*args, **kwargs)

    assert len(_Container.instances) == 1
    return _Container.c, type(_Container.instances[0])


def _find_reg_mux_node(node: Node) -> Union[Tuple[None, None],
                                            Tuple[RegisterNode,
                                                  RegisterMuxNode]]:
    """Helper function to find register node connected to a switch box node, if
    any.
    """
    for neighbor in node:
        if not isinstance(neighbor, RegisterNode):
            continue
        assert len(neighbor) == 1
        reg_mux = list(neighbor)[0]
        return neighbor, reg_mux
    return None, None


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
    _, ckt = _finalize(CB, port_node, ADDR_WIDTH, DATA_WIDTH)

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


@pytest.mark.parametrize('num_tracks', [2, ])#5])
@pytest.mark.parametrize('bit_width', [1, ])#16])
@pytest.mark.parametrize("TSwitchBox", [DisjointSwitchBox,
                                        #WiltonSwitchBox,
                                        #ImranSwitchBox
])
@pytest.mark.parametrize("reg", [[True, 4], ])#[True, 8], [False, 2]])
def test_sb(num_tracks: int, bit_width: int, TSwitchBox: type,
            reg: Tuple[bool, int]):
    """Tests whether the generated circuit matches the graph representation.
    """
    ADDR_WIDTH = 8
    DATA_WIDTH = 32

    switch_box = TSwitchBox(0, 0, num_tracks, bit_width)
    reg_mode, batch_size = reg

    # Insert registers on every side and track.
    if reg_mode:
        for side in SwitchBoxSide:
            for track in range(num_tracks):
                switch_box.add_pipeline_register(side, track)

    builder, ckt = _finalize(SB, switch_box, ADDR_WIDTH, DATA_WIDTH)

    # Also test the RTL.
    tester = BasicTester(ckt, ckt.clk, ckt.reset)

    # Generate addresses based on mux names, which is used to sort the
    # addresses.
    config_names = sorted(builder.get_configs())

    # Some of the sb nodes may turn into a pass-through wire; we still need to
    # test them. We generate pairs of (config data, expected value). If it's a
    # pass-through wire, we don't configure them, yet we still evaluate the
    # outcome to see if it's connected.
    config_data = []
    test_data = []
    all_sbs = switch_box.get_all_sbs()
    for sb in all_sbs:
        mux_sel_name = make_mux_sel_name(sb)
        if mux_sel_name not in config_names:
            assert sb.io == SwitchBoxIO.SB_IN
            connected_sbs = sb.get_conn_in()
            # For a switch box, where each SB_IN connects to 3 different SB_OUT,
            # the SB_IN won't have any incoming edges.
            assert len(connected_sbs) == 0
            input_sb_name = make_name(str(sb))
            # As a result, we configure the fan-out switch boxes to see if they
            # can receive the signal. Note that this is overlapped with the if
            # statement above.  We also want to test if the register mode can be
            # turned on.
            for connected_sb in sb:  # all of type SwitchBoxNode
                entry = []
                mux_sel_name = make_mux_sel_name(connected_sb)
                assert mux_sel_name in config_names
                assert connected_sb.io == SwitchBoxIO.SB_OUT
                index = connected_sb.get_conn_in().index(sb)
                entry.append(builder.get_config_data(mux_sel_name, index))
                # Also configure the register, if connected.
                reg_node, reg_mux_node = _find_reg_mux_node(connected_sb)
                if reg_mux_node is not None:
                    mux_sel_name = make_mux_sel_name(reg_mux_node)
                    assert mux_sel_name in config_names
                    index = reg_mux_node.get_conn_in().index(reg_node)
                    entry.append(builder.get_config_data(mux_sel_name, index))
                config_data.append(entry)
                output_sb_name = make_name(str(connected_sb))  # get port
                entry = []
                for _ in range(batch_size):
                    entry.append((ckt.interface.ports[input_sb_name],
                                  ckt.interface.ports[output_sb_name],
                                  fault.random.random_bv(bit_width)))
                test_data.append(entry)

    # # Compress the config data.
    # for i in range(len(config_data)):
    #     config_data[i] = compress_config_data(config_data[i])

    # Poke and test, without registers configured.
    assert len(config_data) == len(test_data)
    for i in range(len(config_data)):
        tester.reset()
        configs = config_data[i]
        data = test_data[i]
        for addr, index in configs:
            index = BitVector[DATA_WIDTH](index)
            tester.configure(BitVector[ADDR_WIDTH](addr), index)
            tester.configure(BitVector[ADDR_WIDTH](addr), index + 1, False)
            tester.config_read(BitVector[ADDR_WIDTH](addr))
            tester.eval()
            tester.expect(ckt.read_config_data, index)
        if len(data) == 1:  # if pass through
            for input_port, output_port, value in data:
                tester.poke(input_port, value)
                tester.eval()
                tester.expect(output_port, value)
        else:
            for j in range(len(data)):
                if j != 0:
                    tester.eval()
                    tester.expect(data[j - 1][1], data[j - 1][2])
                input_port, _, value = data[j]
                tester.poke(input_port, value)
                tester.step(2)

    with tempfile.TemporaryDirectory() as tempdir:
        tester.compile_and_run(target="verilator",
                               magma_output="coreir-verilog",
                               directory=tempdir,
                               flags=["-Wno-fatal"])
