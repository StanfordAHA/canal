from canal.circuit import *
from canal.logic import RegFIFO
from gemstone.common.testers import BasicTester
import shutil
import tempfile
import pytest


def test_reg_fifo():
    fifo = FifoRegWrapper(16)
    circuit = fifo.circuit()
    tester = BasicTester(circuit,
                         circuit.clk,
                         circuit.ASYNCRESET)

    tester.zero_inputs()
    tester.reset()
    tester.poke(circuit.CE, 1)
    # enable fifo mode
    tester.poke(circuit.fifo_en, 1)
    tester.eval()

    tester.expect(circuit.ready_out, 1)
    # push in data
    tester.poke(circuit.valid_in, 1)
    tester.poke(circuit.I, 42)
    tester.eval()
    tester.step(2)
    # should be full
    tester.expect(circuit.ready_out, 0)
    # pop out data
    tester.expect(circuit.valid_out, 1)
    tester.expect(circuit.O, 42)
    tester.poke(circuit.valid_in, 0)
    tester.poke(circuit.ready_in, 1)
    tester.step(2)

    with tempfile.TemporaryDirectory() as tempdir:
        tester.compile_and_run(target="verilator",
                               magma_output="coreir-verilog",
                               directory=tempdir,
                               flags=["-Wno-fatal", "--trace"])


def test_reg_fifo_sr():
    fifo = FifoRegWrapper(16)
    circuit = fifo.circuit()
    tester = BasicTester(circuit,
                         circuit.clk,
                         circuit.ASYNCRESET)

    tester.zero_inputs()
    tester.reset()
    tester.poke(circuit.CE, 1)
    # enable fifo mode
    tester.poke(circuit.fifo_en, 0)

    for i in range(10):
        tester.poke(circuit.I, i + 1)
        tester.eval()
        if i > 1:
            tester.expect(circuit.O, i + 1 - 1)
        tester.step(2)

    with tempfile.TemporaryDirectory() as tempdir:
        tester.compile_and_run(target="verilator",
                               magma_output="coreir-verilog",
                               directory=tempdir,
                               flags=["-Wno-fatal"])


def test_reg_fifo_only_ports():
    """fifo_only: no fifo_en input, so no pipeline-register mode"""
    assert "fifo_en" in FifoRegWrapper(16, use_non_split_fifos=True).ports
    fifo = FifoRegWrapper(16, use_non_split_fifos=True, fifo_only=True)
    assert "fifo_en" not in fifo.ports
    assert fifo.name() == "FifoRegWrapper_16_fifo_only"
    gen = RegFIFO(16, 1, depth=2, fifo_mode_config=False)
    assert gen.name == "reg_fifo_d_2_w_1_D_16_fifo_only"
    assert "fifo_en" not in gen.ports
    with pytest.raises(AssertionError):
        FifoRegWrapper(16, use_non_split_fifos=False, fifo_only=True)


@pytest.mark.skipif(shutil.which("verilator") is None, reason="needs verilator")
def test_reg_fifo_only_holds_on_stall():
    """A fifo_only register holds its tokens while ready_in is low (a plain
    register would keep shifting them out and lose them)."""
    fifo = FifoRegWrapper(16, use_non_split_fifos=True, fifo_only=True)
    circuit = fifo.circuit()
    tester = BasicTester(circuit, circuit.clk, circuit.ASYNCRESET)
    tester.zero_inputs()
    tester.reset()
    tester.poke(circuit.CE, 1)
    tester.poke(circuit.valid_in, 1)
    for v in (42, 43):
        tester.poke(circuit.I, v)
        tester.eval()
        tester.expect(circuit.ready_out, 1)
        tester.step(2)
    tester.poke(circuit.I, 44)
    tester.eval()
    tester.expect(circuit.ready_out, 0)      # full: 44 is not taken
    tester.step(4)
    tester.expect(circuit.valid_out, 1)
    tester.expect(circuit.O, 42)             # still the first token
    tester.poke(circuit.valid_in, 0)
    tester.poke(circuit.ready_in, 1)
    tester.eval()
    tester.step(2)
    tester.expect(circuit.O, 43)
    tester.step(2)
    tester.expect(circuit.valid_out, 0)
    with tempfile.TemporaryDirectory() as tempdir:
        tester.compile_and_run(target="verilator", magma_output="coreir-verilog",
                               directory=tempdir, flags=["-Wno-fatal"])


if __name__ == "__main__":
    test_reg_fifo_sr()
