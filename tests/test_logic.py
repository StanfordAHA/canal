from canal.circuit import *
from gemstone.common.testers import BasicTester
import tempfile


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

    tester.expect(circuit.ready_out, 1)
    # push in data
    tester.poke(circuit.valid_in, 1)
    tester.poke(circuit.I, 42)
    tester.eval()
    tester.step(2)
    tester.expect(circuit.ready_out, 1)
    tester.poke(circuit.valid_in, 1)
    tester.poke(circuit.I, 43)
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
    tester.expect(circuit.valid_out, 1)
    tester.expect(circuit.O, 43)

    with tempfile.TemporaryDirectory() as tempdir:
        tester.compile_and_run(target="verilator",
                               magma_output="coreir-verilog",
                               directory=tempdir,
                               flags=["-Wno-fatal"])


if __name__ == "__main__":
    test_reg_fifo()
