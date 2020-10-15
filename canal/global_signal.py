"""Useful pass on interconnect that doesn't deal with the routing network"""
import magma
import mantle
import enum
import math
from gemstone.common.transform import pass_signal_through, or_reduction
from gemstone.generator.const import Const
from gemstone.generator.from_magma import FromMagma
from gemstone.common.configurable import ConfigurationType
from .interconnect import Interconnect
from .util import IOSide, get_array_size


@enum.unique
class GlobalSignalWiring(enum.Enum):
    Fanout = enum.auto()
    Meso = enum.auto()
    ParallelMeso = enum.auto()


def get_x_range_cores(interconnect: Interconnect):
    x_min, x_max, = interconnect.x_min, interconnect.x_max
    x_range = []
    for x in range(x_min, x_max + 1):
        column = interconnect.get_column(x)
        # skip the margin
        column = [entry for entry in column if "config" in entry.ports]
        if len(column) == 0:
            continue
        else:
            x_range.append(x)
    x_min = x_range[0]
    x_max = x_range[-1]
    return x_min, x_max


def apply_global_fanout_wiring(interconnect: Interconnect, io_sides: IOSide = IOSide.None_):
    # straight-forward fanout for global signals
    x_min, x_max, = get_x_range_cores(interconnect)
    global_ports = interconnect.globals
    cgra_width = x_max - x_min + 1
    interconnect_read_data_or = \
        FromMagma(mantle.DefineOr(cgra_width, interconnect.config_data_width))
    interconnect_read_data_or.instance_name = "read_config_data_or_final"
    # this is connected on a per-column bases
    for x in range(x_min, x_max + 1):
        column = interconnect.get_column(x)
        # skip the margin
        column = [entry for entry in column if "config" in entry.ports]
        # handle the read config
        column_read_data_or = \
            FromMagma(mantle.DefineOr(len(column),
                      interconnect.config_data_width))
        column_read_data_or.instance_name = f"read_config_data_or_col_{x}"
        for idx, tile in enumerate(column):
            for signal_name in global_ports:
                interconnect.wire(interconnect.ports[signal_name],
                                  tile.ports[signal_name])
            # connect the tile to the column read_data inputs
            interconnect.wire(column_read_data_or.ports[f"I{idx}"],
                              tile.ports.read_config_data)

        # wire it to the interconnect_read_data_or
        idx = x - x_min
        interconnect.wire(interconnect_read_data_or.ports[f"I{idx}"],
                          column_read_data_or.ports.O)

    # wiring the read_config_data
    interconnect.wire(interconnect.ports.read_config_data,
                      interconnect_read_data_or.ports.O)

    return interconnect_read_data_or


def apply_global_meso_wiring(interconnect: Interconnect, io_sides: IOSide = IOSide.None_):
    # "river routing" for global signal
    global_ports = interconnect.globals
    x_min, x_max, = get_x_range_cores(interconnect)
    cgra_width = x_max - x_min + 1
    interconnect_read_data_or = \
        FromMagma(mantle.DefineOr(cgra_width, interconnect.config_data_width))
    interconnect_read_data_or.instance_name = "read_config_data_or_final"

    # looping through on a per-column bases
    for x in range(x_min, x_max + 1):
        column = interconnect.get_column(x)
        # skip the margin
        column = [entry for entry in column if "config" in entry.ports]
        if len(column) == 0:
            continue
        # wire global inputs to first tile in column
        for signal in global_ports:
            interconnect.wire(interconnect.ports[signal],
                              column[0].ports[signal])
        # first pass to make signals pass through
        # pre_ports keep track of ports created by pass_signal_through
        pre_ports = {}
        for signal in global_ports:
            pre_ports[signal] = []
            for tile in column:
                # use the transform pass
                pre_port = pass_signal_through(tile, signal)
                pre_ports[signal].append(pre_port)
        # second pass to wire them up
        for i in range(len(column) - 1):
            next_tile = column[i + 1]
            for signal in global_ports:
                pre_port = pre_ports[signal][i]
                interconnect.wire(pre_port,
                                  next_tile.ports[signal])

        # read_config_data
        # Call tile function that adds input for read_data,
        # along with OR gate to reduce input read_data with
        # that tile's read_data
        # ports_in keep track of new ports created by or_reduction
        ports_in = []
        for tile in column:
            port_in = or_reduction(tile, "read_data_mux", "read_config_data",
                                   interconnect.config_data_width)
            ports_in.append(port_in)

        # Connect 0 to first tile's read_data input
        interconnect.wire(ports_in[0],
                          Const(magma.Bits[interconnect.config_data_width](0)))

        # connect each tile's read_data output to next tile's
        # read_data input
        for i, tile in enumerate(column[:-1]):
            interconnect.wire(tile.ports.read_config_data,
                              ports_in[i + 1])
        # Connect the last tile's read_data output to the global OR
        idx = x - x_min
        interconnect.wire(interconnect_read_data_or.ports[f"I{idx}"],
                          column[-1].ports.read_config_data)

    # wiring the read_config_data
    interconnect.wire(interconnect.ports.read_config_data,
                      interconnect_read_data_or.ports.O)

    return interconnect_read_data_or


def apply_global_parallel_meso_wiring(interconnect: Interconnect,
                                      io_sides: IOSide = IOSide.None_, num_cfg: int = 1):

    interconnect_read_data_or = apply_global_meso_wiring(interconnect)
    # interconnect must have config port
    assert "config" in interconnect.ports
    # there must be at least one configuration path
    assert num_cfg >= 1

    interconnect.remove_port("config")
    # this is not a typo. Total number of bits in configuration address
    # is same as config_data
    config_data_width = interconnect.config_data_width
    interconnect.add_port(
        "config",
        magma.In(magma.Array[num_cfg,
                             ConfigurationType(config_data_width,
                                               config_data_width)]))

    cgra_width = interconnect.x_max - interconnect.x_min + 1
    # number of CGRA columns one configuration controller is in charge of
    col_per_config = math.ceil(cgra_width / num_cfg)

    # looping through on a per-column bases
    for x_coor in range(interconnect.x_min, interconnect.x_max + 1):
        column = interconnect.get_column(x_coor)
        # skip tiles with no config
        column = [entry for entry in column if "config" in entry.ports]
        # select which configuration controller is connected to that column
        config_sel = int(x_coor/col_per_config)
        # wire configuration ports to first tile in column
        interconnect.wire(interconnect.ports.config[config_sel],
                          column[0].ports.config)

    return interconnect_read_data_or
