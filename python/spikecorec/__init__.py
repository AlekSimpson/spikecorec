import sys as _sys

from ._spikecorec import *  # noqa: F401, F403
from ._spikecorec import __version__, nml, units  # noqa: F401

# The submodules are attributes of the extension; registering them makes
# `import spikecorec.nml` and `from spikecorec.units import parse_quantity` work too.
_sys.modules[__name__ + ".nml"] = nml
_sys.modules[__name__ + ".units"] = units

__all__ = [
    "__version__",
    "nml",
    "units",
    # logging and constants
    "set_log_level",
    "LOG_PATH",
    "NEVER_SPIKED_TICK",
    "MAX_RANK_FLOAT4_STRIDE",
    "DEFAULT_BRANCHING_FACTOR",
    # the engine and its weight matrix
    "Spike_Engine",
    "Weight_Matrix",
    "Weight_Stats",
    "Scale_Result",
    "K2_Tree",
    # topology generators
    "square_torus",
    "small_world_torus",
    "random_fixed_outdegree",
    # recording
    "Output_File_Format",
    "Recording_Config",
    "Recording_Selection",
    "Spire_Compression",
    "resolve_spire_compression",
    "read_spire_recording",
    "Spire_Writer",
    "Spire_Reader",
    "Simulation_Recorder",
]
