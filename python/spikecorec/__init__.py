"""spikecorec — NeuroML-driven spiking-network simulation on Metal/CUDA.

Everything here comes from the compiled `_spikecorec` extension; this module only re-exports
it. See python/docs/API_REFERENCE.md for what each name means.
"""
from ._spikecorec import *  # noqa: F401, F403
from ._spikecorec import __version__  # noqa: F401

# Spelled out rather than left at the two names it used to list. `from spikecorec import *`
# silently handed back only __version__ and set_log_level, which is not the surface anyone
# wants and gave no hint that SpikeEngine was missing from it.
__all__ = [
    "__version__",
    "set_log_level",
    "SpikeEngine",
    "WeightMatrix",
    "WeightStats",
    "SimulationRecorder",
    "read_spire_recording",
]
