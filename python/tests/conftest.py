"""Shared scaffolding for the spikecorec Python tests.

Everything here describes the ONE checked-in fixture model, python/tests/fixtures/
glif3_ring_network_top.nml: eight GLIF3 cells in a unidirectional ring, neuron 0 driven by a
pulseGenerator, each neuron wired to the next through a current-based alpha synapse. The
constants below are the model's own numbers, restated so a test can assert against them
rather than against whatever the engine happens to report.
"""
import os
from pathlib import Path

import pytest

import spikecorec

FIXTURES_DIRECTORY = Path(__file__).resolve().parent / "fixtures"

# The network alone (no <Simulation>), for a test that wants to state its own dt, run length
# or recordings around it.
RING_NETWORK_PATH = FIXTURES_DIRECTORY / "glif3_ring_network.nml"

# The complete model, which is what SpikeEngine is handed.
RING_MODEL_PATH = FIXTURES_DIRECTORY / "glif3_ring_network_top.nml"

RING_NEURON_COUNT = 8

# <Simulation length="300ms" step="0.1ms"/>.
RING_STEP_SECONDS = 1e-4
RING_TICK_COUNT = 3000

# <connectionWD weight="1" delay="1ms"/> at 0.1ms per tick.
RING_EDGE_WEIGHT = 1.0
RING_EDGE_DELAY_TICKS = 10


@pytest.fixture(autouse=True)
def quiet_engine_logging():
    """Keeps the engine's per-construction info chatter out of the test output.

    Warnings stay on: the engine warns about things a test should not silently pass over
    (enable_hebbian_learning doing nothing, a recording selecting a variable its cell type
    does not declare).
    """
    spikecorec.set_log_level("warn")


def build_ring_engine(**keyword_arguments):
    """A fresh engine on the checked-in ring model. Keyword arguments go to SpikeEngine."""
    return spikecorec.SpikeEngine(str(RING_MODEL_PATH), **keyword_arguments)


def write_ring_model(directory, output_file_elements="", length="30ms", step="0.1ms"):
    """Writes a LEMS model into `directory` around the checked-in ring network.

    The network file is included by ABSOLUTE path (includes resolve against the including
    file's own directory, and `directory` is pytest's tmp_path, nowhere near the fixtures).
    `output_file_elements` is spliced into the <Simulation> body verbatim, which is how a
    test states recording paths that live under tmp_path instead of in the working tree.

    Returns the path of the file to hand SpikeEngine.
    """
    model_path = os.path.join(str(directory), "ring_model.xml")
    with open(model_path, "w") as model_file:
        model_file.write(
            "<Lems>\n"
            '  <Target component="ringSimulation"/>\n'
            '  <Include file="{network}"/>\n'
            '  <Simulation id="ringSimulation" length="{length}" step="{step}"'
            ' target="Glif3RingNet">\n'
            "{outputs}"
            "  </Simulation>\n"
            "</Lems>\n".format(
                network=RING_NETWORK_PATH,
                length=length,
                step=step,
                outputs=output_file_elements,
            )
        )
    return model_path
