"""End-to-end test for the NML/GLIF codegen path exposed to Python (ticket #133).

Exercises `spikecorec.NmlNetworkRunner` — the pybind11 surface over the NeuroML -> GPU codegen
pipeline (parse -> resolve -> lower -> allocate -> assemble -> step_tick) — against a small,
committed 8-neuron GLIF3 ring-network fixture (fixtures/glif3_ring_network_top.nml). This never
touches `SpikeEngine`, the legacy hardcoded engine `NmlNetworkRunner` was added to bypass.
"""
from pathlib import Path

import numpy as np
import spikecorec

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "glif3_ring_network_top.nml"

NEURON_COUNT = 8
TICK_COUNT = 3000
DT_SECONDS = 1e-4


def _run_network():
    network = spikecorec.NmlNetworkRunner(str(FIXTURE_PATH), dt_seconds=DT_SECONDS)
    network.run(TICK_COUNT)
    return network.membrane_potentials(), network.last_spiked()


def test_glif3_ring_network_constructs_and_reports_neuron_count():
    network = spikecorec.NmlNetworkRunner(str(FIXTURE_PATH), dt_seconds=DT_SECONDS)
    assert network.neuron_count == NEURON_COUNT
    assert network.current_tick == 0

    network.step()
    assert network.current_tick == 1


def test_glif3_ring_network_runs_without_nan_or_inf():
    membrane_potentials, _ = _run_network()
    assert membrane_potentials.shape == (NEURON_COUNT,)
    assert np.all(np.isfinite(membrane_potentials))


def test_glif3_ring_network_spikes_occur():
    _, last_spiked = _run_network()
    assert last_spiked.shape == (NEURON_COUNT,)
    # Neuron 0 is directly current-driven by a real pulseGenerator explicitInput for 200ms
    # (2000 ticks at this dt) — at least one neuron must have fired by the end of the run.
    assert np.any(last_spiked >= 0)


def test_glif3_ring_network_is_deterministic_across_repeated_runs():
    membrane_potentials_first, last_spiked_first = _run_network()
    membrane_potentials_second, last_spiked_second = _run_network()

    np.testing.assert_array_equal(membrane_potentials_first, membrane_potentials_second)
    np.testing.assert_array_equal(last_spiked_first, last_spiked_second)
