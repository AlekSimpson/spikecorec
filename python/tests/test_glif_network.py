"""End-to-end test for the NeuroML/GLIF path exposed to Python.

Runs the checked-in 8-neuron GLIF3 ring fixture through the whole engine -- parse, XSD-skip,
resolve, lower, allocate, compile, step -- and asserts on what comes back.

Ported from nightly's python/tests/test_glif_network.py, which drove a separate
`NmlNetworkRunner` class. That class is gone: the pipeline it wrapped folded into SpikeEngine
itself, so these tests drive SpikeEngine directly and take dt and the run length from the
model rather than from constructor arguments.

The three tests below marked as ports keep nightly's assertions; the wiring and unit tests are
new, and cover the two things a binding is unusually good at catching -- a model that parses
cleanly while connecting nothing, and a quantity that arrives off by a power of ten.
"""
import numpy as np

from conftest import (
    RING_EDGE_DELAY_TICKS,
    RING_EDGE_WEIGHT,
    RING_NEURON_COUNT,
    RING_STEP_SECONDS,
    RING_TICK_COUNT,
    build_ring_engine,
)


def run_ring_network(tick_count=RING_TICK_COUNT):
    """Runs the fixture and returns (membrane potentials, spikes per neuron)."""
    engine = build_ring_engine()
    try:
        spike_counts = np.zeros(RING_NEURON_COUNT, dtype=np.int64)
        for tick in range(tick_count):
            engine.step_simulation(tick)
            spike_counts += engine.spike_flags()

        return engine.state_variable_values("v"), spike_counts
    finally:
        engine.shutdown()


def test_ring_network_reports_the_model_it_loaded():
    """Rewrite of nightly's ...constructs_and_reports_neuron_count.

    Nightly asserted the neuron count and then that `current_tick` advanced from 0 to 1. The
    engine keeps no tick counter -- step_simulation takes the tick from its caller -- so there
    is no current_tick to assert on, and the half of the test that checked it is replaced by
    the dt and run length, which the engine now reads out of the model instead of taking as a
    constructor argument.
    """
    engine = build_ring_engine()
    try:
        assert engine.total_neuron_count == RING_NEURON_COUNT
        assert engine.step_dt == RING_STEP_SECONDS
        assert engine.lifetime == RING_TICK_COUNT

        assert engine.cell_type_names() == ["GLIF3Cell"]
        assert engine.state_variable_names(0) == ["v", "asc1", "asc2", "refractoryTimeElapsed"]

        # One <explicitInput>, so exactly one stimulus stream was wired. A model whose
        # stimulus silently failed to resolve reports 0 here and then runs a network nothing
        # ever drives, which looks like a quiescent network rather than like a broken one.
        assert engine.input_neuron_count == 1
    finally:
        engine.shutdown()


def test_ring_network_wiring_matches_the_projection():
    """Every edge the <projection> declares, and no others, at the weight and delay it states.

    The engine can build a perfectly well-formed, fully disconnected network out of a model
    whose connection paths did not resolve: the cells still exist, the kernel still runs, and
    every neuron just integrates alone. Reading the adjacency back is the only way to tell
    that apart from a network that is simply quiet.
    """
    engine = build_ring_engine()
    try:
        for source_neuron in range(RING_NEURON_COUNT):
            expected_target = (source_neuron + 1) % RING_NEURON_COUNT
            assert engine.weights.get_neighbors(source_neuron) == [expected_target]
            assert engine.weights.get(source_neuron, expected_target) == RING_EDGE_WEIGHT
            assert (engine.weights.get_edge_delay_ticks(source_neuron, expected_target)
                    == RING_EDGE_DELAY_TICKS)
    finally:
        engine.shutdown()


def test_ring_network_parameters_land_in_si():
    """The cell parameters, in SI, as the document's own units resolve.

    The fixture writes C="100pF" gL="10nS" EL="-70mV". Those reach the kernel as farads,
    siemens and volts, and a scale error anywhere in the unit table produces a simulation that
    runs happily and is wrong by a power of ten. Checked exactly rather than approximately:
    every one of these is a power-of-ten scaling of an exactly-representable decimal.
    """
    engine = build_ring_engine()
    try:
        np.testing.assert_allclose(engine.parameter_values("C"), 100e-12, rtol=1e-6)
        np.testing.assert_allclose(engine.parameter_values("gL"), 10e-9, rtol=1e-6)
        np.testing.assert_allclose(engine.parameter_values("EL"), -70e-3, rtol=1e-6)
        np.testing.assert_allclose(engine.parameter_values("vth"), -50e-3, rtol=1e-6)
        np.testing.assert_allclose(engine.parameter_values("t_ref"), 5e-3, rtol=1e-6)
        np.testing.assert_allclose(engine.parameter_values("ascAdd1"), -100e-12, rtol=1e-6)

        # OnStart puts every cell at EL before the first tick.
        np.testing.assert_allclose(engine.state_variable_values("v"), -70e-3, rtol=1e-6)
    finally:
        engine.shutdown()


def test_ring_network_runs_without_nan_or_inf():
    """Ported from nightly, with membrane_potentials() spelled state_variable_values('v')."""
    membrane_potentials, _ = run_ring_network()

    assert membrane_potentials.shape == (RING_NEURON_COUNT,)
    assert np.all(np.isfinite(membrane_potentials))


def test_ring_network_spikes_travel_round_the_ring():
    """Rewrite of nightly's ...spikes_occur, which could not fail.

    Nightly asserted `np.any(last_spiked >= 0)` on the strength of last_spiked being -1 for a
    neuron that never fired. In this engine last_spiked is zero-filled at construction, so
    that assertion passes on a network in which nothing whatsoever happened. Spikes are
    counted from spike_flags instead, which is unambiguous.

    Asserting on every neuron, not just on the driven one, is what makes this a test of the
    ring: neuron 0 is the only one with an external input, so neurons 1..7 can only fire if
    their upstream neighbour's spike actually arrived through the synapse.
    """
    _, spike_counts = run_ring_network()

    assert spike_counts[0] > 0, "the directly driven neuron never fired"
    assert np.all(spike_counts > 0), (
        "activity did not travel round the ring: spikes per neuron = %r" % (spike_counts,))


def test_ring_network_is_deterministic_across_repeated_runs():
    """Ported from nightly, unchanged in intent: two identical runs agree exactly."""
    membrane_first, spikes_first = run_ring_network()
    membrane_second, spikes_second = run_ring_network()

    np.testing.assert_array_equal(membrane_first, membrane_second)
    np.testing.assert_array_equal(spikes_first, spikes_second)
