"""Smoke test for the compiled _spikecorec extension.

Exercises the pybind11 binding layer end to end: import the module, construct a SpikeEngine,
step it, read a bound accessor back, and shut it down. Enough to catch a bindings.cpp /
engine.h mismatch, which is what left `make python` broken through the engine rewrite.

Ported from nightly's python/tests/test_smoke.py. test_import_and_version is unchanged;
test_engine_lifecycle keeps its shape and its intent (construct -> step -> read -> shutdown,
with the alive flag tracking it) but is re-expressed against the NeuroML engine: there is no
adjacency-list constructor, no set_input_neurons, no per-tick input vector and no
get_membrane_potentials any more. `is_alive()` is gone too -- the engine exposes only the
`alive` member, so the two redundant assertions nightly made collapse into one.
"""
import numpy as np
import pytest

import spikecorec

from conftest import RING_NEURON_COUNT, build_ring_engine


def test_import_and_version():
    assert isinstance(spikecorec.__version__, str)


def test_engine_lifecycle():
    engine = build_ring_engine()
    try:
        assert engine.alive
        assert engine.total_neuron_count == RING_NEURON_COUNT

        engine.step_simulation(0)

        membrane_potentials = engine.state_variable_values("v")
        assert membrane_potentials.shape == (RING_NEURON_COUNT,)
        assert membrane_potentials.dtype == np.float32
    finally:
        engine.shutdown()

    assert not engine.alive


def test_run_is_a_loop_over_step_simulation():
    """run(n) must be exactly n step_simulation calls, not an approximation of them.

    run() exists only to keep a multi-thousand-tick loop in C++ (and to drop the GIL while it
    runs), so the two paths have to agree bit for bit. They have separate call sites in
    bindings.cpp, which is enough for them to drift.
    """
    tick_count = 400

    stepped_engine = build_ring_engine()
    for tick in range(tick_count):
        stepped_engine.step_simulation(tick)
    stepped_membrane = stepped_engine.state_variable_values("v")
    stepped_engine.shutdown()

    run_engine = build_ring_engine()
    run_engine.run(tick_count)
    run_membrane = run_engine.state_variable_values("v")
    run_engine.shutdown()

    np.testing.assert_array_equal(stepped_membrane, run_membrane)


def test_a_negative_tick_is_refused():
    """A negative tick is an out-of-bounds device access, not merely a wrong answer.

    The generated kernel resolves the delay-ring row as `tick % ring_depth` in signed
    arithmetic, so tick -1 indexes network_inputs before its first element. Reachable from
    pure Python by a mistyped loop bound, so the binding refuses it.
    """
    engine = build_ring_engine()
    try:
        with pytest.raises(ValueError):
            engine.step_simulation(-1)
        with pytest.raises(ValueError):
            engine.run(10, first_tick=-1)
    finally:
        engine.shutdown()


def test_reading_an_undeclared_state_variable_is_refused():
    """Asking for a variable no cell type declares raises, rather than answering with slot 0.

    This is the one place the binding layer deliberately does NOT copy the engine: the
    engine's own recording path, handed a quantity its cell type does not declare, warns and
    records that neuron's FIRST state variable instead (src/core/engine.cpp,
    flat_cell_state_index). A read-back accessor that did the same would answer a question
    about 'asc1' with a membrane potential and look entirely plausible doing it.
    """
    engine = build_ring_engine()
    try:
        with pytest.raises(RuntimeError) as failure:
            engine.state_variable_values("thisVariableDoesNotExist")

        message = str(failure.value)
        assert "GLIF3Cell" in message
        # The message names what the type DOES declare, so the caller can see the typo.
        assert "asc1" in message
    finally:
        engine.shutdown()
