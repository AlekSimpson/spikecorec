"""Minimal smoke test for the compiled _spikecorec extension.

Exercises the pybind11 binding layer end to end: import the module,
construct a SpikeEngine, step it, and read back a couple of bound
accessors — enough to catch a bindings.cpp/engine.h mismatch like the one
fixed in ticket #139 (is_alive/alive, GpuPointer::pointer).
"""
import spikecorec


def test_import_and_version():
    assert isinstance(spikecorec.__version__, str)


def test_engine_lifecycle():
    network = spikecorec.square_torus(4)
    engine = spikecorec.SpikeEngine(network, shape=[4, 4], rank=2)

    assert engine.is_alive()
    assert engine.alive

    engine.set_input_neurons([0, 1])
    engine.step_simulation([2.0, 2.0], tick=0)

    membrane_potentials = engine.get_membrane_potentials()
    assert len(membrane_potentials) == engine.neuron_count

    engine.shutdown()
    assert not engine.is_alive()
    assert not engine.alive
