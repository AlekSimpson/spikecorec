"""The recordings a model's own <OutputFile>/<EventOutputFile> declarations produce.

New coverage: nightly's NmlNetworkRunner had no recording surface at all, so there was
nothing here to port. It matters because recording is a third of what the Python API is for
-- construct from a model, step it, read the recordings -- and because a recorder writing the
wrong column is invisible from inside the simulation. The frames are checked against the same
values read straight out of the engine's own buffers, which is the only thing that can
distinguish "recorded asc1" from "recorded whatever was in slot 0".
"""
import numpy as np

import spikecorec

from conftest import RING_NEURON_COUNT, write_ring_model


def test_output_columns_record_the_variables_they_name(tmp_path):
    membrane_path = tmp_path / "membrane.spire"
    output_elements = (
        '    <OutputFile id="membrane" fileName="{path}">\n'
        '      <OutputColumn id="v0" quantity="RingPop[0]/v"/>\n'
        '      <OutputColumn id="asc1_0" quantity="RingPop[0]/asc1"/>\n'
        '      <OutputColumn id="v1" quantity="RingPop[1]/v"/>\n'
        "    </OutputFile>\n"
    ).format(path=membrane_path)

    model_path = write_ring_model(tmp_path, output_elements, length="30ms")

    engine = spikecorec.SpikeEngine(model_path)
    assert engine.recording_output_filenames() == [str(membrane_path)]

    tick_count = engine.lifetime
    engine.run(tick_count)

    final_membrane = engine.state_variable_values("v")
    final_after_spike_current = engine.state_variable_values("asc1")

    # finish()ing the recorders is what flushes the buffered frames to disk.
    engine.shutdown()

    frames = spikecorec.read_spire_recording(str(membrane_path))

    assert frames.shape == (tick_count, 3)

    # Column order is selection order, and each column is the variable it named -- not that
    # neuron's first state variable, which is what the engine falls back to for a selection it
    # cannot resolve. asc1 is the column that can tell those apart: it is a current, several
    # orders of magnitude away from a voltage, and it is NOT slot 0.
    np.testing.assert_allclose(frames[-1, 0], final_membrane[0], rtol=1e-6)
    np.testing.assert_allclose(frames[-1, 1], final_after_spike_current[0], rtol=1e-6)
    np.testing.assert_allclose(frames[-1, 2], final_membrane[1], rtol=1e-6)

    # asc1 steps down on every spike and decays back towards zero, so a run in which neuron 0
    # fired has a strictly negative asc1 column somewhere. Without this the assertions above
    # would also hold for an asc1 column that was flat zero all run.
    assert np.any(frames[:, 1] < 0.0), "asc1 never moved, so the column proves nothing"


def test_event_output_file_records_every_neurons_spike_flag(tmp_path):
    spike_path = tmp_path / "spikes.spire"
    output_elements = (
        '    <EventOutputFile id="spikes" fileName="{path}" format="TIME_ID"/>\n'
    ).format(path=spike_path)

    model_path = write_ring_model(tmp_path, output_elements, length="30ms")

    engine = spikecorec.SpikeEngine(model_path)
    tick_count = engine.lifetime

    stepped_spike_counts = np.zeros(RING_NEURON_COUNT, dtype=np.int64)
    for tick in range(tick_count):
        engine.step_simulation(tick)
        stepped_spike_counts += engine.spike_flags()

    engine.shutdown()

    frames = spikecorec.read_spire_recording(str(spike_path))

    # An <EventOutputFile> with no <EventSelection> children records every neuron, one column
    # each, one frame per tick.
    assert frames.shape == (tick_count, RING_NEURON_COUNT)
    assert stepped_spike_counts.sum() > 0, "nothing spiked, so the file proves nothing"
    np.testing.assert_array_equal(frames.sum(axis=0).astype(np.int64), stepped_spike_counts)
