"""How much current a <pulseGenerator> actually injects, and over which ticks.

Not ported from anything: nightly had no stimulus test on the Python side. It exists because
stating what `amplitude`, `delay` and `duration` mean, which is what binding an API forces,
turned up an off-by-one in the delivery window (see the xfail below).

The model is deliberately not the ring fixture. It is one isolated cell with no projection
and a subthreshold amplitude, so the cell never spikes, no synapse ever fires, and dv/dt is
the injected current and the leak alone -- which makes the injected current recoverable
exactly:

    C dv/dt = gL (EL - v) + i    ->    i = C dv/dt - gL (EL - v)

with v taken before the step, because forward Euler is what the generated kernel runs.
"""
import os

import numpy as np
import pytest

import spikecorec

PROBE_NETWORK = """<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="PulseWindowProbe">
  <ComponentType name="ProbeCell" extends="baseCell">
    <Parameter name="C" dimension="capacitance"/>
    <Parameter name="gL" dimension="conductance"/>
    <Parameter name="EL" dimension="voltage"/>
    <Attachments name="synapses" type="basePointCurrent"/>
    <Exposure name="v" dimension="voltage"/>
    <Dynamics>
      <StateVariable name="v" dimension="voltage" exposure="v"/>
      <DerivedVariable name="iSyn" dimension="current" select="synapses[*]/i" reduce="add"/>
      <TimeDerivative variable="v" value="(gL * (EL - v) + iSyn) / C"/>
      <OnStart><StateAssignment variable="v" value="EL"/></OnStart>
    </Dynamics>
  </ComponentType>
  <ProbeCell id="probeCell" C="100pF" gL="10nS" EL="-70mV"/>
  <pulseGenerator id="drive" delay="{delay}" duration="{duration}" amplitude="{amplitude}"/>
  <network id="ProbeNet">
    <population id="ProbePop" component="probeCell" size="1"/>
    <explicitInput target="ProbePop[0]" input="drive"/>
  </network>
</neuroml>
"""

PROBE_CAPACITANCE = 100e-12
PROBE_LEAK_CONDUCTANCE = 10e-9
PROBE_RESTING_POTENTIAL = -70e-3
PROBE_STEP_SECONDS = 1e-4

# Well below the current needed to reach any threshold, and this cell declares none anyway.
PROBE_AMPLITUDE_AMPERES = 100e-12


def write_probe_model(directory, delay, duration, length):
    network_path = os.path.join(str(directory), "probe_network.nml")
    with open(network_path, "w") as network_file:
        network_file.write(PROBE_NETWORK.format(
            delay=delay, duration=duration, amplitude="100pA"))

    model_path = os.path.join(str(directory), "probe_model.xml")
    with open(model_path, "w") as model_file:
        model_file.write(
            "<Lems>\n"
            '  <Target component="probeSimulation"/>\n'
            '  <Include file="{network}"/>\n'
            '  <Simulation id="probeSimulation" length="{length}" step="0.1ms"'
            ' target="ProbeNet"/>\n'
            "</Lems>\n".format(network=network_path, length=length))
    return model_path


def injected_current_per_tick(model_path, tick_count):
    """The current delivered on each tick, recovered from the membrane trajectory."""
    engine = spikecorec.SpikeEngine(model_path)
    try:
        injected = np.zeros(tick_count)
        previous_potential = float(engine.state_variable_values("v")[0])

        for tick in range(tick_count):
            engine.step_simulation(tick)
            potential = float(engine.state_variable_values("v")[0])
            injected[tick] = (
                PROBE_CAPACITANCE * (potential - previous_potential) / PROBE_STEP_SECONDS
                - PROBE_LEAK_CONDUCTANCE * (PROBE_RESTING_POTENTIAL - previous_potential))
            previous_potential = potential

        return injected
    finally:
        engine.shutdown()


def test_pulse_generator_amplitude_and_start_tick(tmp_path):
    """The amplitude is what the model says, and it starts on the tick `delay` names.

    delay="10ms" at 0.1ms/tick is tick 100, and tick 99 must be silent.
    """
    model_path = write_probe_model(tmp_path, delay="10ms", duration="20ms", length="35ms")
    injected = injected_current_per_tick(model_path, 350)

    assert injected[99] == pytest.approx(0.0, abs=1e-13)
    assert injected[100] == pytest.approx(PROBE_AMPLITUDE_AMPERES, rel=1e-3)
    assert injected[200] == pytest.approx(PROBE_AMPLITUDE_AMPERES, rel=1e-3)


@pytest.mark.xfail(
    reason="ENGINE BUG, not a binding bug: the pulseGenerator delivery window is closed at "
           "BOTH ends, so a duration of N ticks injects on N+1 of them. "
           "src/nml/nml.cpp computes injection_end_tick = injection_start_tick + "
           "tick_count_from_seconds(duration) -- an EXCLUSIVE end index -- and "
           "create_event_stream (src/core/engine.cpp) then fills it with `tick <= "
           "last_event_tick`. Either the parser should subtract one or the fill should be "
           "`tick <`. The error is one tick of charge regardless of duration, so it is 0.05% "
           "on the 200ms pulse the other fixtures use and 10% on the 1ms pulse below. Both "
           "files are engine-owned; reported rather than fixed here.")
def test_pulse_generator_duration_is_the_duration_it_states(tmp_path):
    """A duration="1ms" pulse at 0.1ms/tick must inject on exactly 10 ticks, 100..109."""
    model_path = write_probe_model(tmp_path, delay="10ms", duration="1ms", length="20ms")
    injected = injected_current_per_tick(model_path, 200)

    delivering_ticks = np.flatnonzero(np.abs(injected) > 1e-13)

    assert delivering_ticks[0] == 100
    assert delivering_ticks[-1] == 109
    assert delivering_ticks.size == 10

    delivered_charge = injected.sum() * PROBE_STEP_SECONDS
    assert delivered_charge == pytest.approx(PROBE_AMPLITUDE_AMPERES * 1e-3, rel=1e-3)
