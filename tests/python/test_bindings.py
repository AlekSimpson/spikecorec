"""The Python bindings, checked against python/docs/API_REFERENCE.md. Run with `make test-python`."""

import _thread
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import spikecorec as spc
from spikecorec import nml, units

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

SINGLE_CELL_MODEL = """<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="SingleCell">
  <iafCell id="testCell" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude="90 pA"/>
  <network id="singleCellNetwork">
    <population id="cellPopulation" component="testCell" size="1"/>
    <explicitInput target="cellPopulation[0]" input="drive"/>
  </network>
</neuroml>
"""

WEIGHTED_MODEL = """<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Weights">
  <alphaCurrentSynapse id="syn" tau="5 ms" ibase="12 pA"/>
  <alphaCurrentSynapse id="otherSyn" tau="5 ms" ibase="-12 pA"/>
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <network id="weightNetwork">
    <population id="pop" component="c" size="16"/>
    <projection id="proj" presynapticPopulation="pop" postsynapticPopulation="pop" synapse="syn">
      <connectionWD id="0" preCellId="../pop[0]" postCellId="../pop[1]" weight="0.25" delay="1 ms"/>
      <connectionWD id="1" preCellId="../pop[0]" postCellId="../pop[2]" weight="1.75" delay="2 ms"/>
      <connectionWD id="2" preCellId="../pop[1]" postCellId="../pop[3]" weight="0.001" delay="3 ms"/>
    </projection>
  </network>
</neuroml>
"""

# Interspike interval at 90 pA (reset to threshold): tau ln((18 + 5) / (18 - 15)), tau = 20 ms.
SINGLE_CELL_INTERVAL = 20e-3 * math.log((18.0 + 5.0) / (18.0 - 15.0))


def write_lems(directory, model, network_id, length="200ms", step="0.05ms", children=""):
    (directory / "model.nml").write_text(model)
    lems = directory / "LEMS.xml"
    lems.write_text(
        '<Lems><Include file="Cells.xml"/><Include file="Synapses.xml"/><Include file="Inputs.xml"/>'
        '<Include file="Networks.xml"/><Include file="Simulation.xml"/><Include file="model.nml"/>'
        f'<Simulation id="sim1" length="{length}" step="{step}" target="{network_id}">{children}</Simulation>'
        '<Target component="sim1"/></Lems>')
    return str(lems)


@pytest.fixture(autouse=True)
def quiet_log():
    spc.set_log_level("warn")


# ── module ──────────────────────────────────────────────────────────────────────

def test_module_constants():
    assert spc.NEVER_SPIKED_TICK == -1
    assert spc.MAX_RANK_FLOAT4_STRIDE == 2048
    assert spc.DEFAULT_BRANCHING_FACTOR == 4


def test_topology_generators():
    assert sorted(spc.square_torus(3)[0]) == [1, 2, 3, 6]
    rows = spc.random_fixed_outdegree(4, 5, 7)
    assert len(rows) == 16
    assert all(len(row) == 5 and node not in row for node, row in enumerate(rows))
    assert rows == spc.random_fixed_outdegree(4, 5, 7)
    assert all(len(row) == 4 + 3 for row in spc.small_world_torus(4, 3, 1))


def test_units_module():
    assert units.parse_quantity("-60mV") == pytest.approx(-0.06)
    assert units.split_quantity("1.5e-3 mV") == (1.5e-3, "mV")
    assert units.unit_suffix_scale("pA") == pytest.approx(1e-12)
    assert units.seconds_to_ticks(2e-3, 1e-5) == 200
    with pytest.raises(ValueError):
        units.seconds_to_ticks(1.0, 0.0)
    with pytest.raises(ValueError):
        units.ms_to_ticks(float("nan"), 0.1)


# ── spikecorec.nml ──────────────────────────────────────────────────────────────

def test_nml_context_parses_a_model_without_a_gpu(tmp_path):
    context = nml.NML_Context()
    context.parse(write_lems(tmp_path, WEIGHTED_MODEL, "weightNetwork", "10ms", "0.1ms"))

    simulation = context.simulation
    assert simulation.step_dt == pytest.approx(1e-4)
    assert simulation.total_tick_count == 100
    assert simulation.total_neuron_count == 16
    assert simulation.total_edge_count == 3
    assert simulation.maximum_edge_delay == 30
    assert [synapse.id for synapse in simulation.synapse_instances] == ["syn"]
    assert simulation.population_base_indices["weightNetwork/pop"] == 0

    cell = context.find_instance("c")
    assert cell.value_or("leakReversal") == "-65 mV"
    assert context.resolve_quantity(cell.value_or("leakReversal")) == pytest.approx(-0.065)
    assert context.is_instance_of(cell, "iafCell")
    assert context.find_instance("no_such_instance") is None
    assert context.resolve_path("pop[3]/v") == 3
    assert context.neuron_index_of(3) == 3
    assert context.get_cell_state_size() == 16
    with pytest.raises(ValueError):
        context.get_population_size(None)

    edges = simulation.network_data.edge_arrays()
    assert list(edges["child"]) == [1, 2, 3]
    assert list(edges["delay_ticks"]) == [10, 20, 30]


def test_component_types_and_their_dynamics():
    context = nml.NML_Context()
    context.parse(str(FIXTURES / "nml" / "LEMS_glif_family.xml"))

    iaf = context.component_types["iafCell"]
    assert iaf.state_variable_names == ["v"]
    assert iaf.extends.name == "baseIafCapCell"
    assert iaf.find_declaration("var:leakReversal").body.get_attribute("name") == "leakReversal"
    assert iaf.find_declaration("var:no_such_name") is None

    tags = [entry.source_tag for entry in iaf.dynamics]
    assert nml.NML_Declaration_Type.TimeDerivative in tags
    reset = next(entry for entry in iaf.dynamics if entry.source_tag == nml.NML_Declaration_Type.StateAssignment)
    assert (reset.target, reset.expression, reset.condition) == ("v", "reset", "v .gt. thresh")

    populations = nml.network_populations(context)
    assert len(populations) == 5
    assert nml.population_cell(context, populations[0]).component_type.name == "GLIF1Cell"


def test_structure_entries(tmp_path):
    main_file = tmp_path / "root.xml"
    main_file.write_text("<Lems/>")
    context = nml.NML_Context()
    context.parse(str(main_file))
    connection = context.component_types["spike"].structure[2]
    assert connection.source_tag == nml.NML_Declaration_Type.EventConnection
    assert (connection.source, connection.target) == ("a", "b")
    assert context.component_types["timedSynapticInput"].structure[0].component == "synapse"


def test_evaluate_lems_and_the_random_generator():
    assert nml.evaluate_lems("log(2.718281828459045)", {}) == pytest.approx(1.0)
    assert nml.evaluate_lems("H(0)", {}) == 0.0
    assert nml.evaluate_lems("2 ^ 3 ^ 2", {}) == 64.0
    assert nml.evaluate_lems("x .gt. 1", {"x": 2.0}) == 1.0

    generator = nml.Random_Generator(3)
    draws = [nml.evaluate_lems("random(2)", {}, random_generator=generator) for _ in range(2000)]
    assert all(0.0 < draw < 2.0 for draw in draws)
    assert np.mean(draws) == pytest.approx(1.0, abs=0.06)
    with pytest.raises(RuntimeError):
        nml.evaluate_lems("random(1)", {})

    first, second = nml.Random_Generator(9), nml.Random_Generator(9)
    assert [first.uniform() for _ in range(5)] == [second.uniform() for _ in range(5)]


def test_schema_validation(tmp_path):
    invalid = tmp_path / "invalid.nml"
    invalid.write_text('<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="bad">'
                       '<thisTagDoesNotExistInSchema id="x"/></neuroml>')
    valid = tmp_path / "valid.nml"
    valid.write_text(SINGLE_CELL_MODEL)

    context = nml.NML_Context()
    assert context.validate_lems_schema(str(valid)), context.last_schema_validation_errors
    assert not context.validate_lems_schema(str(invalid))
    assert "thisTagDoesNotExistInSchema" in context.last_schema_validation_errors


# ── Spike_Engine ─────────────────────────────────────────────────────────────────

def test_engine_runs_a_single_cell(tmp_path):
    engine = spc.Spike_Engine(write_lems(tmp_path, SINGLE_CELL_MODEL, "singleCellNetwork", "500ms"))
    assert engine.total_neuron_count == 1
    assert engine.lifetime == 10000
    assert engine.read_state_variable(0, "v") == pytest.approx(-0.065, abs=1e-7)
    assert engine.last_spiked[0] == spc.NEVER_SPIKED_TICK

    engine.run()
    times, neurons = engine.recorded_spikes
    assert len(times) == len(neurons) == engine.spike_counts_per_neuron[0] >= 4
    assert times[3] - times[2] == pytest.approx(SINGLE_CELL_INTERVAL, abs=2e-4)
    assert engine.mean_firing_rate_hertz() == pytest.approx(1.0 / SINGLE_CELL_INTERVAL, abs=1.0)
    assert engine.fraction_of_neurons_that_spiked() == 1.0
    with pytest.raises(RuntimeError):
        engine.read_state_variable(0, "no_such_variable")


def test_state_and_device_buffers_are_copies(tmp_path):
    engine = spc.Spike_Engine(write_lems(tmp_path, SINGLE_CELL_MODEL, "singleCellNetwork", "20ms"))
    for tick in range(100):
        engine.step_simulation(tick)
    voltages = engine.state_variable_array("v")
    assert voltages.dtype == np.float32 and voltages.shape == (1,)
    assert voltages[0] == pytest.approx(engine.read_state_variable(0, "v"))
    assert engine.cell_state.shape == (engine.context.get_cell_state_size(),)
    assert engine.network_inputs.shape == (2, 1)
    assert engine.event_arrival_count.shape == (0, 1)   # no cell type has an OnEvent
    # One pulse input on one cell: its weight, then its state (i).
    assert engine.input_entry_count == 1
    assert list(engine.input_row_start) == [0, 1]
    assert list(engine.input_entry_prototype) == [0]
    assert engine.input_values.shape == (1, engine.input_value_stride)
    assert engine.input_values[0, 0] == 1.0
    assert engine.spike_history.shape == (engine.spike_history_row_count, 1)
    assert "master_step" in engine.master_kernel_source

    counts = engine.spike_counts_per_neuron
    engine.shutdown()
    assert counts.shape == (1,)              # copies outlive the engine's buffers
    with pytest.raises(RuntimeError):
        _ = engine.cell_state


COUNTING_MODEL = """<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="CellEvents">
  <ComponentType name="countingCell" extends="baseCellMembPot">
    <EventPort name="in" direction="in"/>
    <Attachments name="synapses" type="basePointCurrent"/>
    <Dynamics>
      <StateVariable name="v" dimension="voltage" exposure="v"/>
      <StateVariable name="received" dimension="none"/>
      <OnEvent port="in">
        <StateAssignment variable="received" value="received + 1"/>
      </OnEvent>
    </Dynamics>
  </ComponentType>
  <alphaCurrentSynapse id="syn" tau="5 ms" ibase="12 pA"/>
  <iafCell id="driven" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <countingCell id="counter"/>
  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude="90 pA"/>
  <network id="eventNetwork">
    <population id="sources" component="driven" size="1"/>
    <population id="targets" component="counter" size="1"/>
    <projection id="projection" presynapticPopulation="sources" postsynapticPopulation="targets" synapse="syn">
      <connectionWD id="0" preCellId="../sources[0]" postCellId="../targets[0]" weight="1" delay="1 ms"/>
    </projection>
    <explicitInput target="sources[0]" input="drive"/>
  </network>
</neuroml>
"""


def test_a_cells_on_event_runs_once_per_arriving_spike(tmp_path):
    engine = spc.Spike_Engine(write_lems(tmp_path, COUNTING_MODEL, "eventNetwork", "200ms", "0.1ms"))
    assert engine.event_arrival_count.dtype == np.uint32
    assert engine.event_arrival_count.shape == (2, 2)

    engine.run()
    times, neurons = engine.recorded_spikes
    # Each source spike reaches the counter 1 ms later and is counted on the tick after.
    delivered = [time for time, neuron in zip(times, neurons)
                 if neuron == 0 and round(time / engine.step_dt) + 11 < engine.lifetime]
    assert len(delivered) >= 3
    assert engine.read_state_variable(1, "received") == len(delivered)


def test_weighted_arrivals_reach_their_target_exactly_a_delay_after_the_spike():
    # A trigger fires at 10 ms; a 0.1 ms and then a 1.5 ms delta synapse each fire the next neuron.
    engine = spc.Spike_Engine(str(FIXTURES / "nml" / "LEMS_brunel_model_a_effective_delay.xml"))
    assert engine.context.simulation.minimum_edge_delay == 1
    assert engine.context.simulation.maximum_edge_delay == 15
    assert engine.network_inputs.shape == (5, 3)
    assert engine.weights.matrix_count == 2
    with pytest.raises(RuntimeError, match="keeps no 'accumulated_jump' per connection"):
        engine.synapse_state_variable_plane("suprathreshold_delta_synapse", "accumulated_jump")

    engine.run()
    times, neurons = engine.recorded_spikes
    ticks = np.rint(np.asarray(times) / engine.step_dt).astype(int)
    assert list(zip(neurons.tolist(), ticks.tolist())) == [(0, 100), (1, 101), (2, 116)]
    # The model A neuron reads its total only less applied_jump, so its running sum holds what it
    # has not applied yet: nothing, once the one 25 mV jump that reached it is applied.
    assert list(engine.network_inputs[4][1:]) == [0.0, 0.0]
    assert [engine.read_state_variable(neuron, "applied_jump") for neuron in (1, 2)] == pytest.approx([0.025, 0.025], rel=1e-3)


def test_written_spikes_match_the_recorded_ones(tmp_path):
    spike_file = tmp_path / "events.dat"
    children = (f'<EventOutputFile id="spikes" fileName="{tmp_path / "declared.dat"}" format="TIME_ID">'
                '<EventSelection id="0" select="cellPopulation[0]" eventPort="spike"/></EventOutputFile>')
    engine = spc.Spike_Engine(write_lems(tmp_path, SINGLE_CELL_MODEL, "singleCellNetwork", "200ms", children=children))
    engine.run()
    engine.write_spike_file(str(spike_file))
    engine.write_recordings()

    times, _ = engine.recorded_spikes
    written = [float(line.split()[0]) for line in spike_file.read_text().splitlines() if line.strip()]
    declared = [float(line.split()[0]) for line in (tmp_path / "declared.dat").read_text().splitlines() if line.strip()]
    assert written == pytest.approx(list(times))
    assert declared == pytest.approx(list(times))


def test_an_engine_runs_on_a_worker_thread(tmp_path):
    # Metal hands back autoreleased objects; the worker's pool drains them when the thread ends.
    engine = spc.Spike_Engine(write_lems(tmp_path, SINGLE_CELL_MODEL, "singleCellNetwork", "20ms"))
    worker = threading.Thread(target=lambda: [engine.step_simulation(tick) for tick in range(10)])
    worker.start()
    worker.join()
    worker = threading.Thread(target=lambda: [engine.step_simulation(tick) for tick in range(10, 20)])
    worker.start()
    worker.join()
    assert engine.read_state_variable(0, "v") > -0.065


def test_results_read_during_a_run_on_another_thread_are_consistent(tmp_path):
    children = ('<OutputFile id="trace" fileName="trace.dat">'
                '<OutputColumn id="v" quantity="cellPopulation[0]/v"/></OutputFile>')
    engine = spc.Spike_Engine(write_lems(tmp_path, SINGLE_CELL_MODEL, "singleCellNetwork", "1000ms", "0.025ms",
                                         children))
    runner = threading.Thread(target=engine.run)
    runner.start()
    snapshots = []
    while runner.is_alive() and len(snapshots) < 200:
        snapshots.append(engine.recorded_traces)
        time.sleep(0.001)
    runner.join()

    final = engine.recorded_traces
    assert final.shape == (engine.lifetime, 1)
    for snapshot in snapshots:
        assert snapshot.shape[1] == 1
        assert np.array_equal(snapshot, final[:snapshot.shape[0]])


def test_run_calls_back_after_each_tick_and_stops_on_an_exception(tmp_path):
    engine = spc.Spike_Engine(write_lems(tmp_path, SINGLE_CELL_MODEL, "singleCellNetwork", "5ms"))
    ticks = []
    engine.run(after_each_tick=ticks.append)
    assert ticks == list(range(engine.lifetime))

    stopped = spc.Spike_Engine(write_lems(tmp_path, SINGLE_CELL_MODEL, "singleCellNetwork", "5ms"))

    def stop_at_tick_five(tick):
        if tick == 5:
            raise ValueError("stop here")

    with pytest.raises(ValueError, match="stop here"):
        stopped.run(after_each_tick=stop_at_tick_five)


def test_ctrl_c_stops_a_run(tmp_path):
    engine = spc.Spike_Engine(write_lems(tmp_path, SINGLE_CELL_MODEL, "singleCellNetwork", "100000ms", "0.025ms"))
    ticks = []
    interrupter = threading.Timer(0.3, _thread.interrupt_main)
    interrupter.start()
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        engine.run(after_each_tick=ticks.append)
    interrupter.join()
    assert time.monotonic() - started < 5.0
    assert 0 < len(ticks) < engine.lifetime


def test_ticks_run_once_in_order(tmp_path):
    engine = spc.Spike_Engine(write_lems(tmp_path, SINGLE_CELL_MODEL, "singleCellNetwork", "5ms"))
    with pytest.raises(RuntimeError, match="next is 0"):
        engine.step_simulation(3)
    for tick in range(10):
        engine.step_simulation(tick)
    assert engine.ticks_stepped == 10
    engine.run()
    assert engine.ticks_stepped == engine.lifetime
    with pytest.raises(RuntimeError, match="new Spike_Engine"):
        engine.run()


def test_running_after_shutdown_raises(tmp_path):
    engine = spc.Spike_Engine(write_lems(tmp_path, SINGLE_CELL_MODEL, "singleCellNetwork", "5ms"))
    weights = engine.weights
    engine.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        engine.step_simulation(0)
    with pytest.raises(RuntimeError, match="shut down"):
        engine.run()
    with pytest.raises(RuntimeError, match="no storage"):
        weights.resize_delta_capacity(16)


def test_a_topology_constructor_splits_the_edges_into_shares(tmp_path):
    lems = write_lems(tmp_path, WEIGHTED_MODEL, "weightNetwork", "10ms", "0.1ms")
    adjacency = spc.random_fixed_outdegree(4, 5, 3)
    engine = spc.Spike_Engine(lems, adjacency, ["syn", "otherSyn"], synapse_proportions=[3.0, 1.0],
                             connection_weight=2.0, connection_delay_seconds=1e-3)

    weights = engine.weights
    assert weights.total_edge_count == 80
    prototypes = [weights.get_edge_synapse_prototype(source, target)
                  for source in range(16) for target in adjacency[source]]
    assert prototypes == [0] * 60 + [1] * 20
    assert weights.get(0, adjacency[0][0]) == pytest.approx(2.0, abs=weights.fit_tolerance * 2.0)
    assert weights.get_edge_delay_ticks(0, adjacency[0][0]) == 10

    with pytest.raises(RuntimeError, match="noSuchSynapse"):
        spc.Spike_Engine(lems, adjacency, "noSuchSynapse")


def test_a_topology_constructor_reads_connection_arrays(tmp_path):
    lems = write_lems(tmp_path, WEIGHTED_MODEL, "weightNetwork", "10ms", "0.1ms")
    adjacency = spc.random_fixed_outdegree(4, 5, 3)
    source_neurons = np.array([source for source in range(16) for target in adjacency[source]], dtype=np.uint32)
    target_neurons = np.array([target for source in range(16) for target in adjacency[source]], dtype=np.uint32)
    order = np.random.default_rng(7).permutation(source_neurons.size)
    source_neurons, target_neurons = source_neurons[order], target_neurons[order]
    synapse_indices = (source_neurons % 2).astype(np.int32)
    weights = np.where(target_neurons < 8, 2.0, -3.0)  # float64, converted to a float32 copy
    delays_seconds = np.where(source_neurons < 8, 1e-3, 2.5e-3).astype(np.float32)
    engine = spc.Spike_Engine(lems, ["syn", "otherSyn"], source_neurons, target_neurons, synapse_indices,
                             weights, delays_seconds)

    matrix = engine.weights
    assert matrix.total_edge_count == 80
    for source, target in zip(source_neurons[:10].tolist(), target_neurons[:10].tolist()):
        assert matrix.get_edge_synapse_prototype(source, target) == source % 2
        expected_weight = 2.0 if target < 8 else -3.0
        assert matrix.get(source, target) == pytest.approx(expected_weight, abs=matrix.fit_tolerance * 3.0)
        assert matrix.get_edge_delay_ticks(source, target) == (10 if source < 8 else 25)

    with pytest.raises(ValueError, match="weights"):
        spc.Spike_Engine(lems, ["syn", "otherSyn"], source_neurons, target_neurons, synapse_indices,
                         weights[:-1], delays_seconds)
    repeated = np.append(source_neurons, source_neurons[0]), np.append(target_neurons, target_neurons[0])
    with pytest.raises(RuntimeError, match="given twice"):
        spc.Spike_Engine(lems, ["syn", "otherSyn"], repeated[0], repeated[1], np.append(synapse_indices, 0),
                         np.append(weights, 1.0), np.append(delays_seconds, 1e-3))


# ── Weight_Matrix ────────────────────────────────────────────────────────────────

def test_weight_matrix_reads_every_plane(tmp_path):
    engine = spc.Spike_Engine(write_lems(tmp_path, WEIGHTED_MODEL, "weightNetwork", "10ms", "0.1ms"))
    weights = engine.weights

    # fit_tolerance bounds the error against the larger of the value and the plane's RMS.
    weight_rms = math.sqrt((0.25 ** 2 + 1.75 ** 2 + 0.001 ** 2) / 3.0)
    assert weights.get(0, 1) == pytest.approx(0.25, abs=weights.fit_tolerance * weight_rms)
    assert weights.get_for_matrix(1, 3, weights.WEIGHT_PLANE) == pytest.approx(0.001, abs=weights.fit_tolerance * weight_rms)
    delay_rms = math.sqrt((10.0 ** 2 + 20.0 ** 2 + 30.0 ** 2) / 3.0)
    for read, declared in zip(weights.neighbor_weights_for_matrix(weights.DELAY_PLANE), [10.0, 20.0, 30.0]):
        assert read == pytest.approx(declared, abs=weights.fit_tolerance * max(declared, delay_rms))
    assert [weights.get_edge_delay_ticks(0, 1), weights.get_edge_delay_ticks(1, 3)] == [10, 30]
    assert weights.edge_ordinal(0, 2) == 1
    assert weights.edge_ordinal(3, 0) is None
    assert list(weights.get_neighbors(0)) == [1, 2]
    assert list(weights.get_predecessors(3)) == [1]
    assert list(weights.edge_row_offset[:3]) == [0, 2, 3]

    # alphaCurrentSynapse's I and J follow weight and delay, starting at zero.
    assert weights.matrix_count == weights.FIRST_STATE_VARIABLE_PLANE + 2
    current_plane = engine.synapse_state_variable_plane("syn", "I")
    assert list(weights.neighbor_weights_for_matrix(current_plane)) == [0.0, 0.0, 0.0]
    assert weights.U_matrix.shape == (16, weights.rank)
    assert weights.coefficients.shape == (weights.matrix_count, weights.rank)
    assert max(weights.measured_fit_error) <= weights.fit_tolerance


@pytest.mark.skip(reason="plasticity tests are disabled for now: this one updates the weight plane")
def test_weight_matrix_updates_and_refits(tmp_path):
    engine = spc.Spike_Engine(write_lems(tmp_path, WEIGHTED_MODEL, "weightNetwork", "10ms", "0.1ms"))
    weights = engine.weights
    assert spc.Weight_Matrix.DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION == pytest.approx(0.2)
    assert spc.Weight_Matrix.MASKED_ALTERNATING_LEAST_SQUARES_STALL_WINDOW_SWEEP_COUNT == 10

    weights.accumulate_edge_delta(weights.WEIGHT_PLANE, 0, 1, 0.5)
    assert weights.get(0, 1) == pytest.approx(0.75)
    ordinals, deltas = weights.sparse_deltas(weights.WEIGHT_PLANE)
    assert list(ordinals) == [0] and list(deltas) == pytest.approx([0.5])
    assert weights.sparse_delta_occupancy_fraction() == pytest.approx(1.0 / 3.0)
    assert weights.is_refit_due(0.3) and not weights.is_refit_due(0.5)

    delay_before = weights.neighbor_weights_for_matrix(weights.DELAY_PLANE)
    weights.refit()
    assert list(weights.sparse_delta_entry_count) == [0] * weights.matrix_count
    # fit_tolerance bounds the error against the larger of the value and the plane's RMS.
    plane_rms = math.sqrt((0.75 ** 2 + 1.75 ** 2 + 0.001 ** 2) / 3.0)
    assert weights.get(0, 1) == pytest.approx(0.75, abs=weights.fit_tolerance * max(0.75, plane_rms))
    # Only the weight plane had updates, so the delay plane is untouched.
    assert np.array_equal(weights.neighbor_weights_for_matrix(weights.DELAY_PLANE), delay_before)


def test_refit_settings(tmp_path):
    engine = spc.Spike_Engine(write_lems(tmp_path, WEIGHTED_MODEL, "weightNetwork", "10ms", "0.1ms"))
    assert engine.refit_occupancy_threshold_fraction == pytest.approx(0.2)
    assert engine.minimum_ticks_between_refits == 1000
    assert engine.fit_tolerance == pytest.approx(spc.Spike_Engine.FIT_TOLERANCE_STANDARD)
    assert spc.Spike_Engine.FIT_TOLERANCE_PRECISE < spc.Spike_Engine.FIT_TOLERANCE_STANDARD

    engine.set_refit_occupancy_threshold_fraction(0.5)
    assert engine.weights.sparse_delta_capacity == engine.weights.delta_capacity_for_threshold(0.5)


# ── recording ───────────────────────────────────────────────────────────────────

def test_spire_files_round_trip(tmp_path):
    frames = np.arange(12, dtype=np.float32).reshape(4, 3)
    raw_path = str(tmp_path / "raw.spire")
    writer = spc.Spire_Writer(raw_path, 3)
    for frame in frames:
        writer.write_frame(frame)
    del writer
    assert np.array_equal(spc.read_spire_recording(raw_path), frames)

    reader = spc.Spire_Reader(raw_path)
    assert reader.neuron_count == 3
    assert np.array_equal(reader.read_frame(), frames[0])

    # A live writer can be flushed and closed, so the file reads back without waiting for del.
    live_path = str(tmp_path / "live.spire")
    live_writer = spc.Spire_Writer(live_path, 3)
    live_writer.write_frame(frames[0])
    live_writer.flush()
    assert np.array_equal(spc.read_spire_recording(live_path), frames[:1])
    live_writer.write_frame(frames[1])
    live_writer.close()
    live_writer.close()
    assert np.array_equal(spc.read_spire_recording(live_path), frames[:2])
    with pytest.raises(RuntimeError):
        live_writer.write_frame(frames[2])
    with spc.Spire_Writer(live_path, 3) as context_writer:
        context_writer.write_frame(frames[3])
    assert np.array_equal(spc.read_spire_recording(live_path), frames[3:])

    compressed_path = str(tmp_path / "frames.spire.gz")
    recorder = spc.Simulation_Recorder(compressed_path, 3)
    for frame in frames:
        recorder.record_frame(frame)
    recorder.finish()
    assert np.array_equal(spc.read_spire_recording(compressed_path), frames)
    assert spc.resolve_spire_compression(compressed_path)[0] == spc.Spire_Compression.Gzip
